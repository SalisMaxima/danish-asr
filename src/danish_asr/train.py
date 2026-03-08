"""ASR training pipeline with PyTorch Lightning."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from danish_asr.metrics import compute_cer, compute_wer
from danish_asr.model import build_model
from danish_asr.utils import get_device

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = str(_PROJECT_ROOT / "configs")

profiling_dir = _PROJECT_ROOT / "outputs" / "profiling"
profiling_dir.mkdir(parents=True, exist_ok=True)


class ASRLitModel(pl.LightningModule):
    """Lightning module for ASR training (CTC and seq2seq)."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model = build_model(cfg.model)
        self.mode = "ctc" if cfg.model.name == "wav2vec2_asr" else "seq2seq"
        self.save_hyperparameters(ignore=["model"])

        # Load processor/tokenizer
        self.processor, self.tokenizer = self._build_processor()

        # Gradient checkpointing
        if cfg.get("hardware", {}).get("gradient_checkpointing", False):
            self._enable_gradient_checkpointing()

        # Validation accumulators
        self._val_predictions: list[str] = []
        self._val_references: list[str] = []
        self._test_predictions: list[str] = []
        self._test_references: list[str] = []

    def _build_processor(self):
        model_name = self.cfg.model.get("model_name", "")
        revision = self.cfg.model.get("revision")
        # Security note: revision can be None (latest) or pinned via config.
        # Using standard HF checkpoints for research baselines is intentional.
        kwargs = {"revision": revision} if revision is not None else {}
        if self.mode == "ctc":
            from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, **kwargs)  # nosec B615
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name, **kwargs)  # nosec B615
            processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
            return processor, tokenizer
        from transformers import WhisperProcessor, WhisperTokenizer

        processor = WhisperProcessor.from_pretrained(model_name, **kwargs)  # nosec B615
        tokenizer = WhisperTokenizer.from_pretrained(model_name, **kwargs)  # nosec B615
        return processor, tokenizer

    def _enable_gradient_checkpointing(self):
        inner = self.model.model if hasattr(self.model, "model") else self.model
        if hasattr(inner, "gradient_checkpointing_enable"):
            inner.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def _predict_batch(self, batch: dict) -> tuple[torch.Tensor, list[str]]:
        if self.mode == "ctc":
            outputs = self.model(
                input_values=batch["input_values"],
                attention_mask=batch.get("attention_mask"),
                labels=batch["labels"],
            )
            logits = outputs["logits"]
            pred_ids = torch.argmax(logits, dim=-1)
            predictions = self.processor.batch_decode(pred_ids)
            return outputs["loss"], predictions

        outputs = self.model(
            input_features=batch["input_features"],
            labels=batch["labels"],
        )
        inner = self.model.model if hasattr(self.model, "model") else self.model
        generated_ids = inner.generate(
            batch["input_features"],
            language=self.cfg.model.get("language", "da"),
            task="transcribe",
            max_new_tokens=225,
        )
        predictions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return outputs["loss"], predictions

    def _finalize_epoch_metrics(self, predictions: list[str], references: list[str], prefix: str) -> None:
        if not predictions:
            return

        wer = compute_wer(predictions, references)
        cer = compute_cer(predictions, references)
        self.log(f"{prefix}_wer", wer, prog_bar=(prefix == "val"))
        self.log(f"{prefix}_cer", cer, prog_bar=(prefix == "val"))
        logger.info(f"{prefix.upper()} WER: {wer:.4f}, CER: {cer:.4f}")

    def training_step(self, batch, batch_idx: int):
        if self.mode == "ctc":
            outputs = self.model(
                input_values=batch["input_values"],
                attention_mask=batch.get("attention_mask"),
                labels=batch["labels"],
            )
        else:
            outputs = self.model(
                input_features=batch["input_features"],
                labels=batch["labels"],
            )
        loss = outputs["loss"]
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        loss, predictions = self._predict_batch(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        self._val_predictions.extend(predictions)
        self._val_references.extend(batch["text"])
        return loss

    def test_step(self, batch, batch_idx: int):
        loss, predictions = self._predict_batch(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True)

        self._test_predictions.extend(predictions)
        self._test_references.extend(batch["text"])
        return loss

    def on_validation_epoch_end(self):
        self._finalize_epoch_metrics(self._val_predictions, self._val_references, "val")
        self._val_predictions.clear()
        self._val_references.clear()

    def on_test_epoch_end(self):
        self._finalize_epoch_metrics(self._test_predictions, self._test_references, "test")
        self._test_predictions.clear()
        self._test_references.clear()

    def configure_optimizers(self):
        opt_cfg = self.cfg.train.optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
            betas=tuple(opt_cfg.betas),
        )

        sched_cfg = self.cfg.train.scheduler
        warmup_steps = sched_cfg.get("warmup_steps", 500)

        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-7 / opt_cfg.lr,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }


def configure_logging(output_dir: str) -> None:
    """Configure loguru for file and console logging."""
    logger.remove()
    log_path = Path(output_dir) / "training.log"
    logger.add(
        log_path,
        level="DEBUG",
        rotation="100 MB",
        retention=5,
        compression="gz",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    )
    logger.add(
        sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}"
    )
    logger.info(f"Logging configured. Logs saved to {log_path}")


def set_seed(seed: int, device: torch.device) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def train_model(cfg: DictConfig, output_dir: str, wandb_logger: WandbLogger | None = None) -> str:
    """Train ASR model using PyTorch Lightning."""
    output_path = Path(output_dir)
    train_cfg = cfg.train
    hw_cfg = cfg.get("hardware", {})
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    set_seed(cfg.seed, get_device())
    pl.seed_everything(cfg.seed, workers=True)

    # Model
    lit_model = ASRLitModel(cfg)
    total_params = sum(p.numel() for p in lit_model.model.parameters())
    trainable_params = sum(p.numel() for p in lit_model.model.parameters() if p.requires_grad)
    logger.info(f"Model: {cfg.model.name} | Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # Data
    from danish_asr.data import CoRalDataModule

    datamodule = CoRalDataModule(cfg.data)
    datamodule.set_processor_and_tokenizer(lit_model.processor, lit_model.tokenizer)

    # Apply hardware overrides to datamodule
    if hw_cfg.get("batch_size") is not None:
        datamodule.batch_size = hw_cfg.batch_size
    if hw_cfg.get("num_workers") is not None:
        datamodule.num_workers = hw_cfg.num_workers
    if hw_cfg.get("max_duration") is not None:
        datamodule.max_duration = hw_cfg.max_duration

    datamodule.setup(stage="fit")

    # Callbacks
    callbacks = []
    cb_cfg = train_cfg.callbacks
    ckpt_cfg = cb_cfg.checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_path),
        filename=ckpt_cfg.get("filename", "best_model"),
        monitor=ckpt_cfg.monitor,
        mode=ckpt_cfg.mode,
        save_top_k=ckpt_cfg.save_top_k,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    es_cfg = cb_cfg.early_stopping
    callbacks.append(
        EarlyStopping(
            monitor=es_cfg.monitor,
            patience=es_cfg.patience,
            mode=es_cfg.mode,
            min_delta=0.001,
            verbose=True,
        )
    )

    callbacks.append(LearningRateMonitor(logging_interval="step"))

    # Trainer
    trainer_cfg = train_cfg.trainer
    trainer = pl.Trainer(
        default_root_dir=str(output_path),
        max_epochs=trainer_cfg.max_epochs,
        accelerator=hw_cfg.get("accelerator", "auto"),
        devices=hw_cfg.get("devices", "auto"),
        precision=hw_cfg.get("precision", "bf16-mixed"),
        gradient_clip_val=trainer_cfg.gradient_clip_val,
        accumulate_grad_batches=trainer_cfg.get("accumulate_grad_batches", 1),
        log_every_n_steps=trainer_cfg.get("log_every_n_steps", 10),
        val_check_interval=trainer_cfg.get("val_check_interval", 0.5),
        callbacks=callbacks,
        logger=wandb_logger,
        enable_checkpointing=True,
    )

    trainer.fit(lit_model, datamodule=datamodule)

    # Test
    best_ckpt_path = checkpoint_callback.best_model_path
    if best_ckpt_path:
        trainer.test(ckpt_path=best_ckpt_path, datamodule=datamodule)

    # Save final model
    final_model_path = output_path / "model.pt"
    torch.save(lit_model.model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

    # Log artifact
    if wandb_logger is not None:
        artifact = wandb.Artifact(
            name=f"{cfg.experiment_name}_model",
            type="model",
            metadata={"model_name": cfg.model.name, "num_params": total_params},
        )
        if best_ckpt_path:
            artifact.add_file(best_ckpt_path)
        artifact.add_file(str(final_model_path))
        wandb.log_artifact(artifact)

    return str(final_model_path)


@hydra.main(config_path=_CONFIG_PATH, config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> None:
    """Train an ASR model (Hydra entry point)."""
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    configure_logging(output_dir)
    device = get_device()
    logger.info(f"Training on {device}")

    wandb_cfg = cfg.wandb
    wandb_logger = WandbLogger(
        project=wandb_cfg.project,
        entity=wandb_cfg.get("entity"),
        name=f"{cfg.experiment_name}_{cfg.model.name}",
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=list(wandb_cfg.get("tags", [])),
        mode=wandb_cfg.get("mode", "online"),
        save_dir=output_dir,
        job_type="train",
    )

    try:
        model_path = train_model(cfg, output_dir, wandb_logger)
        logger.info(f"Training complete. Model saved to {model_path}")
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        raise
    finally:
        wandb.finish()


if __name__ == "__main__":
    train()
