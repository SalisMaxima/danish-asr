"""Training pipeline with PyTorch Lightning."""

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

from danish_asr.losses import build_loss
from danish_asr.model import build_model
from danish_asr.utils import get_device

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = str(_PROJECT_ROOT / "configs")

profiling_dir = _PROJECT_ROOT / "outputs" / "profiling"
profiling_dir.mkdir(parents=True, exist_ok=True)


class LitModel(pl.LightningModule):
    """Lightning module wrapping the model."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model = build_model(cfg)
        self.criterion = build_loss(cfg)
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _compute_accuracy(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = torch.argmax(logits, dim=1)
        return (preds == targets).float().mean()

    def training_step(self, batch, batch_idx: int):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = self._compute_accuracy(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = self._compute_accuracy(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx: int):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = self._compute_accuracy(y_hat, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        opt_cfg = self.cfg.train.optimizer
        sched_cfg = self.cfg.train.scheduler
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
            betas=tuple(opt_cfg.betas),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.train.max_epochs,
            eta_min=sched_cfg.eta_min,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1}}


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
    """Train using PyTorch Lightning with full MLOps features."""
    output_path = Path(output_dir)
    train_cfg = cfg.train
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    set_seed(cfg.seed, get_device())
    pl.seed_everything(cfg.seed, workers=True)

    # Data
    from danish_asr.data import CoRalDataModule

    datamodule = CoRalDataModule(cfg.data)
    datamodule.setup(stage="fit")

    # Model
    lit_model = LitModel(cfg)
    total_params = sum(p.numel() for p in lit_model.model.parameters())
    trainable_params = sum(p.numel() for p in lit_model.model.parameters() if p.requires_grad)
    logger.info(f"Model: {cfg.model.name} | Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # Callbacks
    callbacks = []
    ckpt_cfg = train_cfg.checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_path),
        filename="best_model",
        monitor=ckpt_cfg.monitor,
        mode=ckpt_cfg.mode,
        save_top_k=ckpt_cfg.save_top_k,
        save_last=ckpt_cfg.save_last,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    es_cfg = train_cfg.early_stopping
    if es_cfg.enabled:
        callbacks.append(
            EarlyStopping(
                monitor=es_cfg.monitor,
                patience=es_cfg.patience,
                mode=es_cfg.mode,
                min_delta=0.001,
                verbose=True,
            )
        )

    callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    # Trainer
    trainer = pl.Trainer(
        default_root_dir=str(output_path),
        max_epochs=train_cfg.max_epochs,
        min_epochs=train_cfg.get("min_epochs", 1),
        accelerator=train_cfg.get("accelerator", "auto"),
        devices=train_cfg.get("devices", "auto"),
        precision=train_cfg.get("precision", 32),
        gradient_clip_val=train_cfg.gradient_clip_val,
        accumulate_grad_batches=train_cfg.get("accumulate_grad_batches", 1),
        log_every_n_steps=10,
        callbacks=callbacks,
        logger=wandb_logger,
        enable_checkpointing=True,
    )

    trainer.fit(lit_model, datamodule=datamodule)

    # Test
    best_ckpt_path = output_path / "best_model.ckpt"
    if best_ckpt_path.exists():
        trainer.test(ckpt_path=str(best_ckpt_path), datamodule=datamodule, weights_only=False)

    # Save final model
    final_model_path = output_path / "model.pt"
    torch.save(lit_model.model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

    # Log artifact
    artifact = wandb.Artifact(
        name=f"{cfg.experiment_name}_model",
        type="model",
        metadata={"model_name": cfg.model.name, "num_params": total_params},
    )
    if best_ckpt_path.exists():
        artifact.add_file(str(best_ckpt_path))
    artifact.add_file(str(final_model_path))
    wandb.log_artifact(artifact)

    return str(final_model_path)


@hydra.main(config_path=_CONFIG_PATH, config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> None:
    """Train a model (Hydra entry point)."""
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
