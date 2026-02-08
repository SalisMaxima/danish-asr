"""W&B sweep-compatible training entrypoint.

Maps sweep CLI flags into Hydra config overrides.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import typer
import wandb
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger

from danish_asr.train import configure_logging, train_model

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_DIR = str(_PROJECT_ROOT / "configs")

_OVERRIDE_MAP: list[tuple[str, str]] = [
    ("model", "model"),
    ("lr", "train.optimizer.lr"),
    ("weight_decay", "train.optimizer.weight_decay"),
    ("batch_size", "data.batch_size"),
    ("max_epochs", "train.max_epochs"),
    ("seed", "seed"),
    ("wandb_project", "wandb.project"),
    ("wandb_entity", "wandb.entity"),
    ("wandb_mode", "wandb.mode"),
    ("dropout", "model.dropout"),
]


def _build_overrides(**kwargs: Any) -> list[str]:
    """Build Hydra config overrides from non-None CLI parameters."""
    return [f"{cfg_key}={kwargs[param]}" for param, cfg_key in _OVERRIDE_MAP if kwargs.get(param) is not None]


def _resolve_output_base(cfg: DictConfig) -> Path:
    resolved = OmegaConf.to_container(cfg, resolve=True)
    out = resolved.get("output_dir", "outputs") if isinstance(resolved, dict) else "outputs"
    out_path = Path(out)
    if out_path.is_absolute():
        return out_path
    return (_PROJECT_ROOT / out_path).resolve()


def sweep_train(
    lr: float = typer.Option(None, help="Learning rate"),
    weight_decay: float = typer.Option(None, "--weight_decay", "--weight-decay", help="Weight decay"),
    batch_size: int = typer.Option(None, "--batch_size", "--batch-size", help="Batch size"),
    model: str = typer.Option(None, help="Model config group"),
    max_epochs: int = typer.Option(None, "--max_epochs", "--max-epochs", help="Max epochs"),
    seed: int = typer.Option(None, help="Random seed"),
    wandb_project: str = typer.Option(None, "--wandb_project", "--wandb-project", help="Override W&B project"),
    wandb_entity: str = typer.Option(None, "--wandb_entity", "--wandb-entity", help="Override W&B entity"),
    wandb_mode: str = typer.Option(None, "--wandb_mode", "--wandb-mode", help="online | offline | disabled"),
    dropout: float = typer.Option(None, help="Dropout rate"),
) -> None:
    """Train one run (designed to be launched by `wandb agent`)."""
    overrides = _build_overrides(
        model=model,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        seed=seed,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_mode=wandb_mode,
        dropout=dropout,
    )

    with hydra.initialize_config_dir(config_dir=_CONFIG_DIR, version_base="1.3"):
        cfg = hydra.compose(config_name="config", overrides=overrides)

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.get("entity"),
        tags=list(cfg.wandb.get("tags", [])),
        mode=cfg.wandb.get("mode", "online"),
        job_type="train",
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    try:
        output_base = _resolve_output_base(cfg)
        output_dir = output_base / "sweeps" / run.id
        output_dir.mkdir(parents=True, exist_ok=True)
        configure_logging(str(output_dir))

        wandb_logger = WandbLogger(experiment=run, save_dir=str(output_dir))
        model_path = train_model(cfg, str(output_dir), wandb_logger)
        logger.info(f"Sweep training complete. Model saved to {model_path}")
    except Exception as e:
        logger.exception(f"Sweep run failed: {e}")
        raise
    finally:
        wandb.finish()


def main() -> None:
    typer.run(sweep_train)


if __name__ == "__main__":
    main()
