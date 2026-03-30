"""Shared infrastructure for HF baseline training scripts (Wav2Vec2, Whisper)."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from loguru import logger
from torch.utils.data import ConcatDataset

from danish_asr.data import PreprocessedCoRalDataset
from danish_asr.metrics import compute_cer, compute_wer
from danish_asr.text import normalize_coral_text

from .common import PROJECT_DIR, SCRATCH_DIR

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(config_path: str | Path) -> dict:
    """Load a YAML config file and return its contents as a dict."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open() as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------

_PREPROCESSED_DIR = SCRATCH_DIR / "data" / "preprocessed"


def build_datasets(
    processor: Any,
    tokenizer: Any | None = None,
    text_normalizer: Callable[[str], str] | None = None,
    subsets: list[str] | None = None,
    max_duration: float = 30.0,
    sample_rate: int = 16000,
) -> tuple[ConcatDataset | PreprocessedCoRalDataset, ConcatDataset | PreprocessedCoRalDataset]:
    """Build train and validation datasets by concatenating subsets.

    Returns ``(train_dataset, val_dataset)``.

    Each subset (e.g. ``read_aloud``, ``conversation``) is loaded from
    ``/work3/$USER/data/preprocessed/{subset}/{split}/`` via
    :class:`~danish_asr.data.PreprocessedCoRalDataset`.
    """
    if subsets is None:
        subsets = ["read_aloud", "conversation"]

    train_parts: list[PreprocessedCoRalDataset] = []
    val_parts: list[PreprocessedCoRalDataset] = []

    for subset in subsets:
        for split, parts_list in [("train", train_parts), ("validation", val_parts)]:
            parquet_dir = _PREPROCESSED_DIR / subset / split
            if not parquet_dir.exists():
                raise FileNotFoundError(
                    f"Preprocessed data not found: {parquet_dir}\nRun the preprocessing pipeline first."
                )
            ds = PreprocessedCoRalDataset(
                parquet_dir=parquet_dir,
                processor=processor,
                tokenizer=tokenizer,
                text_normalizer=text_normalizer,
                max_duration=max_duration,
                sample_rate=sample_rate,
            )
            logger.info(f"Loaded {subset}/{split}: {len(ds)} samples")
            parts_list.append(ds)

    train_dataset = ConcatDataset(train_parts) if len(train_parts) > 1 else train_parts[0]
    val_dataset = ConcatDataset(val_parts) if len(val_parts) > 1 else val_parts[0]

    logger.info(f"Total train: {len(train_dataset)}, val: {len(val_dataset)}")
    return train_dataset, val_dataset


# ---------------------------------------------------------------------------
# Data collators
# ---------------------------------------------------------------------------


class CTCDataCollator:
    """Collator for Wav2Vec2 CTC training.

    Pads ``input_values`` to the longest sample in the batch, creates
    ``attention_mask``, and pads ``labels`` with ``-100``.
    """

    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, features: list[dict]) -> dict:
        input_values = [f["input_values"] for f in features]
        batch = self.processor.feature_extractor.pad(
            {"input_values": [iv.numpy() for iv in input_values]},
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        labels = [f["labels"] for f in features]
        batch["labels"] = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
        )
        return batch


class Seq2SeqDataCollator:
    """Collator for Whisper seq2seq training.

    Stacks fixed-size ``input_features`` (80x3000 mel spectrograms) and pads
    ``labels`` with ``-100`` for loss masking.
    """

    def __init__(self, processor: Any) -> None:
        self.processor = processor
        self.pad_token_id = processor.tokenizer.pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        input_features = torch.stack([f["input_features"] for f in features])

        labels = [f["labels"] for f in features]
        max_len = max(lab.shape[0] for lab in labels)
        padded = []
        for lab in labels:
            pad_len = max_len - lab.shape[0]
            if pad_len > 0:
                lab = torch.cat([lab, torch.full((pad_len,), self.pad_token_id, dtype=lab.dtype)])
            padded.append(lab)
        labels_tensor = torch.stack(padded)
        labels_tensor[labels_tensor == self.pad_token_id] = -100

        return {"input_features": input_features, "labels": labels_tensor}


# ---------------------------------------------------------------------------
# Compute metrics factories
# ---------------------------------------------------------------------------


def make_ctc_compute_metrics(processor: Any) -> Callable:
    """Return a ``compute_metrics`` function for CTC models.

    Decodes logits via argmax, then computes WER and CER via jiwer.
    """

    def compute_metrics(pred: Any) -> dict[str, float]:
        pred_ids = np.argmax(pred.predictions, axis=-1)

        # Replace -100 with pad token for decoding
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(label_ids, group_tokens=False)

        return {
            "wer": compute_wer(pred_str, label_str),
            "cer": compute_cer(pred_str, label_str),
        }

    return compute_metrics


def make_seq2seq_compute_metrics(processor: Any) -> Callable:
    """Return a ``compute_metrics`` function for seq2seq models.

    Decodes token IDs and normalizes text before computing WER and CER.
    """

    def compute_metrics(pred: Any) -> dict[str, float]:
        pred_ids = pred.predictions
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        # Normalize for fair comparison
        pred_str = [normalize_coral_text(s) for s in pred_str]
        label_str = [normalize_coral_text(s) for s in label_str]

        return {
            "wer": compute_wer(pred_str, label_str),
            "cer": compute_cer(pred_str, label_str),
        }

    return compute_metrics


# ---------------------------------------------------------------------------
# W&B initialization
# ---------------------------------------------------------------------------


def init_wandb(
    project: str,
    name: str | None,
    tags: list[str],
    config: dict,
    config_path: Path | None = None,
    resume: str = "allow",
    job_type: str = "train",
) -> Any | None:
    """Initialise a W&B run with git metadata and LSF job info.

    Sets ``WANDB_RUN_ID`` in the environment so the HF Trainer's built-in
    W&B callback attaches to the same run.

    Returns the run object, or ``None`` if W&B is unavailable.
    """
    try:
        import wandb

        run = wandb.init(
            project=project,
            name=name or None,
            tags=tags or None,
            job_type=job_type,
            config=config,
            resume=resume,
        )

        # Metric summaries for the runs table
        run.define_metric("eval/wer", summary="min")
        run.define_metric("eval/cer", summary="min")
        run.define_metric("eval/loss", summary="min")
        run.define_metric("train/loss", summary="min")

        # Git metadata
        try:
            git_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(PROJECT_DIR),
                text=True,
            ).strip()
            git_branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=str(PROJECT_DIR),
                text=True,
            ).strip()
            git_dirty = bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"],
                    cwd=str(PROJECT_DIR),
                    text=True,
                ).strip()
            )
            run.config.update({"git_sha": git_sha, "git_branch": git_branch, "git_dirty": git_dirty})
        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
            logger.warning(f"Could not read git metadata: {type(e).__name__}: {e}")

        # HPC job ID
        job_id = os.environ.get("LSB_JOBID")
        if job_id:
            run.config.update({"lsf_job_id": job_id})

        # Save config file to W&B Files tab
        if config_path and config_path.exists():
            wandb.save(str(config_path), base_path=str(config_path.parent), policy="now")

        # Set WANDB_RUN_ID so HF Trainer callback attaches to this run
        os.environ["WANDB_RUN_ID"] = run.id

        logger.info(f"W&B run initialised: {run.url}")
        return run

    except ImportError:
        logger.warning("wandb not installed — skipping W&B logging")
        return None
    except Exception as e:
        logger.error(f"W&B init failed ({type(e).__name__}: {e}) — continuing without W&B")
        return None


def finish_wandb(run: Any | None, exit_code: int = 0) -> None:
    """Finish a W&B run gracefully."""
    if run is None:
        return
    try:
        import wandb

        wandb.finish(exit_code=exit_code)
    except Exception as e:
        logger.warning(f"W&B finish failed: {e}")
