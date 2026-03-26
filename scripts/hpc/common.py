"""Shared infrastructure for HPC scripts: logging, paths, environment helpers."""

from __future__ import annotations

import getpass
import os
import platform
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

from danish_asr.preprocessing import HF_SPLITS, LANGUAGE, SPLIT_MAP, SUBSETS

# Re-export for convenience in sibling scripts
__all__ = [
    "HF_SPLITS",
    "LANGUAGE",
    "SPLIT_MAP",
    "SUBSETS",
    "FAIRSEQ2_DIR",
    "LOGS_DIR",
    "OUTPUT_DIR",
    "PROJECT_DIR",
    "SCRATCH_DIR",
    "UNIVERSAL_DIR",
    "LOSS_PATTERN",
    "WER_PATTERN",
    "CER_PATTERN",
    "STEP_PATTERN",
    "log_line_to_wandb",
    "log_gpu_info",
    "log_system_info",
    "setup_hpc_environment",
    "setup_logging",
]

# ---------------------------------------------------------------------------
# Path constants (overridable via env vars)
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(os.environ.get("DANISH_ASR_PROJECT_DIR", Path.home() / "danish_asr"))
SCRATCH_DIR = Path(os.environ.get("DANISH_ASR_SCRATCH_DIR", f"/work3/{getpass.getuser()}"))
UNIVERSAL_DIR = SCRATCH_DIR / "data" / "preprocessed"
FAIRSEQ2_DIR = SCRATCH_DIR / "data" / "parquet" / "version=0"
OUTPUT_DIR = SCRATCH_DIR / "outputs"
LOGS_DIR = SCRATCH_DIR / "logs" / "python"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(script_name: str) -> None:
    """Configure loguru with console (INFO) and file (DEBUG) sinks."""
    logger.remove()

    # Console sink — no color (LSF captures stderr)
    logger.add(
        sys.stderr,
        level="INFO",
        colorize=False,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    )

    # File sink — DEBUG level, timestamped
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"{script_name}_{timestamp}.log"
    logger.add(
        str(log_file),
        level="DEBUG",
        rotation="500 MB",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} | {message}",
    )
    logger.info(f"Log file: {log_file}")


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def setup_hpc_environment() -> None:
    """Set HF_HOME, FAIRSEQ2_CACHE_DIR, TMPDIR for HPC jobs."""
    env_vars = {
        "HF_HOME": str(SCRATCH_DIR / "hf_cache"),
        "HF_DATASETS_CACHE": str(SCRATCH_DIR / "hf_cache" / "datasets"),
        "FAIRSEQ2_CACHE_DIR": str(SCRATCH_DIR / "fairseq2_cache"),
        "TMPDIR": str(SCRATCH_DIR / "tmp"),
    }
    for key, default in env_vars.items():
        value = os.environ.get(key, default)
        os.environ[key] = value
        logger.debug(f"ENV {key}={value}")

    # Ensure TMPDIR exists
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------


def log_system_info() -> None:
    """Log hostname, user, Python version, CUDA_VISIBLE_DEVICES, LSB_JOBID, disk space."""
    logger.info(f"Hostname: {platform.node()}")
    logger.info(f"User: {os.environ.get('USER', 'unknown')}")
    logger.info(f"Python: {sys.executable} ({sys.version})")
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    logger.info(f"LSB_JOBID: {os.environ.get('LSB_JOBID', 'not set')}")

    for path_name, path in [("SCRATCH_DIR", SCRATCH_DIR), ("PROJECT_DIR", PROJECT_DIR)]:
        try:
            usage = shutil.disk_usage(path)
            used_gb = usage.used / (1024**3)
            total_gb = usage.total / (1024**3)
            logger.info(f"Disk {path_name} ({path}): {used_gb:.1f} / {total_gb:.1f} GB used")
        except OSError as e:
            logger.warning(f"Disk {path_name} ({path}): not accessible — {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Fairseq2 metric extraction patterns
# ---------------------------------------------------------------------------
# Fairseq2 CTC typically logs lines like:
#   | train | step 100 | loss 1.234 | ...
#   | valid | step 500 | wer 0.42 | cer 0.12 | ...

LOSS_PATTERN = re.compile(r"\bloss[:\s]+([\d.]+)", re.IGNORECASE)
WER_PATTERN = re.compile(r"\bwer[:\s]+([\d.]+)", re.IGNORECASE)
CER_PATTERN = re.compile(r"\bcer[:\s]+([\d.]+)", re.IGNORECASE)
STEP_PATTERN = re.compile(r"\bstep[:\s]+(\d+)", re.IGNORECASE)


def log_line_to_wandb(line: str, wandb_run: object | None) -> None:
    """Parse a fairseq2 output line and log matching metrics to W&B."""
    if wandb_run is None:
        return
    try:
        import wandb

        metrics: dict[str, float] = {}
        step: int | None = None

        step_match = STEP_PATTERN.search(line)
        if step_match:
            step = int(step_match.group(1))

        if "train" in line.lower():
            loss_match = LOSS_PATTERN.search(line)
            if loss_match:
                metrics["train/loss"] = float(loss_match.group(1))
        elif "valid" in line.lower():
            wer_match = WER_PATTERN.search(line)
            cer_match = CER_PATTERN.search(line)
            loss_match = LOSS_PATTERN.search(line)
            if wer_match:
                metrics["val/wer"] = float(wer_match.group(1))
            if cer_match:
                metrics["val/cer"] = float(cer_match.group(1))
            if loss_match:
                metrics["val/loss"] = float(loss_match.group(1))

        if metrics and step is not None:
            wandb.log(metrics, step=step)
    except Exception as e:
        logger.debug(f"W&B metric parse failed for line ({type(e).__name__}: {e}): {line[:80]}")


def log_gpu_info() -> None:
    """Log torch CUDA availability, device names, and VRAM."""
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not installed — cannot check GPU")
        return

    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                logger.info(f"GPU {i}: {name} ({total:.1f} GB)")
        else:
            logger.warning("CUDA not available")
    except RuntimeError as e:
        logger.warning(f"Failed to query GPU info: {type(e).__name__}: {e}")
