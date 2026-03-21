"""Shared infrastructure for HPC scripts: logging, paths, environment helpers."""

from __future__ import annotations

import os
import platform
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
    "log_gpu_info",
    "log_system_info",
    "setup_hpc_environment",
    "setup_logging",
]

# ---------------------------------------------------------------------------
# Path constants (overridable via env vars)
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(os.environ.get("DANISH_ASR_PROJECT_DIR", Path.home() / "danish_asr"))
SCRATCH_DIR = Path(os.environ.get("DANISH_ASR_SCRATCH_DIR", f"/work3/{os.environ.get('USER', 's204696')}"))
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
        except OSError:
            logger.debug(f"Disk {path_name} ({path}): not accessible")


def log_gpu_info() -> None:
    """Log torch CUDA availability, device names, and VRAM."""
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                total = torch.cuda.get_device_properties(i).total_mem / (1024**3)
                logger.info(f"GPU {i}: {name} ({total:.1f} GB)")
        else:
            logger.warning("CUDA not available")
    except ImportError:
        logger.warning("PyTorch not installed — cannot check GPU")
