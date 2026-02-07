"""Model monitoring and drift detection tasks."""

import os
from pathlib import Path

from invoke import Context, task
from loguru import logger

WINDOWS = os.name == "nt"
PROJECT_NAME = "danish_asr"


@task
def extract_stats(ctx: Context, checkpoint: str = "", output: str = "reference_stats.json") -> None:
    """Extract reference statistics for drift detection."""
    if not checkpoint:
        logger.error("ERROR: --checkpoint is required")
        return
    if not Path(checkpoint).exists():
        logger.error(f"ERROR: Checkpoint not found: {checkpoint}")
        return
    ctx.run(
        f"uv run python -m {PROJECT_NAME}.monitoring.extract_stats --checkpoint {checkpoint} --output {output}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def check_drift(ctx: Context, reference: str = "reference_stats.json", data_path: str = "") -> None:
    """Check for data drift against reference statistics."""
    if not Path(reference).exists():
        logger.error(f"ERROR: Reference statistics not found: {reference}")
        return
    cmd = f"uv run python -m {PROJECT_NAME}.monitoring.drift_check --reference {reference}"
    if data_path:
        cmd += f" --data-path {data_path}"
    ctx.run(cmd, echo=True, pty=not WINDOWS)
