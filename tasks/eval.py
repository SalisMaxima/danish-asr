"""Model evaluation and analysis tasks."""

import os

from invoke import Context, task
from loguru import logger

WINDOWS = os.name == "nt"
PROJECT_NAME = "danish_asr"


@task
def analyze(ctx: Context, args: str = "") -> None:
    """Run model analysis CLI."""
    if not args:
        logger.error("ERROR: args required. Use 'invoke eval.analyze \"--help\"' for usage.")
        return
    ctx.run(f"uv run python -m {PROJECT_NAME}.analysis.cli {args}", echo=True, pty=not WINDOWS)


@task
def benchmark(ctx: Context, checkpoint: str = "", dataset: str = "test", batch_size: int = 32) -> None:
    """Benchmark model inference speed and throughput."""
    if not checkpoint:
        logger.error("ERROR: --checkpoint is required")
        return
    from pathlib import Path

    if not Path(checkpoint).exists():
        logger.error(f"ERROR: Checkpoint not found: {checkpoint}")
        return
    ctx.run(
        f"uv run python -m {PROJECT_NAME}.inference_benchmark --checkpoint {checkpoint} --dataset {dataset} --batch-size {batch_size}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def profile(ctx: Context, checkpoint: str = "", output: str = "profile_stats.txt") -> None:
    """Profile model and training performance using cProfile."""
    if checkpoint:
        from pathlib import Path

        if not Path(checkpoint).exists():
            logger.error(f"ERROR: Checkpoint not found: {checkpoint}")
            return
        ctx.run(
            f"uv run python -m cProfile -o {output} -m {PROJECT_NAME}.inference_benchmark --checkpoint {checkpoint}",
            echo=True,
            pty=not WINDOWS,
        )
    else:
        ctx.run(
            f"uv run python -m cProfile -o {output} -m {PROJECT_NAME}.train train.max_epochs=1",
            echo=True,
            pty=not WINDOWS,
        )
    print(f"\n✓ Profile saved to {output}")


@task
def model_info(ctx: Context, checkpoint: str) -> None:
    """Show model size, parameters, and architecture info."""
    if not checkpoint:
        logger.error("ERROR: --checkpoint is required")
        return
    from pathlib import Path

    if not Path(checkpoint).exists():
        logger.error(f"ERROR: Checkpoint not found: {checkpoint}")
        return
    ctx.run(
        f"uv run python -m {PROJECT_NAME}.model_info --checkpoint {checkpoint}",
        echo=True,
        pty=not WINDOWS,
    )
