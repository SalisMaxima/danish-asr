"""Asset management tasks (model checkpoint pre-download)."""

from pathlib import Path

from invoke import Context, task

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PULL_SCRIPT = PROJECT_ROOT / "scripts" / "hpc" / "pull_llm_assets.py"


@task(name="pull-llm")
def pull_llm(ctx: Context, size: str = "300m") -> None:
    """Pre-download omniASR_LLM_{300m,1b}_v2 checkpoint to FAIRSEQ2_CACHE_DIR.

    Run on the DTU HPC login node (or locally) before submitting a training
    job. Respects the FAIRSEQ2_CACHE_DIR env var set by scripts/hpc/env.sh
    (/work3/$USER/fairseq2_cache on HPC).

    Usage:
        invoke assets.pull-llm --size 300m
        invoke assets.pull-llm --size 1b
    """
    ctx.run(f"uv run python {PULL_SCRIPT} --size {size}", echo=True, pty=True)
