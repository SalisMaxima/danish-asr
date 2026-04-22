"""Asset management tasks (model checkpoint pre-download)."""

import importlib.util
import os
import shlex
from pathlib import Path

from invoke import Context, task

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_SCRIPT = PROJECT_ROOT / "scripts" / "hpc" / "env.sh"
PULL_SCRIPT = PROJECT_ROOT / "scripts" / "hpc" / "pull_llm_assets.py"
VALID_SIZES = {"300m", "1b"}
WINDOWS = os.name == "nt"


def _has_module(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


@task(name="pull-llm")
def pull_llm(ctx: Context, size: str = "300m") -> None:
    """Pre-download omniASR_LLM_300M_v2 or omniASR_LLM_1B_v2 to FAIRSEQ2_CACHE_DIR.

    Run on the DTU HPC login node (or locally) before submitting a training
    job. Respects the FAIRSEQ2_CACHE_DIR env var set by scripts/hpc/env.sh
    (/work3/$USER/fairseq2_cache on HPC).

    Usage:
        invoke assets.pull-llm --size 300m
        invoke assets.pull-llm --size 1b
    """
    if size not in VALID_SIZES:
        raise ValueError(f"Invalid size {size!r}. Must be one of: {sorted(VALID_SIZES)}")

    cmd = f"python {shlex.quote(str(PULL_SCRIPT))} --size {shlex.quote(size)}"

    if _has_module("omnilingual_asr"):
        ctx.run(cmd, echo=True, pty=not WINDOWS)
        return

    if WINDOWS:
        raise RuntimeError(
            "omnilingual_asr is not installed in the active environment. "
            "Install the omni dependencies first or run this task from the HPC login node."
        )

    hpc_cmd = shlex.quote(f"source {ENV_SCRIPT} && setup_omniasr && {cmd}")
    ctx.run(f"bash -lc {hpc_cmd}", echo=True, pty=not WINDOWS)
