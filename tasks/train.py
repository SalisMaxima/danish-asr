"""Training and hyperparameter tuning tasks."""

import os
import re
from datetime import datetime
from pathlib import Path

from invoke import Context, task
from loguru import logger

HPC_USER = os.environ.get("HPC_USER", "s204696")
if not re.fullmatch(r"[a-zA-Z0-9_.-]+", HPC_USER):
    raise ValueError(f"Invalid HPC_USER: {HPC_USER!r}")
HPC_LOGIN = f"{HPC_USER}@login.hpc.dtu.dk"

WINDOWS = os.name == "nt"
PROJECT_NAME = "danish_asr"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@task
def train(ctx: Context, entity: str = "", args: str = "") -> None:
    """Train model with wandb logging."""
    entity_override = f"wandb.entity={entity}" if entity else ""
    full_args = f"{entity_override} {args}".strip()
    ctx.run(f"uv run python -m {PROJECT_NAME}.train {full_args}", echo=True, pty=not WINDOWS)


@task
def sweep(
    ctx: Context, sweep_config: str = "configs/sweeps/train_sweep.yaml", project: str = "", entity: str = ""
) -> None:
    """Create a W&B sweep from a sweep YAML."""
    sweep_config_path = Path(sweep_config)
    if not sweep_config_path.exists():
        logger.error(f"ERROR: Sweep config not found: {sweep_config}")
        return
    cmd = f"uv run wandb sweep {sweep_config}"
    if project:
        cmd += f" --project {project}"
    if entity:
        cmd += f" --entity {entity}"
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def sweep_agent(ctx: Context, sweep_id: str) -> None:
    """Run a W&B sweep agent."""
    ctx.run(f"uv run wandb agent {sweep_id}", echo=True, pty=not WINDOWS)


@task
def sweep_best(ctx: Context, sweep_id: str, metric: str = "val_acc", goal: str = "maximize") -> None:
    """Print the best run (and its config) for a sweep."""
    ctx.run(
        f"uv run python -m {PROJECT_NAME}.sweep_best {sweep_id} --metric {metric} --goal {goal}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def omniasr(ctx: Context, hardware: str = "local", output_dir: str = "", args: str = "") -> None:
    """Train omniASR_CTC_300M_v2 via fairseq2 recipe."""
    config = PROJECT_ROOT / "configs" / "fairseq2" / f"ctc-finetune-{hardware}.yaml"
    if not config.exists():
        logger.error(f"Config not found: {config}")
        logger.error("Available: local, hpc")
        return

    if not output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = str(PROJECT_ROOT / "outputs" / f"omniasr_{hardware}_{timestamp}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cmd = f"uv run python -m workflows.recipes.wav2vec2.asr {output_dir} --config-file {config}"
    if args:
        cmd += f" {args}"
    logger.info(f"Starting omniASR training: hardware={hardware}, output={output_dir}")
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task(name="hpc-smoke")
def hpc_smoke(ctx: Context) -> None:
    """Submit 50-step smoke test to DTU HPC gpua100 queue (requires VPN). Validates full pipeline.

    Runs 'git pull' on the HPC node before submitting, so the job always uses
    the latest pushed commit — local uncommitted changes are NOT included.
    """
    ctx.run(
        f"ssh {HPC_LOGIN} '"
        f"cd ~/danish_asr && git pull && "
        f"mkdir -p /work3/$USER/logs/lsf && "
        f"bsub -o /work3/$USER/logs/lsf/smoke_%J.out -e /work3/$USER/logs/lsf/smoke_%J.err "
        f"< scripts/hpc/05_smoke_test.sh'",
        pty=not WINDOWS,
    )


@task(name="omniasr-sweep")
def omniasr_sweep(ctx: Context, project: str = "danish-asr", entity: str = "") -> None:
    """Create a W&B sweep for omniASR hyperparameter tuning. Prints the sweep ID."""
    sweep_config = PROJECT_ROOT / "configs" / "sweeps" / "omniasr_sweep.yaml"
    if not sweep_config.exists():
        logger.error(f"Sweep config not found: {sweep_config}")
        return
    cmd = f"uv run wandb sweep {sweep_config} --project {project}"
    if entity:
        cmd += f" --entity {entity}"
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task(name="hpc-sweep")
def hpc_sweep(ctx: Context, sweep_id: str) -> None:
    """Submit W&B sweep agent job array to DTU HPC (requires VPN).

    Usage: invoke train.hpc-sweep --sweep-id entity/project/abc123
    """
    if not re.fullmatch(r"[a-zA-Z0-9_./-]+", sweep_id):
        logger.error(f"Invalid sweep ID: {sweep_id!r}")
        return
    ctx.run(
        f"ssh {HPC_LOGIN} '"
        f"cd ~/danish_asr && git pull && "
        f"mkdir -p /work3/$USER/logs/lsf && "
        f"SWEEP_ID={sweep_id} "
        f"bsub < scripts/hpc/06_sweep.sh'",
        pty=not WINDOWS,
    )


@task
def omniasr_eval(ctx: Context, checkpoint_dir: str, hardware: str = "local") -> None:
    """Evaluate omniASR checkpoint."""
    config = PROJECT_ROOT / "configs" / "fairseq2" / f"ctc-finetune-{hardware}.yaml"
    if not config.exists():
        logger.error(f"Config not found: {config}")
        return

    if not Path(checkpoint_dir).exists():
        logger.error(f"Checkpoint dir not found: {checkpoint_dir}")
        return

    cmd = f"uv run python -m workflows.recipes.wav2vec2.asr.eval.recipe {checkpoint_dir} --config-file {config}"
    logger.info(f"Evaluating omniASR: checkpoint={checkpoint_dir}")
    ctx.run(cmd, echo=True, pty=not WINDOWS)
