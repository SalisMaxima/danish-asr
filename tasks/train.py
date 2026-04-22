"""Training and hyperparameter tuning tasks."""

import importlib.util
import os
import re
import shlex
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
ENV_SCRIPT = PROJECT_ROOT / "scripts" / "hpc" / "env.sh"


def _has_module(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def _run_with_omni_workflows(ctx: Context, cmd: str) -> None:
    if _has_module("workflows.recipes.wav2vec2.asr"):
        ctx.run(cmd, echo=True, pty=not WINDOWS)
        return

    if WINDOWS:
        logger.error(
            "omnilingual-asr workflows are not installed in the active environment. "
            "Install them first before using train.omniasr."
        )
        return

    hpc_cmd = shlex.quote(f"source {ENV_SCRIPT} && setup_omniasr && {cmd}")
    ctx.run(f"bash -lc {hpc_cmd}", echo=True, pty=not WINDOWS)


def _available_llm_hardware(prefix: str, config_dir: Path) -> list[str]:
    return sorted(
        f"{prefix}-{path.stem.removeprefix('llm-finetune-')}" for path in config_dir.glob("llm-finetune-*.yaml")
    )


def _resolve_omniasr_config(hardware: str) -> Path:
    if hardware == "local":
        return PROJECT_ROOT / "configs" / "fairseq2" / "300m" / "ctc-finetune-local.yaml"
    if hardware == "hpc":
        return PROJECT_ROOT / "configs" / "fairseq2" / "legacy" / "ctc-finetune-hpc.yaml"
    if hardware == "llm-300m":
        available = ", ".join(_available_llm_hardware("llm-300m", PROJECT_ROOT / "configs" / "fairseq2" / "llm_300m"))
        raise ValueError(f"Invalid hardware {hardware!r}. Available 300M LLM targets: {available}")
    if hardware == "llm-1b":
        available = ", ".join(_available_llm_hardware("llm-1b", PROJECT_ROOT / "configs" / "fairseq2" / "llm_1b"))
        raise ValueError(f"Invalid hardware {hardware!r}. Available 1B LLM targets: {available}")
    if hardware.startswith("llm-300m-"):
        config_dir = PROJECT_ROOT / "configs" / "fairseq2" / "llm_300m"
        suffix = hardware.removeprefix("llm-300m-")
        config = config_dir / f"llm-finetune-{suffix}.yaml"
        if not config.exists():
            available = ", ".join(_available_llm_hardware("llm-300m", config_dir))
            raise ValueError(f"Unknown hardware {hardware!r}. Available 300M LLM targets: {available}")
        return config
    if hardware.startswith("llm-1b-"):
        config_dir = PROJECT_ROOT / "configs" / "fairseq2" / "llm_1b"
        suffix = hardware.removeprefix("llm-1b-")
        config = config_dir / f"llm-finetune-{suffix}.yaml"
        if not config.exists():
            available = ", ".join(_available_llm_hardware("llm-1b", config_dir))
            raise ValueError(f"Unknown hardware {hardware!r}. Available 1B LLM targets: {available}")
        return config
    return PROJECT_ROOT / "configs" / "fairseq2" / "300m" / f"ctc-finetune-{hardware}.yaml"


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
    """Train omniASR via the fairseq2 recipe."""
    try:
        config = _resolve_omniasr_config(hardware)
    except ValueError as exc:
        logger.error(str(exc))
        return

    if not config.exists():
        logger.error(f"Config not found: {config}")
        logger.error("Available: local, hpc, ctc config suffixes like hpc-20k, or LLM targets like llm-300m-smoke")
        return

    if not output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = str(PROJECT_ROOT / "outputs" / f"omniasr_{hardware}_{timestamp}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cmd = f"python -m workflows.recipes.wav2vec2.asr {shlex.quote(output_dir)} --config-file {shlex.quote(str(config))}"
    if args:
        cmd += f" {args}"
    logger.info(f"Starting omniASR training: hardware={hardware}, output={output_dir}")
    _run_with_omni_workflows(ctx, cmd)


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
        f"< scripts/hpc/300m/05_smoke_test.sh'",
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
        f"bsub < scripts/hpc/legacy/06_sweep.sh'",
        pty=not WINDOWS,
    )


@task
def omniasr_eval(ctx: Context, checkpoint_dir: str, config: str = "") -> None:
    """Evaluate omniASR checkpoint with an explicit eval config."""
    config_path = Path(config) if config else PROJECT_ROOT / "configs" / "fairseq2" / "300m" / "ctc-eval-e2.yaml"
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        return

    if not Path(checkpoint_dir).exists():
        logger.error(f"Checkpoint dir not found: {checkpoint_dir}")
        return

    cmd = (
        f"python -m workflows.recipes.wav2vec2.asr.eval {shlex.quote(checkpoint_dir)} "
        f"--config-file {shlex.quote(str(config_path))}"
    )
    logger.info(f"Evaluating omniASR: checkpoint={checkpoint_dir}, config={config_path}")
    _run_with_omni_workflows(ctx, cmd)
