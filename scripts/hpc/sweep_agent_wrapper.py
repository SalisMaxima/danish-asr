"""W&B sweep agent wrapper for omniASR fairseq2 training.

Invoked by ``wandb agent``.  Each call:
1. Reads sweep-assigned hyperparameters from ``wandb.config``
2. Patches the base fairseq2 YAML config with those values
3. Runs the fairseq2 training recipe as a subprocess
4. Streams stdout, parses metrics, and logs them to W&B

Usage (typically called by wandb agent, not directly):
    python scripts/hpc/sweep_agent_wrapper.py
    python scripts/hpc/sweep_agent_wrapper.py --base-config configs/fairseq2/ctc-finetune-hpc.yaml
"""

from __future__ import annotations

import argparse
import copy
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from scripts.hpc.common import (
    FAIRSEQ2_DIR,
    OUTPUT_DIR,
    PROJECT_DIR,
    SCRATCH_DIR,
    log_gpu_info,
    log_line_to_wandb,
    log_system_info,
    setup_hpc_environment,
    setup_logging,
)

_HEARTBEAT_INTERVAL = 300  # seconds


def _apply_sweep_overrides(base: dict[str, Any], sweep_config: dict[str, Any]) -> dict[str, Any]:
    """Patch base fairseq2 config with sweep hyperparameters."""
    config = copy.deepcopy(base)

    if "lr" in sweep_config:
        config.setdefault("optimizer", {}).setdefault("config", {})["lr"] = sweep_config["lr"]

    if "freeze_encoder_for_n_steps" in sweep_config:
        config.setdefault("trainer", {})["freeze_encoder_for_n_steps"] = sweep_config["freeze_encoder_for_n_steps"]

    if "max_num_elements" in sweep_config:
        config.setdefault("dataset", {}).setdefault("asr_task_config", {})["max_num_elements"] = sweep_config[
            "max_num_elements"
        ]

    if "grad_accumulation" in sweep_config:
        config.setdefault("trainer", {}).setdefault("grad_accumulation", {})["num_batches"] = sweep_config[
            "grad_accumulation"
        ]

    return config


def _ensure_data_symlink() -> None:
    """Create symlink: PROJECT_DIR/data -> SCRATCH_DIR/data."""
    link_path = PROJECT_DIR / "data"
    target = SCRATCH_DIR / "data"

    if link_path.is_symlink():
        if link_path.resolve() == target.resolve():
            return
        link_path.unlink()
    elif link_path.exists():
        logger.error(f"{link_path} exists and is not a symlink")
        sys.exit(1)

    link_path.symlink_to(target)
    logger.info(f"Created data symlink: {link_path} -> {target}")


def _check_prerequisites(config_path: Path) -> None:
    """Minimal prerequisite checks."""
    if not config_path.exists():
        logger.error(f"Base config not found: {config_path}")
        sys.exit(1)

    stats_tsv = FAIRSEQ2_DIR / "language_distribution_0.tsv"
    if not stats_tsv.exists():
        logger.error(f"Stats TSV not found: {stats_tsv} — run convert_to_fairseq2.py first")
        sys.exit(1)

    try:
        import torch

        if not torch.cuda.is_available():
            logger.error("CUDA not available")
            sys.exit(1)
    except ImportError:
        logger.error("PyTorch not installed")
        sys.exit(1)


def main() -> None:
    default_config = PROJECT_DIR / "configs" / "fairseq2" / "ctc-finetune-hpc.yaml"

    parser = argparse.ArgumentParser(description="W&B sweep agent wrapper for omniASR")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=default_config,
        help="Base fairseq2 config to patch with sweep hyperparameters",
    )
    parser.add_argument("--extra-args", type=str, default="", help="Additional args passed to fairseq2 recipe")
    args = parser.parse_args()

    setup_logging("sweep_agent")
    setup_hpc_environment()
    log_system_info()
    log_gpu_info()

    _check_prerequisites(args.base_config)
    _ensure_data_symlink()

    # --- W&B init (sweep controller populates wandb.config) ---
    import wandb

    run = wandb.init()
    sweep_config = dict(wandb.config)
    logger.info(f"W&B sweep run: {run.url}")
    logger.info(f"Sweep hyperparameters: {sweep_config}")

    # --- Build patched config ---
    with args.base_config.open() as f:
        base_config = yaml.safe_load(f) or {}

    patched = _apply_sweep_overrides(base_config, sweep_config)

    # Per-run output directory
    output_dir = OUTPUT_DIR / "sweeps" / run.id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write patched config for this run
    run_config_path = output_dir / "sweep_config.yaml"
    with run_config_path.open("w") as f:
        yaml.safe_dump(patched, f, default_flow_style=False)
    logger.info(f"Patched config written to: {run_config_path}")

    # Log full config to W&B
    wandb.config.update({"fairseq2": patched, "base_config": str(args.base_config)})
    wandb.save(str(run_config_path), base_path=str(output_dir), policy="now")

    # --- Run fairseq2 recipe ---
    cmd = [
        sys.executable,
        "-m",
        "workflows.recipes.wav2vec2.asr",
        str(output_dir),
        "--config-file",
        str(run_config_path),
    ]
    if args.extra_args:
        cmd.extend(shlex.split(args.extra_args))

    logger.info(f"Command: {' '.join(cmd)}")

    try:
        start_time = time.time()
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(PROJECT_DIR),
        )
        logger.info(f"Training subprocess started (PID={process.pid})")

        last_heartbeat = time.time()
        for line_count, line in enumerate(process.stdout, 1):
            line = line.rstrip()
            logger.info(f"[fairseq2] {line}")
            log_line_to_wandb(line, run)

            now = time.time()
            if now - last_heartbeat >= _HEARTBEAT_INTERVAL:
                elapsed = now - start_time
                logger.info(f"[heartbeat] Job alive — elapsed {elapsed:.0f}s, lines={line_count}")
                last_heartbeat = now

        return_code = process.wait()
        elapsed = time.time() - start_time

        if return_code < 0:
            sig_num = -return_code
            sig_map = {s.value: s.name for s in signal.Signals}
            sig_name = sig_map.get(sig_num, str(sig_num))
            return_code = 128 + sig_num
            logger.error(f"Training KILLED by signal {sig_name} after {elapsed / 3600:.1f}h")
        elif return_code != 0:
            logger.error(f"Training FAILED after {elapsed / 3600:.1f}h (exit code: {return_code})")
        else:
            logger.info(f"Training completed successfully in {elapsed / 3600:.1f}h")

        # Summary
        checkpoints = sorted(output_dir.glob("**/*.pt"))
        wandb.summary["exit_code"] = return_code
        wandb.summary["elapsed_hours"] = round(elapsed / 3600, 2)
        wandb.summary["num_checkpoints"] = len(checkpoints)

        if checkpoints:
            logger.info(f"Checkpoints: {len(checkpoints)}")
            for ckpt in checkpoints:
                logger.info(f"  {ckpt}")

    except Exception as e:
        logger.exception(f"Sweep run failed: {e}")
        wandb.finish(exit_code=1)
        sys.exit(1)

    wandb.finish(exit_code=return_code)
    sys.exit(return_code)


if __name__ == "__main__":
    main()
