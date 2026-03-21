"""Training wrapper for omniASR on HPC.

Sets up environment, creates data symlink, and runs the fairseq2 training recipe
with full logging of stdout/stderr and W&B metric tracking.

Usage:
    python scripts/hpc/run_training.py
    python scripts/hpc/run_training.py --config configs/fairseq2/ctc-finetune-hpc.yaml
    python scripts/hpc/run_training.py --config configs/fairseq2/ctc-finetune-smoke.yaml --wandb-tags smoke,hpc
"""

from __future__ import annotations

import argparse
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

from scripts.hpc.common import (
    FAIRSEQ2_DIR,
    OUTPUT_DIR,
    PROJECT_DIR,
    SCRATCH_DIR,
    log_gpu_info,
    log_system_info,
    setup_hpc_environment,
    setup_logging,
)

# Fairseq2 metric extraction patterns — validated against smoke test output.
# Fairseq2 CTC typically logs lines like:
#   | train | step 100 | loss 1.234 | ...
#   | valid | step 500 | wer 0.42 | cer 0.12 | ...
# All fairseq2 stdout lines are forwarded to INFO; these patterns are used
# best-effort to extract scalar metrics for W&B logging.
_LOSS_PATTERN = re.compile(r"\bloss[:\s]+([\d.]+)", re.IGNORECASE)
_WER_PATTERN = re.compile(r"\bwer[:\s]+([\d.]+)", re.IGNORECASE)
_CER_PATTERN = re.compile(r"\bcer[:\s]+([\d.]+)", re.IGNORECASE)
_STEP_PATTERN = re.compile(r"\bstep[:\s]+(\d+)", re.IGNORECASE)

_HEARTBEAT_INTERVAL = 300  # seconds between heartbeat log lines


def ensure_data_symlink() -> None:
    """Create symlink: PROJECT_DIR/data → SCRATCH_DIR/data so fairseq2 relative paths resolve."""
    link_path = PROJECT_DIR / "data"
    target = SCRATCH_DIR / "data"

    if link_path.is_symlink():
        current_target = link_path.resolve()
        if current_target == target.resolve():
            logger.info(f"Data symlink already exists: {link_path} → {target}")
            return
        logger.warning(f"Removing stale symlink: {link_path} → {current_target}")
        link_path.unlink()
    elif link_path.exists():
        logger.error(f"{link_path} exists and is not a symlink — cannot create data symlink")
        sys.exit(1)

    link_path.symlink_to(target)
    logger.info(f"Created data symlink: {link_path} → {target}")


def check_prerequisites(config: Path) -> None:
    """Verify training prerequisites."""
    if not config.exists():
        logger.error(f"Config not found: {config}")
        sys.exit(1)

    stats_tsv = FAIRSEQ2_DIR / "language_distribution_0.tsv"
    if not stats_tsv.exists():
        logger.error(f"Stats TSV not found: {stats_tsv}")
        logger.error("Run convert_to_fairseq2.py first.")
        sys.exit(1)

    # Check at least one corpus directory exists
    corpus_dirs = list(FAIRSEQ2_DIR.glob("corpus=*"))
    if not corpus_dirs:
        logger.error(f"No corpus directories in {FAIRSEQ2_DIR}")
        logger.error("Run convert_to_fairseq2.py first.")
        sys.exit(1)

    try:
        import torch

        if not torch.cuda.is_available():
            logger.error("CUDA not available — GPU required for training")
            sys.exit(1)
    except ImportError:
        logger.error("PyTorch not installed")
        sys.exit(1)

    try:
        import workflows.recipes.wav2vec2.asr  # noqa: F401

        logger.info("fairseq2 recipe module: OK")
    except ImportError as e:
        logger.error(f"Cannot import training recipe: {e}")
        logger.error("omnilingual-asr not available. Activate your venv and run: uv sync --group omni")
        sys.exit(1)


def _init_wandb(args: argparse.Namespace, config: Path) -> object | None:
    """Initialise W&B run. Returns the run object or None if W&B is unavailable."""
    try:
        import wandb

        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or None,
            tags=tags or None,
            config={"config_file": str(config), "output_dir": str(args.output_dir)},
            resume=args.wandb_resume,
        )
        logger.info(f"W&B run initialised: {run.url}")
        return run
    except ImportError:
        logger.warning("wandb not installed — skipping W&B logging")
        return None
    except Exception as e:
        logger.warning(f"W&B init failed ({type(e).__name__}: {e}) — continuing without it")
        return None


def _log_line_to_wandb(line: str, wandb_run: object | None) -> None:
    """Parse a fairseq2 output line and log matching metrics to W&B."""
    if wandb_run is None:
        return
    try:
        import wandb

        metrics: dict[str, float] = {}
        step: int | None = None

        step_match = _STEP_PATTERN.search(line)
        if step_match:
            step = int(step_match.group(1))

        if "train" in line.lower():
            loss_match = _LOSS_PATTERN.search(line)
            if loss_match:
                metrics["train/loss"] = float(loss_match.group(1))
        elif "valid" in line.lower():
            wer_match = _WER_PATTERN.search(line)
            cer_match = _CER_PATTERN.search(line)
            loss_match = _LOSS_PATTERN.search(line)
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


def main() -> None:
    default_config = PROJECT_DIR / "configs" / "fairseq2" / "ctc-finetune-hpc.yaml"

    parser = argparse.ArgumentParser(description="omniASR training wrapper")
    parser.add_argument("--config", type=Path, default=default_config, help="fairseq2 config file")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (auto-generated if omitted)")
    parser.add_argument("--extra-args", type=str, default="", help="Additional args passed to fairseq2 recipe")
    parser.add_argument("--wandb-project", type=str, default="danish-asr", help="W&B project name")
    parser.add_argument("--wandb-name", type=str, default="", help="W&B run name (auto-generated if omitted)")
    parser.add_argument("--wandb-tags", type=str, default="hpc", help="Comma-separated W&B tags")
    parser.add_argument(
        "--wandb-resume",
        type=str,
        default="allow",
        choices=["allow", "never", "must"],
        help="W&B resume mode: 'never' for smoke tests, 'allow' for full training",
    )
    args = parser.parse_args()

    setup_logging("run_training")
    setup_hpc_environment()
    log_system_info()
    log_gpu_info()

    check_prerequisites(args.config)
    ensure_data_symlink()

    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = OUTPUT_DIR / f"omniasr_hpc_{timestamp}"
    args.output_dir = output_dir  # make available to W&B init
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    wandb_run = _init_wandb(args, args.config)

    cmd = [
        sys.executable,
        "-m",
        "workflows.recipes.wav2vec2.asr",
        str(output_dir),
        "--config-file",
        str(args.config),
    ]
    if args.extra_args:
        cmd.extend(shlex.split(args.extra_args))

    logger.info(f"Command: {' '.join(cmd)}")

    try:
        start_time = time.time()
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(PROJECT_DIR),
            )
        except OSError as e:
            logger.error(f"Failed to launch training subprocess: {e}")
            logger.error(f"Command was: {' '.join(cmd)}")
            sys.exit(1)

        logger.info(f"Training subprocess started (PID={process.pid})")

        last_heartbeat = time.time()
        for line_count, line in enumerate(process.stdout, 1):
            line = line.rstrip()
            logger.info(f"[fairseq2] {line}")

            _log_line_to_wandb(line, wandb_run)

            now = time.time()
            if now - last_heartbeat >= _HEARTBEAT_INTERVAL:
                elapsed = now - start_time
                logger.info(f"[heartbeat] Job alive — elapsed {elapsed:.0f}s, lines_logged={line_count}")
                last_heartbeat = now

        return_code = process.wait()
        elapsed = time.time() - start_time

        if return_code < 0:
            sig_num = -return_code
            sig_map = {s.value: s.name for s in signal.Signals}
            sig_name = sig_map.get(sig_num, str(sig_num))
            return_code = 128 + sig_num
            logger.error(f"Training KILLED by signal {sig_name} after {elapsed / 3600:.1f}h (exit code: {return_code})")
            logger.error("If SIGKILL: likely OOM. Check GPU stats CSV and increase rusage[mem] or reduce batch size.")
        elif return_code != 0:
            logger.error(f"Training FAILED after {elapsed / 3600:.1f}h (exit code: {return_code})")
        else:
            logger.info(f"Training completed successfully in {elapsed / 3600:.1f}h")

        # List checkpoints
        checkpoints = sorted(output_dir.glob("**/*.pt"))
        if checkpoints:
            logger.info(f"Checkpoints found ({len(checkpoints)}):")
            for ckpt in checkpoints:
                logger.info(f"  {ckpt}")
        else:
            logger.warning("No checkpoint files found in output directory")

        if wandb_run is not None:
            try:
                import wandb

                wandb.summary["exit_code"] = return_code
                wandb.summary["elapsed_hours"] = round(elapsed / 3600, 2)
                wandb.summary["num_checkpoints"] = len(checkpoints)
                wandb.finish(exit_code=return_code)
            except Exception as e:
                logger.warning(f"W&B finish failed: {type(e).__name__}: {e}")

    except Exception as e:
        logger.exception(f"Unhandled exception in training wrapper: {e}")
        if wandb_run is not None:
            try:
                import wandb

                wandb.finish(exit_code=1)
            except Exception as wandb_err:
                logger.debug(f"W&B cleanup also failed: {type(wandb_err).__name__}: {wandb_err}")
        sys.exit(1)

    sys.exit(return_code)


if __name__ == "__main__":
    main()
