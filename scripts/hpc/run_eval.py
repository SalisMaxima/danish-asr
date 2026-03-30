"""Step 4: Evaluation wrapper for omniASR on HPC.

Runs the fairseq2 eval recipe on a trained checkpoint with full logging.

Usage:
    python scripts/hpc/run_eval.py --checkpoint-dir /work3/$USER/outputs/omniasr_hpc_20260308
    python scripts/hpc/run_eval.py --checkpoint-dir $CHECKPOINT_DIR --config configs/fairseq2/ctc-finetune-hpc.yaml
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

from loguru import logger

from scripts.hpc.common import (
    PROJECT_DIR,
    log_gpu_info,
    log_system_info,
    setup_hpc_environment,
    setup_logging,
)
from scripts.hpc.fairseq2_logging import should_log_fairseq2_line

_HEARTBEAT_INTERVAL = 300  # seconds between heartbeat log lines


def check_prerequisites(checkpoint_dir: Path, config: Path) -> None:
    """Verify evaluation prerequisites."""
    if not checkpoint_dir.exists():
        logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)

    checkpoints = list(checkpoint_dir.glob("**/*.pt"))
    if not checkpoints:
        logger.warning(
            f"No .pt files found in {checkpoint_dir} — fairseq2 may still work with its own checkpoint format"
        )

    if not config.exists():
        logger.error(f"Config not found: {config}")
        sys.exit(1)

    try:
        import torch

        if not torch.cuda.is_available():
            logger.error("CUDA not available — GPU required for evaluation")
            sys.exit(1)
    except ImportError:
        logger.error("PyTorch not installed")
        sys.exit(1)


def main() -> None:
    default_config = PROJECT_DIR / "configs" / "fairseq2" / "ctc-finetune-hpc.yaml"

    parser = argparse.ArgumentParser(description="omniASR evaluation wrapper")
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Directory with trained checkpoint")
    parser.add_argument("--config", type=Path, default=default_config, help="fairseq2 config file")
    parser.add_argument("--extra-args", type=str, default="", help="Additional args passed to fairseq2 eval recipe")
    parser.add_argument("--wandb-project", type=str, default="danish-asr", help="W&B project name")
    parser.add_argument(
        "--wandb-run-id", type=str, default="", help="W&B run ID to resume (links eval to training run)"
    )
    args = parser.parse_args()

    setup_logging("run_eval")
    setup_hpc_environment()
    log_system_info()
    log_gpu_info()

    check_prerequisites(args.checkpoint_dir, args.config)
    logger.info(f"Evaluating checkpoint: {args.checkpoint_dir}")

    # Initialise W&B
    wandb_run = None
    try:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            id=args.wandb_run_id or None,
            resume="allow" if args.wandb_run_id else None,
            job_type="eval",
            tags=["eval", "hpc"],
            config={"checkpoint_dir": str(args.checkpoint_dir), "config_file": str(args.config)},
        )
        logger.info(f"W&B run initialised: {wandb_run.url}")
    except ImportError:
        logger.warning("wandb not installed — skipping W&B logging")
    except Exception as e:
        logger.warning(f"W&B init failed ({type(e).__name__}: {e}) — continuing without it")

    # Build command
    cmd = [
        sys.executable,
        "-m",
        "workflows.recipes.wav2vec2.asr.eval.recipe",
        str(args.checkpoint_dir),
        "--config-file",
        str(args.config),
    ]
    if args.extra_args:
        cmd.extend(shlex.split(args.extra_args))

    logger.info(f"Command: {' '.join(cmd)}")

    try:
        # Run evaluation
        start_time = time.time()
        wer_pattern = re.compile(r"\(WER\):\s*([\d.]+)")
        cer_pattern = re.compile(r"\(CER\):\s*([\d.]+)")
        wer_value = None
        cer_value = None

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(PROJECT_DIR),
            )
        except OSError as e:
            logger.error(f"Failed to launch eval subprocess: {e}")
            logger.error(f"Command was: {' '.join(cmd)}")
            raise RuntimeError("Failed to launch eval subprocess") from e
        logger.info(f"Eval subprocess started (PID={process.pid})")

        last_heartbeat = time.time()
        for lines_read, line in enumerate(process.stdout, 1):
            line = line.rstrip()
            if should_log_fairseq2_line(line):
                logger.info(f"[fairseq2] {line}")

            # Try to parse WER/CER from output
            wer_match = wer_pattern.search(line)
            if wer_match:
                wer_value = float(wer_match.group(1))
            cer_match = cer_pattern.search(line)
            if cer_match:
                cer_value = float(cer_match.group(1))

            now = time.time()
            if now - last_heartbeat >= _HEARTBEAT_INTERVAL:
                elapsed = now - start_time
                logger.info(f"[heartbeat] Job alive — elapsed {elapsed:.0f}s, lines_read={lines_read}")
                last_heartbeat = now

        return_code = process.wait()
        elapsed = time.time() - start_time

        if return_code != 0:
            logger.error(f"Evaluation FAILED after {elapsed / 60:.1f} min (exit code: {return_code})")
        else:
            logger.info(f"Evaluation completed successfully in {elapsed / 60:.1f} min")
        logger.info("=" * 50)
        if wer_value is not None:
            logger.info(f"  WER: {wer_value:.2f}%")
        else:
            logger.warning("  WER: not found in output")
        if cer_value is not None:
            logger.info(f"  CER: {cer_value:.2f}%")
        else:
            logger.warning("  CER: not found in output")
        logger.info("=" * 50)

        if wandb_run is not None:
            try:
                import wandb

                metrics: dict[str, float] = {}
                if wer_value is not None:
                    metrics["test/wer"] = wer_value
                if cer_value is not None:
                    metrics["test/cer"] = cer_value
                if metrics:
                    wandb.log(metrics)
                wandb.summary["exit_code"] = return_code
                wandb.finish(exit_code=return_code)
            except Exception as e:
                logger.warning(f"W&B finish failed: {type(e).__name__}: {e}")

    except Exception as e:
        logger.exception(f"Unhandled exception in eval wrapper: {e}")
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
