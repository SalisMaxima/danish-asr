"""Step 4: Evaluation wrapper for omniASR on HPC.

Runs the fairseq2 eval recipe on a trained checkpoint with full logging.

``--checkpoint-dir`` is the eval *output workspace* — fairseq2 writes eval artifacts
there. It is NOT the trained checkpoint location. The checkpoint to evaluate is
resolved from ``model.path`` in the config file.

Usage:
    python scripts/hpc/run_eval.py \
        --checkpoint-dir /work3/$USER/outputs/omniasr_e2_eval \
        --config configs/fairseq2/ctc-eval-e2.yaml
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import yaml
from loguru import logger

from scripts.hpc.common import (
    PROJECT_DIR,
    log_gpu_info,
    log_system_info,
    setup_hpc_environment,
    setup_logging,
)

_HEARTBEAT_INTERVAL = 300  # seconds between heartbeat log lines


def check_prerequisites(checkpoint_dir: Path, config: Path) -> None:
    """Verify evaluation prerequisites.

    ``checkpoint_dir`` is the eval output workspace, not the checkpoint source.
    The checkpoint to evaluate is resolved from ``model.path`` in the config.
    """
    if not checkpoint_dir.exists():
        logger.error(f"Eval workspace directory not found: {checkpoint_dir}")
        sys.exit(1)

    if not config.exists():
        logger.error(f"Config not found: {config}")
        sys.exit(1)

    # Validate the checkpoint path set via model.path in the config.
    try:
        raw = config.read_text()
    except OSError as e:
        logger.error(f"Cannot read config file {config}: {e}")
        sys.exit(1)

    try:
        cfg = yaml.safe_load(raw)
        if isinstance(cfg, dict):
            model_path_str = cfg.get("model", {}).get("path")
            if model_path_str:
                model_path = Path(model_path_str)
                if not model_path.exists():
                    logger.error(f"Checkpoint not found: {model_path}")
                    logger.error(f"Path is configured via model.path in {config}")
                    sys.exit(1)
                logger.info(f"Checkpoint verified: {model_path}")
    except yaml.YAMLError as e:
        logger.warning(f"Config is not valid YAML ({e}) — skipping model.path check; fairseq2 will validate on launch")

    try:
        import torch

        if not torch.cuda.is_available():
            logger.error("CUDA not available — GPU required for evaluation")
            sys.exit(1)
    except ImportError:
        logger.error("PyTorch not installed")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="omniASR evaluation wrapper")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Eval output workspace (fairseq2 writes artifacts here; checkpoint is set via model.path in config)",
    )
    parser.add_argument("--config", type=Path, required=True, help="fairseq2 eval config file (must set model.path)")
    parser.add_argument("--extra-args", type=str, default="", help="Additional args passed to fairseq2 eval recipe")
    parser.add_argument("--wandb-project", type=str, default="danish-asr", help="W&B project name")
    parser.add_argument(
        "--wandb-run-id", type=str, default="", help="W&B run ID to resume (links eval to training run)"
    )
    parser.add_argument("--wandb-tags", type=str, default="", help="Comma-separated W&B tags for this eval run")
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

        extra_tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        wandb_run = wandb.init(
            project=args.wandb_project,
            id=args.wandb_run_id or None,
            resume="allow" if args.wandb_run_id else None,
            job_type="eval",
            tags=["eval", "hpc"] + extra_tags,
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
        for line_count, line in enumerate(process.stdout, 1):
            line = line.rstrip()
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
                logger.info(f"[heartbeat] Job alive — elapsed {elapsed:.0f}s, lines_logged={line_count}")
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
        if return_code == 0 and wer_value is None and cer_value is None:
            logger.warning(
                "Subprocess exited 0 but no WER or CER found — recipe may have completed without computing metrics; verify output manually"
            )
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
