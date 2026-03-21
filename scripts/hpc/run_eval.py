"""Step 4: Evaluation wrapper for omniASR on HPC.

Runs the fairseq2 eval recipe on a trained checkpoint with full logging.

Usage:
    python scripts/hpc/run_eval.py --checkpoint-dir /work3/s204696/outputs/omniasr_hpc_20260308
    python scripts/hpc/run_eval.py --checkpoint-dir $CHECKPOINT_DIR --config configs/fairseq2/ctc-finetune-hpc.yaml
"""

from __future__ import annotations

import argparse
import re
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
    args = parser.parse_args()

    setup_logging("run_eval")
    setup_hpc_environment()
    log_system_info()
    log_gpu_info()

    check_prerequisites(args.checkpoint_dir, args.config)
    logger.info(f"Evaluating checkpoint: {args.checkpoint_dir}")

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
        cmd.extend(args.extra_args.split())

    logger.info(f"Command: {' '.join(cmd)}")

    # Run evaluation
    start_time = time.time()
    wer_pattern = re.compile(r"WER[:\s]+(\d+\.?\d*)", re.IGNORECASE)
    cer_pattern = re.compile(r"CER[:\s]+(\d+\.?\d*)", re.IGNORECASE)
    wer_value = None
    cer_value = None

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(PROJECT_DIR),
    )

    for line in process.stdout:
        line = line.rstrip()
        logger.info(f"[fairseq2] {line}")

        # Try to parse WER/CER from output
        wer_match = wer_pattern.search(line)
        if wer_match:
            wer_value = float(wer_match.group(1))
        cer_match = cer_pattern.search(line)
        if cer_match:
            cer_value = float(cer_match.group(1))

    return_code = process.wait()
    elapsed = time.time() - start_time

    # Summary
    logger.info(f"Evaluation finished in {elapsed / 60:.1f} min (exit code: {return_code})")
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

    sys.exit(return_code)


if __name__ == "__main__":
    main()
