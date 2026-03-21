"""Step 3: Training wrapper for omniASR on HPC.

Sets up environment, creates data symlink, and runs the fairseq2 training recipe
with full logging of stdout/stderr.

Usage:
    python scripts/hpc/run_training.py
    python scripts/hpc/run_training.py --config configs/fairseq2/ctc-finetune-hpc.yaml
"""

from __future__ import annotations

import argparse
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


def main() -> None:
    default_config = PROJECT_DIR / "configs" / "fairseq2" / "ctc-finetune-hpc.yaml"

    parser = argparse.ArgumentParser(description="omniASR training wrapper")
    parser.add_argument("--config", type=Path, default=default_config, help="fairseq2 config file")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (auto-generated if omitted)")
    parser.add_argument("--extra-args", type=str, default="", help="Additional args passed to fairseq2 recipe")
    args = parser.parse_args()

    setup_logging("run_training")
    setup_hpc_environment()
    log_system_info()
    log_gpu_info()

    check_prerequisites(args.config)
    ensure_data_symlink()

    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = OUTPUT_DIR / f"omniasr_hpc_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Build command
    cmd = [
        sys.executable,
        "-m",
        "workflows.recipes.wav2vec2.asr",
        str(output_dir),
        "--config-file",
        str(args.config),
    ]
    if args.extra_args:
        cmd.extend(args.extra_args.split())

    logger.info(f"Command: {' '.join(cmd)}")

    # Run training
    start_time = time.time()
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

    return_code = process.wait()
    elapsed = time.time() - start_time

    # Post-training summary
    logger.info(f"Training finished in {elapsed / 3600:.1f}h (exit code: {return_code})")

    # Log GPU memory stats
    try:
        import torch

        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
            logger.info(f"Peak GPU memory: {peak_mem:.1f} GB")
    except Exception:
        pass

    # List checkpoints
    checkpoints = sorted(output_dir.glob("**/*.pt"))
    if checkpoints:
        logger.info(f"Checkpoints found ({len(checkpoints)}):")
        for ckpt in checkpoints:
            logger.info(f"  {ckpt}")
    else:
        logger.warning("No checkpoint files found in output directory")

    sys.exit(return_code)


if __name__ == "__main__":
    main()
