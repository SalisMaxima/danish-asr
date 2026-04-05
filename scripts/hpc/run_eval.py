"""Step 4: Evaluation wrapper for omniASR on HPC.

Runs the fairseq2 eval recipe on a trained checkpoint with full logging.

``--checkpoint-dir`` is the eval *output workspace* — fairseq2 writes eval artifacts
there. It is NOT the trained checkpoint location. The checkpoint to evaluate is
resolved from ``model.path`` in the config file.

Score file handling
-------------------
fairseq2 stores validation scores alongside checkpoints at:
  {model_path.parent}/scores/{model_path.name}.txt

These score files are written during training (on the dev split) and are reused by
the eval recipe to decide whether evaluation is already done. If a score file exists,
the recipe silently no-ops — even if the eval config targets a different split (e.g.
``valid_split: "test"``).

This wrapper renames any existing score file to a ``.val.bak`` sibling (for example,
``{name}.txt`` → ``{name}.val.bak``) before invoking the recipe, so the recipe
always runs fresh on whatever split the config specifies. After the recipe writes
the fresh score to ``{name}.txt``, this wrapper reads it directly (the score is
stored as a negative float: WER = abs(score)).

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


def check_prerequisites(checkpoint_dir: Path, config: Path) -> Path | None:
    """Verify evaluation prerequisites.

    Returns the resolved model_path (from model.path in the config), or None
    if model.path is not set. Exits on any unrecoverable error.

    ``checkpoint_dir`` is the eval output workspace, not the checkpoint source.
    The checkpoint to evaluate is resolved from ``model.path`` in the config.
    """
    if not checkpoint_dir.exists():
        logger.error(f"Eval workspace directory not found: {checkpoint_dir}")
        sys.exit(1)

    if not config.exists():
        logger.error(f"Config not found: {config}")
        sys.exit(1)

    model_path: Path | None = None

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
            else:
                logger.warning(
                    f"model.path not set in {config} — fairseq2 will attempt to resolve the checkpoint itself and may fail"
                )
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

    return model_path


def _get_score_file(model_path: Path) -> Path:
    """Return the fairseq2 score file path for a checkpoint directory.

    fairseq2 writes checkpoint scores to:
      {model_path.parent}/scores/{model_path.name}.txt

    e.g. for model_path = .../ws_1.abc/checkpoints/step_40000:
      → .../ws_1.abc/checkpoints/scores/step_40000.txt
    """
    return model_path.parent / "scores" / f"{model_path.name}.txt"


def _backup_score_file(score_file: Path) -> Path | None:
    """Rename an existing score file to a unique .val.bak sibling so the recipe re-runs.

    Returns the backup path if a backup was made, else None.
    """
    if not score_file.exists():
        return None

    backup = score_file.with_suffix(".val.bak")
    if backup.exists():
        counter = 1
        while True:
            candidate = score_file.with_suffix(f".val.{counter}.bak")
            if not candidate.exists():
                backup = candidate
                break
            counter += 1

    score_file.rename(backup)
    logger.info(f"Renamed existing score file to {backup.name} — recipe will run fresh on the configured split")
    return backup


def _read_score_file(score_file: Path) -> float | None:
    """Read WER from a fairseq2 score file.

    fairseq2 stores scores as negative floats (higher-is-better convention).
    WER = abs(stored_value).
    Returns None if the file does not exist or cannot be parsed.
    """
    if not score_file.exists():
        return None
    try:
        raw = score_file.read_text().strip()
        return abs(float(raw))
    except (ValueError, OSError, UnicodeError) as e:
        logger.warning(f"Could not read score file {score_file}: {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="omniASR evaluation wrapper")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Eval output workspace (fairseq2 writes artifacts here; checkpoint is set via model.path in config)",
    )
    parser.add_argument("--config", type=Path, required=True, help="fairseq2 eval config file (should set model.path)")
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

    model_path = check_prerequisites(args.checkpoint_dir, args.config)
    logger.info(f"Eval workspace:        {args.checkpoint_dir}")
    logger.info(f"Config:                {args.config}")

    # Rename any existing score file so the recipe runs fresh on the configured split
    # (fairseq2 silently no-ops if a score file already exists for the checkpoint).
    score_file: Path | None = None
    if model_path is not None:
        score_file = _get_score_file(model_path)
        _backup_score_file(score_file)

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

        # If stdout parsing didn't find WER, read directly from the score file.
        # fairseq2 stores scores as negative floats; WER = abs(value).
        if wer_value is None and score_file is not None:
            wer_value = _read_score_file(score_file)
            if wer_value is not None:
                logger.info(f"WER read from score file ({score_file.name}): {wer_value:.4f}%")

        logger.info("=" * 50)
        if wer_value is not None:
            logger.info(f"  WER: {wer_value:.4f}%")
        else:
            logger.warning("  WER: not found in output or score file")
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
