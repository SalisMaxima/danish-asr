"""Whisper-large-v3 seq2seq fine-tuning on CoRal-v3 via HuggingFace Seq2SeqTrainer.

Usage::

    python scripts/hpc/train_whisper.py --config configs/hf_baseline/whisper_full.yaml
    python scripts/hpc/train_whisper.py --config configs/hf_baseline/whisper_smoke.yaml
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from danish_asr.text import normalize_coral_text

from .common import OUTPUT_DIR, log_gpu_info, log_system_info, setup_hpc_environment, setup_logging
from .train_common import (
    Seq2SeqDataCollator,
    build_datasets,
    finish_wandb,
    init_wandb,
    load_config,
    make_seq2seq_compute_metrics,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Whisper seq2seq fine-tuning on CoRal-v3")
    parser.add_argument("--config", type=Path, required=True, help="YAML config file")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory")
    parser.add_argument(
        "--resume-from-checkpoint", type=str, default=None, help='Checkpoint path or "latest" for auto-detect'
    )
    parser.add_argument("--wandb-project", type=str, default="danish-asr")
    parser.add_argument("--wandb-name", type=str, default="")
    parser.add_argument("--wandb-tags", type=str, default="hpc,whisper,seq2seq")
    parser.add_argument("--wandb-resume", type=str, default="allow", choices=["allow", "never", "must"])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # --- Setup ---
    setup_logging("train_whisper")
    setup_hpc_environment()
    log_system_info()
    log_gpu_info()

    cfg = load_config(args.config)
    logger.info(f"Config: {args.config}")

    # --- Output directory ---
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = OUTPUT_DIR / f"whisper_{timestamp}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output: {output_dir}")

    # --- Processor ---
    processor = WhisperProcessor.from_pretrained(cfg["model_name"])
    logger.info(f"Processor: {cfg['model_name']}")

    # --- Datasets ---
    train_dataset, val_dataset = build_datasets(
        processor=processor,
        tokenizer=processor.tokenizer,
        text_normalizer=normalize_coral_text,
        subsets=cfg.get("subsets", ["read_aloud", "conversation"]),
        max_duration=cfg.get("max_duration", 30.0),
        sample_rate=cfg.get("sample_rate", 16000),
    )

    # --- Model ---
    model = WhisperForConditionalGeneration.from_pretrained(cfg["model_name"])

    # Configure for Danish transcription
    model.generation_config.language = cfg.get("language", "da")
    model.generation_config.task = cfg.get("task", "transcribe")
    model.generation_config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {total:,} total, {trainable:,} trainable")

    # --- W&B ---
    tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
    wandb_run = init_wandb(
        project=args.wandb_project,
        name=args.wandb_name or None,
        tags=tags,
        config={"config_file": str(args.config), "hf_baseline": cfg},
        config_path=args.config,
        resume=args.wandb_resume,
    )

    # --- Training arguments ---
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 16),
        max_steps=cfg["max_steps"],
        learning_rate=cfg.get("learning_rate", 1e-5),
        warmup_steps=cfg.get("warmup_steps", 500),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "linear"),
        weight_decay=cfg.get("weight_decay", 0.01),
        bf16=cfg.get("bf16", True),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        predict_with_generate=cfg.get("predict_with_generate", True),
        generation_max_length=cfg.get("generation_max_length", 225),
        logging_steps=cfg.get("logging_steps", 100),
        eval_strategy="steps",
        eval_steps=cfg.get("eval_steps", 2000),
        save_strategy="steps",
        save_steps=cfg.get("save_steps", 2000),
        save_total_limit=cfg.get("save_total_limit", 3),
        load_best_model_at_end=True,
        metric_for_best_model=cfg.get("metric_for_best_model", "cer"),
        greater_is_better=False,
        report_to="wandb" if wandb_run else "none",
        dataloader_num_workers=cfg.get("dataloader_num_workers", 4),
        remove_unused_columns=False,
        seed=cfg.get("seed", 42),
    )

    # --- Trainer ---
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=Seq2SeqDataCollator(processor),
        compute_metrics=make_seq2seq_compute_metrics(processor),
        processing_class=processor,
    )

    # --- Train ---
    resume_ckpt = args.resume_from_checkpoint
    if resume_ckpt == "latest":
        existing_ckpts = sorted(output_dir.glob("checkpoint-*"))
        if not existing_ckpts:
            logger.error(
                f"--resume-from-checkpoint latest requested but no checkpoint-* "
                f"directories found in {output_dir}. "
                f"Omit --resume-from-checkpoint to start fresh, or point --output-dir "
                f"at a directory that already contains checkpoints."
            )
            finish_wandb(wandb_run, exit_code=1)
            sys.exit(1)
        logger.info(f"Resuming from latest checkpoint: {existing_ckpts[-1].name}")
        resume_ckpt = True  # HF Trainer auto-detects latest checkpoint in output_dir

    logger.info(f"Starting training (max_steps={cfg['max_steps']}, resume={resume_ckpt})")

    try:
        trainer.train(resume_from_checkpoint=resume_ckpt)
    except KeyboardInterrupt:
        logger.warning("Training interrupted (SIGINT) — model NOT saved")
        finish_wandb(wandb_run, exit_code=130)
        sys.exit(130)
    except Exception:
        logger.exception("Training failed")
        finish_wandb(wandb_run, exit_code=1)
        sys.exit(1)

    # --- Save ---
    try:
        logger.info("Saving model and processor")
        trainer.save_model(str(output_dir / "final"))
        processor.save_pretrained(str(output_dir / "final"))
    except OSError:
        logger.exception(
            f"Failed to save model to {output_dir / 'final'} — check disk space on /work3/. "
            f"Training completed but final checkpoint was NOT saved. "
            f"Best checkpoint may still be available in {output_dir}."
        )
        finish_wandb(wandb_run, exit_code=1)
        sys.exit(1)

    # --- Evaluate on validation ---
    try:
        logger.info("Running final evaluation")
        metrics = trainer.evaluate()
        logger.info(f"Final eval metrics: {metrics}")
    except Exception:
        logger.exception("Final evaluation failed after successful training")
        finish_wandb(wandb_run, exit_code=1)
        sys.exit(1)

    finish_wandb(wandb_run)
    logger.info("Done")


if __name__ == "__main__":
    main()
