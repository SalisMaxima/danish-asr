# HF Baseline Training: Wav2Vec2 & Whisper

Fine-tune Wav2Vec2-XLS-R-300M (CTC) and Whisper-large-v3 (seq2seq) on CoRal-v3 Danish data using HuggingFace Trainer, to produce baselines comparable to CoRal's published benchmarks.

## Target Benchmarks (CoRal)

| Model | read_aloud CER | conversation CER |
|-------|---------------|-----------------|
| Wav2Vec2-XLS-R-300M | 5.9% | 13.7% |
| Whisper-large-v3 | 4.5% | 11.6% |

## Prerequisites

1. **Preprocessed Parquet data** on HPC at `/work3/$USER/data/preprocessed/`
   - Subsets: `read_aloud/`, `conversation/`
   - Splits: `train/`, `validation/`, `test/`
2. **Vocab file**: `configs/vocab/coral_wav2vec2_ctc.json` (72 tokens)
3. **HPC access**: DTU LSF cluster with A100 GPUs
4. **W&B login**: `wandb login` in your venv

## Quick Start — Smoke Tests

Run smoke tests first to validate the pipeline (<15 min each):

```bash
# Wav2Vec2 smoke (500 steps, ~10 min)
bsub < scripts/hpc/09_smoke_wav2vec2.sh

# Whisper smoke (500 steps, ~15 min)
bsub < scripts/hpc/10_smoke_whisper.sh
```

Monitor with `bpeek -f <jobid>` or check W&B dashboard.

## Full Training

### Wav2Vec2 (~57h, needs 3x24h jobs)

```bash
# First submission
bsub < scripts/hpc/07_train_wav2vec2.sh

# After first job finishes, resume from checkpoint
RESUME_CKPT=latest bsub < scripts/hpc/07_train_wav2vec2.sh

# Third submission if needed
RESUME_CKPT=latest bsub < scripts/hpc/07_train_wav2vec2.sh
```

### Whisper (~17h, fits in one 24h job)

```bash
bsub < scripts/hpc/08_train_whisper.sh
```

## Resuming Training

The scripts support checkpoint resume via the `RESUME_CKPT` environment variable:

- **`RESUME_CKPT=latest`** — HF Trainer auto-detects the latest `checkpoint-*` directory in the output dir
- **`RESUME_CKPT=/path/to/checkpoint-NNNNN`** — Resume from a specific checkpoint

W&B run continuity is handled by `--wandb-resume allow` (default for full training).

## Configuration

Configs are in `configs/hf_baseline/`:

| Config | Steps | Batch (eff.) | Eval freq | Subsets |
|--------|-------|-------------|-----------|---------|
| `wav2vec2_full.yaml` | 100k | 4×8 = 32 | 2k steps | both |
| `wav2vec2_smoke.yaml` | 500 | 4×2 = 8 | 250 steps | read_aloud |
| `whisper_full.yaml` | 30k | 2×16 = 32 | 2k steps | both |
| `whisper_smoke.yaml` | 500 | 2×4 = 8 | 250 steps | read_aloud |

## Monitoring

- **W&B**: Metrics logged automatically via HF Trainer callback (`report_to="wandb"`)
- **LSF logs**: `/work3/$USER/logs/lsf/w2v2_full_<jobid>.{out,err}`
- **Python logs**: `/work3/$USER/logs/python/train_wav2vec2_*.log`
- **Live output**: `bpeek -f <jobid>`

Key W&B metrics:
- `eval/wer`, `eval/cer` — validation error rates
- `eval/loss` — validation loss
- `train/loss` — training loss

## Resource Requirements

| Model | Queue | Memory | Walltime | GPU VRAM |
|-------|-------|--------|----------|----------|
| Wav2Vec2 (315M) | gpua100 | 16GB | 24h/job | ~15GB |
| Whisper (1.5B) | gpua100 | 20GB | 24h | ~25GB |

## Troubleshooting

### CUDA OOM on Whisper

Reduce batch size and increase gradient accumulation in `whisper_full.yaml`:
```yaml
per_device_train_batch_size: 1    # was 2
gradient_accumulation_steps: 32   # was 16 (keeps effective batch = 32)
```

### Data path not found

Ensure preprocessed Parquet exists:
```bash
ls /work3/$USER/data/preprocessed/read_aloud/train/part-*.parquet
ls /work3/$USER/data/preprocessed/conversation/train/part-*.parquet
```

### W&B connectivity issues

If W&B fails to init, training continues without logging. Check:
```bash
wandb login --verify
```

### CTC outputs all blanks

Normal for the first ~500 steps of Wav2Vec2 CTC training. WER will be 1.0 initially, then drop rapidly. First meaningful eval at step 2000.
