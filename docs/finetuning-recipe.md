# CTC Finetuning Recipe

Guide for fine-tuning `omniASR_CTC_300M` on CoRal-v2 Danish data using the omnilingual ASR training recipe.

## Prerequisites

1. Data converted to Parquet format (see [data-preparation.md](data-preparation.md))
2. `omnilingual-asr` package installed
3. fairseq2 asset card created for dataset
4. `language_distribution_0.tsv` stats file generated

## Training Command

```bash
export OUTPUT_DIR="/path/to/checkpoints"

python -m workflows.recipes.wav2vec2.asr $OUTPUT_DIR \
    --config-file workflows/recipes/wav2vec2/asr/configs/ctc-finetune.yaml
```

The recipe reads the config YAML and launches training via fairseq2's trainer.

## Reference Config: ctc-finetune.yaml (upstream default)

```yaml
model:
  name: "omniASR_CTC_300M"

dataset:
  name: "example_dataset"           # → replace with "coral_v2_danish"
  train_split: "train"
  valid_split: "dev"
  storage_mode: "MIXTURE_PARQUET"
  task_mode: "ASR"
  mixture_parquet_storage_config:
    dataset_summary_path: "/path/to/language_distribution_0.tsv"
    beta_corpus: 0.5
    beta_language: 0.5
    fragment_loading:
      cache: true
  asr_task_config:
    min_audio_len: 32_000            # 2s minimum
    max_audio_len: 960_000           # 60s maximum
    max_num_elements: 960_000        # batch size control
    batch_shuffle_window: 1
    normalize_audio: true
    example_shuffle_window: 1

tokenizer:
  name: "omniASR_tokenizer_v1"

optimizer:
  config:
    lr: 5e-05

trainer:
  freeze_encoder_for_n_steps: 0
  mixed_precision:
    dtype: "torch.bfloat16"
  grad_accumulation:
    num_batches: 4

regime:
  num_steps: 20_000
  validate_after_n_steps: 0
  validate_every_n_steps: 1000
  checkpoint_every_n_steps: 1000
  publish_metrics_every_n_steps: 200
```

## Our Adapted Config: ctc-finetune-danish.yaml

Adjusted for a single-GPU setup (12GB local, or single A100 on DTU HPC):

```yaml
model:
  name: "omniASR_CTC_300M"

dataset:
  name: "coral_v2_danish"
  train_split: "train"
  valid_split: "dev"
  storage_mode: "MIXTURE_PARQUET"
  task_mode: "ASR"
  mixture_parquet_storage_config:
    dataset_summary_path: "data/parquet/language_distribution_0.tsv"
    beta_corpus: 0.5
    beta_language: 0.5
    fragment_loading:
      cache: true
  asr_task_config:
    min_audio_len: 32_000            # 2s minimum
    max_audio_len: 480_000           # 30s max (reduced for VRAM)
    max_num_elements: 3_840_000      # reduced batch capacity
    batch_shuffle_window: 1
    normalize_audio: true
    example_shuffle_window: 1

tokenizer:
  name: "omniASR_tokenizer_v1"

optimizer:
  config:
    lr: 1e-05                        # lower LR for finetuning

trainer:
  freeze_encoder_for_n_steps: 0
  mixed_precision:
    dtype: "torch.bfloat16"
  grad_accumulation:
    num_batches: 8                   # higher to compensate smaller batch

regime:
  num_steps: 5_000                   # shorter — single language convergence
  validate_after_n_steps: 0
  validate_every_n_steps: 500
  checkpoint_every_n_steps: 500
  publish_metrics_every_n_steps: 100
```

### Key Differences from Upstream Default

| Parameter | Upstream | Ours | Rationale |
|---|---|---|---|
| `max_audio_len` | 960,000 (60s) | 480,000 (30s) | VRAM constraint on 12GB GPU |
| `max_num_elements` | 960,000 | 3,840,000 | Allow multiple shorter samples per batch |
| `lr` | 5e-05 | 1e-05 | More conservative for finetuning |
| `grad_accumulation` | 4 | 8 | Effective larger batch with limited VRAM |
| `num_steps` | 20,000 | 5,000 | Single language, less data diversity |
| `validate_every_n_steps` | 1,000 | 500 | More frequent validation |
| `checkpoint_every_n_steps` | 1,000 | 500 | More frequent checkpointing |

## Multi-GPU Training (DTU HPC)

The upstream recipe is designed for distributed training:

| Model Size | Recommended GPUs |
|---|---|
| 300M | 32 GPUs (upstream) |
| 1B | 64 GPUs |
| 3B | 96 GPUs |

For our single-language finetuning, far fewer GPUs are needed. On DTU HPC:

**Single A100 (40GB or 80GB):**
```yaml
max_audio_len: 960_000       # can handle 60s
max_num_elements: 7_680_000  # larger batches
grad_accumulation:
  num_batches: 4             # less accumulation needed
```

**2x A100 (via LSF):**
```bash
# Submit multi-GPU job — see dtu-hpc-setup.md for full script
#BSUB -n 8
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -q gpua100
```

fairseq2 handles distributed training internally via `torch.distributed`.

## Evaluation

The recipe includes a built-in eval mode. After training:

```bash
python -m workflows.recipes.wav2vec2.asr.eval.recipe $OUTPUT_DIR \
    --config-file workflows/recipes/wav2vec2/asr/configs/ctc-finetune-danish.yaml
```

For custom evaluation (per-dialect WER), use our `metrics.py`:

```python
from danish_asr.metrics import compute_wer, compute_cer
```

## Hyperparameter Tuning Suggestions

| Hyperparameter | Range to Try | Notes |
|---|---|---|
| `lr` | 5e-06 to 5e-05 | Start low, increase if underfitting |
| `num_steps` | 3,000 to 10,000 | Monitor validation loss for convergence |
| `freeze_encoder_for_n_steps` | 0 to 2,000 | Freeze encoder initially to stabilize |
| `max_audio_len` | 480,000 to 960,000 | Depends on available VRAM |
| `grad_accumulation.num_batches` | 4 to 16 | Trade-off: speed vs effective batch size |

## Checkpoints & Storage

- Each checkpoint: ~1.3 GiB (model size)
- With checkpointing every 500 steps and 5,000 total: **~13 GiB** for checkpoints
- Output directory contains: checkpoints, training logs, metrics

## Training Duration Estimates

| Setup | Estimated Time |
|---|---|
| 1x RTX 3060Ti (12GB) | ~8-12 hours for 5K steps |
| 1x A100 (40GB, DTU) | ~2-4 hours for 5K steps |
| 2x A100 (DTU) | ~1-2 hours for 5K steps |

These are rough estimates — actual times depend on batch size, audio lengths, and I/O.
