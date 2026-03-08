# Training Infrastructure

Configuration and rationale for training three ASR models across two hardware targets.

## Models

| Model | Params | Strategy | Framework | Config group |
|---|---|---|---|---|
| `omniASR_CTC_300M_v2` | 325M | Full fine-tuning | fairseq2 recipe | `configs/fairseq2/` |
| Whisper Large v3 | 1.55B | LoRA (PEFT) | PyTorch Lightning + HF | `model=whisper train=whisper` |
| Wav2Vec2-XLSR-53 | 300M | LoRA (PEFT) | PyTorch Lightning + HF | `model=wav2vec2 train=wav2vec2` |

omniASR uses a separate fairseq2 pipeline (not Hydra). Whisper and Wav2Vec2 share the same Lightning trainer driven by Hydra configs.

---

## Hardware Profiles

### Local — RTX 3080 12GB

```yaml
# configs/hardware/local.yaml
gradient_checkpointing: true   # saves ~35% VRAM at ~25% compute cost
precision: bf16-mixed
num_workers: 4
accelerator: auto
devices: 1
max_duration: 30.0             # clips longer than 30s are truncated
```

### HPC — A100 40/80GB (DTU `gpua100` queue)

```yaml
# configs/hardware/hpc.yaml
gradient_checkpointing: false  # enough VRAM headroom, skip the overhead
precision: bf16-mixed
num_workers: 8
accelerator: auto
devices: 1
max_duration: 30.0
```

---

## VRAM Budget

### Wav2Vec2-XLSR-53 LoRA

| Setting | Local (12GB) | HPC A100 |
|---|---|---|
| Batch size | 8 | 32 |
| Gradient accumulation | 4 | 1 |
| Effective batch size | 32 | 32 |
| Gradient checkpointing | ON | OFF |
| Training VRAM | ~10–11 GB | ~18–20 GB |

### Whisper Large v3 LoRA

| Setting | Local (12GB) | HPC A100 |
|---|---|---|
| Batch size | 4 | 16 |
| Gradient accumulation | 8 | 2 |
| Effective batch size | 32 | 32 |
| Gradient checkpointing | ON | OFF |
| Training VRAM | ~11–12 GB | ~28–32 GB |

### omniASR CTC (fairseq2, bucket-based batching)

| Setting | Local (12GB) | HPC A100 |
|---|---|---|
| `max_audio_len` | 480,000 (30s) | 960,000 (60s) |
| `max_num_elements` | 3,840,000 | 7,680,000 |
| `grad_accumulation.num_batches` | 8 | 4 |
| Training VRAM | ~10–12 GB | ~30–36 GB |

---

## Why Extra VRAM Matters

### 1. Larger batches without serial accumulation

With 12GB, gradient accumulation simulates a large effective batch by running multiple forward/backward passes sequentially — the GPU is idle between micro-batches' synchronisation points. On an A100, the full effective batch fits in one pass:

```
Local:  batch=8,  grad_accum=4 → 4 serial forward/backward passes
HPC:    batch=32, grad_accum=1 → 1 forward/backward pass (same gradient)
```

Same gradient signal, 3–4x faster iteration.

### 2. No gradient checkpointing overhead

Gradient checkpointing discards intermediate activations during the forward pass and recomputes them on demand during backprop. This trades ~35% VRAM for ~25% extra compute. On the A100 there is enough headroom to keep all activations in memory.

### 3. Longer audio (omniASR)

The fairseq2 recipe batches by total element count (`max_num_elements`). More VRAM allows doubling the maximum clip length from 30s to 60s, so fewer utterances in CoRal's conversation subset get truncated. This matters especially for longer conversational turns that carry natural Danish prosody and coarticulation.

### 4. Wall-clock speed

| Setup | omniASR 5K steps |
|---|---|
| RTX 3080 12GB (local) | ~8–12 hours |
| A100 40GB (DTU HPC) | ~2–4 hours |

Roughly 3–5× faster end-to-end due to larger batches, no checkpointing overhead, and raw A100 throughput.

---

## Hydra Config Composition (Wav2Vec2 / Whisper)

The root `configs/config.yaml` composes four groups:

```
model=<wav2vec2|whisper>  +  data=coral  +  train=<wav2vec2|whisper>  +  hardware=<local|hpc>
```

### Training configs

**`configs/train/wav2vec2.yaml`** — CTC training for Wav2Vec2:
- AdamW, lr=3e-5, linear warmup 500 steps
- 30 epochs, gradient clip 1.0
- Early stopping on `val_wer` (patience=5)

**`configs/train/whisper.yaml`** — Seq2seq training for Whisper:
- AdamW, lr=1e-4, linear warmup 500 steps
- 20 epochs, gradient clip 1.0, grad_accum=8 (local default)
- Early stopping on `val_wer` (patience=5)

Both configs save top-3 checkpoints by `val_wer`.

### Overriding at launch

Any config value can be overridden on the command line:

```bash
# Raise LR for HPC run
uv run python -m danish_asr.train model=wav2vec2 train=wav2vec2 hardware=hpc \
    train.optimizer.lr=5e-5

# Disable early stopping
uv run python -m danish_asr.train model=whisper train=whisper hardware=local \
    train.callbacks.early_stopping.patience=999
```

---

## fairseq2 Configs (omniASR)

These are standalone YAMLs consumed directly by the fairseq2 recipe, not by Hydra.

| File | Target | max_audio_len | max_num_elements | grad_accum |
|---|---|---|---|---|
| `configs/fairseq2/ctc-finetune-local.yaml` | RTX 3080 12GB | 480,000 (30s) | 3,840,000 | 8 |
| `configs/fairseq2/ctc-finetune-hpc.yaml` | A100 40/80GB | 960,000 (60s) | 7,680,000 | 4 |

Both use: `lr=1e-5`, `num_steps=5000`, validate/checkpoint every 500 steps, `bfloat16`.

---

## Usage Commands

### Wav2Vec2 LoRA

```bash
# Local
uv run python -m danish_asr.train model=wav2vec2 data=coral train=wav2vec2 hardware=local

# HPC
uv run python -m danish_asr.train model=wav2vec2 data=coral train=wav2vec2 hardware=hpc
```

### Whisper LoRA

```bash
# Local
uv run python -m danish_asr.train model=whisper data=coral train=whisper hardware=local

# HPC
uv run python -m danish_asr.train model=whisper data=coral train=whisper hardware=hpc
```

### omniASR (fairseq2)

```bash
# Local
invoke train.omniasr --hardware=local

# HPC
invoke train.omniasr --hardware=hpc

# Custom output directory
invoke train.omniasr --hardware=hpc --output-dir=/path/to/outputs

# Pass extra fairseq2 args
invoke train.omniasr --hardware=local --args="--no-sweep"
```

### Evaluate omniASR checkpoint

```bash
invoke train.omniasr-eval --checkpoint-dir=outputs/omniasr_local_20250308_120000 --hardware=local
```

### Verify config without training

```bash
# Print resolved config — no model loading, no data download
uv run python -m danish_asr.train --cfg job model=wav2vec2 train=wav2vec2 hardware=local
uv run python -m danish_asr.train --cfg job model=whisper train=whisper hardware=hpc
```

---

## File Map

```
configs/
  config.yaml                       # Root Hydra config (defaults: model, data, train, hardware)
  hardware/
    local.yaml                      # RTX 3080 12GB profile
    hpc.yaml                        # A100 40/80GB profile
  train/
    default.yaml                    # Generic ASR train config (used as fallback)
    wav2vec2.yaml                   # CTC-specific settings
    whisper.yaml                    # Seq2seq-specific settings
  fairseq2/
    ctc-finetune-local.yaml         # omniASR local (12GB)
    ctc-finetune-hpc.yaml           # omniASR HPC (A100)
  model/
    wav2vec2.yaml                   # Wav2Vec2-XLSR-53 + LoRA
    whisper.yaml                    # Whisper Large v3 + LoRA
  data/
    coral.yaml                      # CoRal-v3 dataset config

src/danish_asr/
  train.py                          # ASRLitModel + train_model() + Hydra entrypoint
  model.py                          # Wav2Vec2ASR, WhisperASR, build_model()
  data.py                           # CoRalDataset, CoRalDataModule, collate_fn
  metrics.py                        # compute_wer(), compute_cer()

tasks/
  train.py                          # invoke train.omniasr, invoke train.omniasr-eval
```
