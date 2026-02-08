# Omnilingual ASR: Model Overview

Reference for the [facebookresearch/omnilingual-asr](https://github.com/facebookresearch/omnilingual-asr) system used in this project.

## Project Summary

Meta's Omnilingual ASR covers **1,600+ languages** using self-supervised wav2vec2 encoders with CTC or LLM decoders. Apache 2.0 licensed.

## Model We Use

**`omniASR_CTC_300M`** — the smallest CTC model, ideal for our 12GB local GPU and DTU HPC.

| Property | Value |
|---|---|
| Parameters | 325M |
| Download size | 1.3 GiB |
| Inference VRAM | ~2 GiB |
| Training VRAM | ~10-12 GiB (single GPU, grad accum) |
| Architecture | wav2vec2 encoder + CTC head |
| Max audio length | 40s (CTC models) |
| Tokenizer | `omniASR_tokenizer_v1` |

## All Available Models

### CTC Models (fast inference, no LM needed)

| Model | Params | Size | VRAM |
|---|---|---|---|
| omniASR_CTC_300M | 325M | 1.3 GiB | ~2 GiB |
| omniASR_CTC_300M_v2 | 325M | 1.3 GiB | ~2 GiB |
| omniASR_CTC_1B | ~1B | — | — |
| omniASR_CTC_3B | ~3B | — | — |
| omniASR_CTC_7B_v2 | 6.5B | 25.0 GiB | ~15 GiB |

### LLM Models (higher accuracy, language-conditioned)

| Model | Params | Size | VRAM |
|---|---|---|---|
| omniASR_LLM_300M_v2 | 1.6B | 6.1 GiB | ~5 GiB |
| omniASR_LLM_7B_v2 | 7.8B | 30.0 GiB | ~17 GiB |
| omniASR_LLM_Unlimited_7B_v2 | 7.8B | 30.0 GiB | ~17 GiB |

### SSL Encoders (feature extraction only, used as pretrained backbones)

- `omniASR_W2V_300M`, `omniASR_W2V_1B`, `omniASR_W2V_3B`, `omniASR_W2V_7B`

## Architecture

```
Audio (16kHz) → wav2vec2 encoder → CTC projection → character/token sequence
```

- Encoder: self-supervised wav2vec2 (pretrained on massive multilingual data)
- Decoder: CTC (greedy or beam search)
- Tokenizer: character-level with language-specific tokens
- Language code format: `{iso639-3}_{script}` (e.g., `dan_Latn` for Danish)

## Installation

```bash
# Core package
uv add omnilingual-asr

# With data preparation tools
uv add "omnilingual-asr[data]"
```

System dependency: `libsndfile` (usually pre-installed on Linux).

Built on **fairseq2** — Meta's sequence modeling toolkit. Models auto-download to `~/.cache/fairseq2/assets/` on first use.

## Quick Inference Test

```python
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

pipeline = ASRInferencePipeline(model_card="omniASR_CTC_300M")
transcriptions = pipeline.transcribe(
    ["/path/to/danish_audio.wav"],
    lang=["dan_Latn"],
    batch_size=1,
)
print(transcriptions)
```

## Finetuning Strategy Options

Two approaches for CTC finetuning:

| Strategy | Config | Starting point | Use case |
|---|---|---|---|
| **Finetune CTC checkpoint** | `ctc-finetune.yaml` | `omniASR_CTC_300M` | Continue from released model (recommended) |
| **Train CTC from encoder** | `ctc-from-encoder.yaml` | `omniASR_W2V_300M` | New CTC head on pretrained encoder |

For our project: **ctc-finetune** — we start from the released `omniASR_CTC_300M` checkpoint, which already has a trained CTC head.

## Key Limitations

- CTC models: max 40s audio (960,000 samples at 16kHz)
- No LoRA/PEFT support — fairseq2 does full fine-tuning
- `Unlimited` variants do not support finetuning recipes
- Requires Parquet-format data (see [data-preparation.md](data-preparation.md))

## References

- Repository: <https://github.com/facebookresearch/omnilingual-asr>
- Data prep: <https://github.com/facebookresearch/omnilingual-asr/blob/main/workflows/dataprep/README.md>
- Finetuning: <https://github.com/facebookresearch/omnilingual-asr/blob/main/workflows/recipes/wav2vec2/asr/README.md>
