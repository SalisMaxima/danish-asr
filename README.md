# Danish ASR

Fine-tuning Meta's Omnilingual ASR for Danish Speech Recognition

## Overview

This project fine-tunes [Meta's omniASR_CTC_300M](https://github.com/facebookresearch/omnilingual-asr) (325M parameters) for Danish speech recognition using the [CoRal-v2 dataset](https://huggingface.co/datasets/CoRal-project/coral-v2) (~480 hours of Danish speech).

**Course:** 5 ECTS Special Course at DTU (Feb-May 2026)

### Why This Project?

Danish ASR remains a challenge — existing models often struggle with regional dialects and non-standard speech. Research presented at the CoRal 2026 conference highlighted that ASR models perform significantly worse for elderly speakers with strong dialects compared to younger speakers using standard Danish ("rigsdansk"). This raises fairness concerns when deploying ASR in contexts like elder care.

The CoRal project addresses this by providing a diverse, high-quality dataset with rich demographic metadata, enabling both improved model training and systematic fairness evaluation across speaker groups.

### Why omniASR (CTC) Over Whisper?

| | omniASR CTC | Whisper |
|---|---|---|
| Architecture | Encoder-only (CTC head) | Encoder-Decoder |
| Hallucination | Cannot hallucinate (predicts per-frame) | Decoder can hallucinate (generates freely) |
| Model size | 325M parameters | 1.5B+ parameters (Large V3) |
| Inference | Fast, parallel output | Slower, autoregressive |
| Framework | fairseq2 | HuggingFace Transformers |

CTC models predict one character per audio frame and collapse repeated predictions, making them structurally incapable of hallucination — a known issue with Whisper, especially on domain-specific terminology. The CoRal team's own benchmarks showed that their large Whisper model achieved the best overall accuracy on Danish, but CTC-based models provide a more predictable and lightweight alternative.

### The CoRal Dataset

The [CoRal project](https://huggingface.co/CoRal-project) collected Danish speech data primarily through libraries across Denmark, chosen as cultural gathering points to maximize demographic diversity. Over 1,000 Danes donated approximately 2 hours each, contributing:

- **Read-aloud recordings** (3 x 2-hour sessions per speaker): 425 hours
- **Conversational recordings** (pairs of speakers from the same dialect region): 48 hours

The final dataset contains ~710 hours of material after quality filtering (~20% removed from the ~1,000-hour collection target). Dialect representation was captured through multiple signals: interview location, current address, self-reported dialect, school postal code, and childhood region.

**Known dataset characteristics:**
- Women are overrepresented (~65/35 split)
- Lolland-Falster is underrepresented
- Some coverage gaps on the west coast of Jutland
- Studio-quality audio (no "real-life" noisy conditions — no bad microphones, echo, etc.)
- Non-native Danish speakers were recruited via university students

**License:** Open for most uses, except synthetic speech generation and person identification.

**V3 is now released** (as of Feb 2026), with V3.1 expected.

## Quick Start

```bash
# Clone and setup
git clone <your-repo-url>
cd danish_asr
invoke core.setup-dev

# Activate environment
source .venv/bin/activate

# See all available commands
invoke --list
```

## Data

```bash
invoke data.download    # Download CoRal-v2 dataset (cached by HuggingFace)
invoke data.stats       # Show split sizes and audio duration stats
```

## Training

Training uses fairseq2's CTC finetuning recipe (full fine-tuning, not LoRA):

```bash
# Convert CoRal-v2 to Parquet format (required by fairseq2)
python scripts/convert_coral_to_parquet.py

# Run finetuning on single GPU
python -m workflows.recipes.wav2vec2.asr $OUTPUT_DIR \
    --config-file configs/omniasr/ctc-finetune-danish.yaml
```

See [docs/finetuning-recipe.md](docs/finetuning-recipe.md) for full configuration details and HPC job scripts.

## Evaluation

The primary evaluation focuses on both overall accuracy and demographic fairness:

- **WER** (Word Error Rate) — primary metric
- **CER** (Character Error Rate) — secondary metric
- **Per-group analysis:** WER/CER broken down by dialect, age group, and gender

Research by Anders Sogaard (KU) presented at the CoRal conference showed that enforcing equal performance across demographic groups can actually improve overall model performance — a finding that motivates our fairness-focused evaluation approach.

## Development

```bash
invoke quality.ruff    # Lint + format
invoke quality.test    # Run tests
invoke quality.ci      # Full CI pipeline
```

## Project Structure

```
danish_asr/
├── src/danish_asr/                     # Source code
│   ├── data.py                        # CoRalDataset + CoRalDataModule
│   ├── model.py                       # Wav2Vec2/Whisper baselines (comparison only)
│   ├── metrics.py                     # WER/CER via jiwer
│   ├── train.py                       # Legacy training template (not used)
│   └── analysis/                      # Model analysis CLI
├── configs/                           # Configuration files
│   ├── data/coral.yaml               # CoRal dataset config
│   └── omniasr/                      # fairseq2 finetuning configs
├── scripts/                           # Data conversion + HPC job scripts
├── tasks/                             # Invoke task modules
├── tests/                             # Unit tests (11 passing)
├── .github/workflows/                 # CI/CD
└── docs/                              # Documentation
```

## Stack

| Category | Tools |
|----------|-------|
| ASR Framework | fairseq2 + omnilingual-asr |
| Fine-tuning | Full fine-tuning via fairseq2 (no LoRA) |
| Audio | torchaudio, soundfile |
| Metrics | jiwer (WER/CER) |
| Experiment Tracking | Weights & Biases |
| Data Versioning | DVC |
| Package Manager | uv |
| Task Runner | invoke |
| Code Quality | ruff, pre-commit |
| CI/CD | GitHub Actions |
| HPC | DTU HPC (A100 GPUs via LSF) |

## Documentation

- [Project Roadmap](docs/project-roadmap.md) — phases, timelines, resource budget
- [Omnilingual ASR Overview](docs/omnilingual-asr-overview.md) — model architecture, installation
- [CoRal-v2 Dataset](docs/coral-v2-dataset.md) — splits, fields, demographics, collection methodology
- [Data Preparation](docs/data-preparation.md) — CoRal-v2 to Parquet conversion
- [Finetuning Recipe](docs/finetuning-recipe.md) — configs, hyperparameters
- [DTU HPC Setup](docs/dtu-hpc-setup.md) — GPU queues, LSF job scripts

## References

- [CoRal Project](https://huggingface.co/CoRal-project) — Danish speech dataset
- [omnilingual-asr](https://github.com/facebookresearch/omnilingual-asr) — Meta's ASR framework
- [fairseq2](https://github.com/facebookresearch/fairseq2) — Meta's sequence modeling toolkit
