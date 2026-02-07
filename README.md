# Danish ASR

Fine-tuning and Evaluation of Modern ASR Models for Danish

## Overview

This project fine-tunes state-of-the-art ASR models for Danish speech recognition using the [CoRal dataset](https://huggingface.co/datasets/alexandrainst/coral) (480+ hours of Danish speech). It uses parameter-efficient fine-tuning (LoRA) to adapt large pretrained models with minimal compute.

**Models:**
- **Wav2Vec2 XLSR-53** - CTC-based ASR with LoRA fine-tuning
- **Whisper Large V3** - Encoder-decoder ASR with LoRA fine-tuning

**Dataset:** CoRal (read-aloud + conversational Danish speech)

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
invoke data.download                  # Download CoRal dataset (cached by HuggingFace)
invoke data.stats                     # Show split sizes and audio duration stats
invoke data.validate                  # Check audio integrity
```

## Training

```bash
# Train Wav2Vec2 with LoRA (default)
invoke train.train

# Train Whisper with LoRA
invoke train.train --args "model=whisper"

# Hyperparameter sweep
invoke train.sweep                    # Create W&B sweep
invoke train.sweep-agent --sweep-id <SWEEP_ID>
```

## Development

```bash
invoke quality.ruff    # Lint + format
invoke quality.test    # Run tests
invoke quality.ci      # Full CI pipeline
```

## Deployment

```bash
invoke deploy.api       # Run API server
invoke docker.build     # Build Docker image
```

## Project Structure

```
danish_asr/
├── src/danish_asr/                     # Source code
│   ├── model.py                       # Wav2Vec2ASR + WhisperASR with LoRA
│   ├── data.py                        # CoRalDataset + CoRalDataModule
│   ├── metrics.py                     # WER/CER via jiwer
│   ├── train.py                       # Lightning training pipeline
│   ├── losses.py                      # Loss functions
│   ├── api.py                         # FastAPI server
│   ├── sweep_train.py                 # W&B sweep bridge
│   ├── sweep_best.py                  # Best sweep run finder
│   ├── monitoring/                    # Drift detection
│   └── analysis/                      # Model analysis CLI
├── configs/                           # Hydra configs
│   ├── config.yaml                    # Main config
│   ├── model/wav2vec2.yaml           # Wav2Vec2 XLSR-53 + LoRA
│   ├── model/whisper.yaml            # Whisper Large V3 + LoRA
│   ├── data/coral.yaml               # CoRal dataset config
│   ├── train/default.yaml            # ASR training config
│   └── sweeps/train_sweep.yaml       # Sweep configuration
├── tasks/                             # Invoke task modules (12 namespaces)
├── tests/                             # Unit tests
├── dockerfiles/                       # Docker images (train, API, CUDA)
├── .github/workflows/                 # CI/CD (7 workflows)
└── docs/                              # MkDocs documentation
```

## Stack

| Category | Tools |
|----------|-------|
| ML Framework | PyTorch Lightning, HuggingFace Transformers |
| Fine-tuning | PEFT/LoRA, bitsandbytes |
| Audio | torchaudio, librosa, soundfile |
| Metrics | jiwer (WER/CER) |
| Config | Hydra, OmegaConf |
| Experiment Tracking | Weights & Biases |
| Data Versioning | DVC + GCS |
| Package Manager | uv |
| Task Runner | invoke |
| Code Quality | ruff, mypy, pre-commit, bandit |
| API | FastAPI, uvicorn |
| Containerization | Docker (CPU + CUDA) |
| CI/CD | GitHub Actions |
