# Danish ASR

Fine-tuning and evaluation of modern ASR models for Danish, using the CoRal dataset.

**Course:** 5 ECTS Special Course (Feb-May 2026) at DTU
**Goal:** Fine-tune Meta's omnilingual ASR model and Whisper for Danish using CoRal (480h Danish speech)

## IMPORTANT
- **ALWAYS activate environment first: `source .venv/bin/activate`**
- **After code changes run: `invoke quality.ruff`**
- **After changing CLAUDE.md run: `invoke core.sync-ai-config` to sync copilot-instructions.md (do not edit copilot-instructions.md directly; CLAUDE.md is the source of truth)**
- **Always use `uv run` for Python commands** (e.g., `uv run python`, `uv run pytest`)
- **Always use `uv add` to install packages** (never `pip install`)

## Stack
- Python 3.12, PyTorch Lightning, Hydra configs
- uv for package management, invoke for tasks
- W&B for experiment tracking, DVC for data versioning
- HuggingFace Transformers + PEFT/LoRA for parameter-efficient fine-tuning
- torchaudio for audio processing, jiwer for WER/CER metrics

## Models
- **Wav2Vec2 XLSR-53** (`wav2vec2_asr`) - Facebook's cross-lingual speech representations, CTC-based
- **Whisper Large V3** (`whisper_asr`) - OpenAI's multilingual ASR, encoder-decoder
- Both support **LoRA fine-tuning** via PEFT (r=8, alpha=16, targeting q/v/k/out projections)

## Dataset: CoRal
- **Source:** `alexandrainst/coral` on HuggingFace
- **Subsets:** `read_aloud` (primary), `conversational`
- **Audio:** 16kHz, variable length (max 30s default)
- **Language:** Danish

## Task Namespaces
Tasks are organized into namespaces. Use `invoke <namespace>.<task>` or `invoke --list` to see all.

**Key namespaces:**
- `core` - Environment setup (bootstrap, sync, setup-dev)
- `data` - Data management (download, preprocess, stats, validate)
- `train` - Training & sweeps (train, sweep)
- `eval` - Evaluation (analyze, benchmark, profile)
- `quality` - Code quality (ruff, test, ci, security-check)
- `deploy` - Deployment (api, frontend, promote-model)
- `docker` - Docker ops (build, train, clean)
- `git` - Git ops (status, commit, branch)
- `dvc` - DVC ops (pull, push, add)
- `utils` - Utilities (clean-all, env-info, check-gpu)
- `monitor` - Drift detection (extract-stats, check-drift)
- `docs` - Documentation (build, serve)

## Essential Commands
```bash
# Setup (run once)
invoke core.setup-dev                # Complete dev environment setup

# Development workflow
invoke quality.ruff                  # Lint + format (run after code changes)
invoke quality.test                  # Run tests
invoke quality.ci                    # Run full CI pipeline locally

# Data & Training
invoke data.download                 # Download CoRal dataset
invoke data.stats                    # Show dataset statistics (splits, duration)
invoke data.validate                 # Validate audio integrity
invoke train.train                   # Train model (default: wav2vec2 + LoRA)
invoke train.train --args "model=whisper"  # Train Whisper model

# Sweeps
invoke train.sweep                   # Create W&B sweep
invoke train.sweep-agent --sweep-id <ID>  # Run sweep agent

# Deployment
invoke deploy.api                    # Run API server

# Utilities
invoke utils.clean-all               # Clean all artifacts
invoke utils.check-gpu               # Check GPU availability
invoke utils.env-info                # Show environment info
```

## Key Paths
- `src/danish_asr/` - Main source code
  - `model.py` - Wav2Vec2ASR + WhisperASR with LoRA
  - `data.py` - CoRalDataset + CoRalDataModule
  - `metrics.py` - WER/CER computation via jiwer
  - `train.py` - Lightning training pipeline
  - `api.py` - FastAPI server
- `configs/` - Hydra configs
  - `model/wav2vec2.yaml` - Wav2Vec2 XLSR-53 config
  - `model/whisper.yaml` - Whisper Large V3 config
  - `data/coral.yaml` - CoRal dataset config
  - `train/default.yaml` - ASR training config (CTC loss, AdamW, bf16)
- `tests/` - Unit tests
- `tasks.py` - All invoke commands

## Key Metrics
- **WER** (Word Error Rate) - Primary metric, monitored for early stopping/checkpointing
- **CER** (Character Error Rate) - Secondary metric
