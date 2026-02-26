# Danish ASR

Fine-tuning Meta's omnilingual ASR (`omniASR_CTC_300M`) for Danish using CoRal-v3 (~710h Danish speech).

**Course:** 5 ECTS Special Course (Feb-May 2026) at DTU

## IMPORTANT
- **ALWAYS activate environment first: `source .venv/bin/activate`**
- **After code changes run: `invoke quality.ruff`**
- **After changing CLAUDE.md run: `invoke core.sync-ai-config` to sync copilot-instructions.md (do not edit copilot-instructions.md directly; CLAUDE.md is the source of truth)**
- **Always use `uv run` for Python commands** (e.g., `uv run python`, `uv run pytest`)
- **Always use `uv add` to install packages** (never `pip install`)

## Stack
- Python 3.12, uv (packages), invoke (tasks)
- **fairseq2** + `omnilingual-asr` for training (NOT HF Transformers)
- torchaudio for audio, jiwer for WER/CER
- W&B for tracking, DVC for data versioning

## Model
- **`omniASR_CTC_300M`** (325M params) — CTC finetuning via fairseq2 recipe
- No LoRA — fairseq2 does full fine-tuning
- Language code: `dan_Latn`
- Existing Wav2Vec2/Whisper code in `model.py` is baseline comparison only

## Dataset
- **Source:** `CoRal-project/coral-v3` on HuggingFace (NOT `alexandrainst/coral`)
- **Subsets:** `read_aloud` + `conversation` — both used
- **Splits:** `train` / `validation` / `test` (HF naming; map `validation` → `dev` for fairseq2)
- **Audio:** 48kHz native, resample to 16kHz
- See [docs/coral-dataset.md](docs/coral-dataset.md) for fields and demographics

## Key Paths
- `src/danish_asr/data.py` — CoRalDataset + CoRalDataModule (loads coral-v3)
- `src/danish_asr/model.py` — Wav2Vec2/Whisper baselines (LoRA)
- `src/danish_asr/metrics.py` — WER/CER via jiwer (reusable)
- `src/danish_asr/train.py` — BROKEN classification template, not used for omnilingual ASR
- `configs/data/coral.yaml` — Data config
- `tests/test_data.py` — Data pipeline tests
- `tasks.py` — All invoke commands (`invoke --list` to see all)

## Essential Commands
```bash
invoke core.setup-dev       # One-time dev setup
invoke quality.ruff         # Lint + format
invoke quality.test         # Run tests
invoke quality.ci           # Full CI pipeline
invoke data.download        # Download CoRal
invoke data.stats           # Dataset statistics
invoke utils.check-gpu      # GPU availability
```

## Documentation
Detailed guides in `docs/`:
- [Project Roadmap](docs/project-roadmap.md) — phases, timelines, resource budget
- [Omnilingual ASR Overview](docs/omnilingual-asr-overview.md) — model details, installation
- [Data Preparation](docs/data-preparation.md) — CoRal-v3 → Parquet conversion
- [Finetuning Recipe](docs/finetuning-recipe.md) — configs, hyperparameters
- [DTU HPC Setup](docs/dtu-hpc-setup.md) — GPU queues, LSF job scripts
- [CoRal Dataset](docs/coral-dataset.md) — splits, fields, demographics

## Key Metrics
- **WER** (Word Error Rate) — primary
- **CER** (Character Error Rate) — secondary
