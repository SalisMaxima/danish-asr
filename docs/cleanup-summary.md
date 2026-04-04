# Repository Cleanup and Simplification

**Date:** 2026-04-04
**Branch:** `cleanup/remove-dead-code-and-deduplicate-hpc`

## Context

This repository was initialized from a classification-focused project template. Over time, the actual work focused on three things:

1. **fairseq2/omniASR fine-tuning** on DTU HPC (the primary training path)
2. **HuggingFace baseline training** (Wav2Vec2 + Whisper comparisons)
3. **Data preprocessing** (CoRal-v3 → Parquet conversion)

A large amount of template scaffolding — API endpoints, Docker infrastructure, monitoring, evaluation stubs, classification loss functions, deployment tasks — was never implemented and remained as dead code. Additionally, the 16+ HPC shell scripts contained significant duplicated boilerplate (environment variables, directory creation, project activation).

This cleanup removes ~27 dead files (~900+ lines), drops 16 unused dependencies, extracts shared HPC environment setup into a single sourced file, and fixes inconsistent script permissions.

---

## Changes

### 1. Dead Source Modules Removed

| File | Lines | Reason |
|------|-------|--------|
| `src/danish_asr/evaluate.py` | 11 | Stub that logged "not yet implemented" and returned `{}`. Zero imports. |
| `src/danish_asr/losses.py` | 99 | `FocalLoss`, `LabelSmoothingLoss`, `build_loss()` — classification losses irrelevant to ASR. Zero imports. |
| `src/danish_asr/api.py` | 142 | FastAPI scaffold with model loading commented out (`TODO`), predict endpoint returning dummy `"class": "unknown"`. |
| `src/danish_asr/feedback_store.py` | 104 | SQLite feedback store only used by `api.py`. Schema referenced `image_path` — a classification template artifact. |
| `src/danish_asr/analysis/__init__.py` | 0 | Empty. |
| `src/danish_asr/analysis/cli.py` | 35 | Typer CLI with two commands that both logged "Implement your logic here". Zero imports. |
| `src/danish_asr/monitoring/__init__.py` | 0 | Empty. |
| `src/danish_asr/monitoring/drift_check.py` | 22 | Stub with `TODO: Implement drift detection`. Zero imports. |
| `src/danish_asr/monitoring/extract_stats.py` | ~20 | Stub with `TODO: Implement stats extraction`. Zero imports. |

### 2. Dead Task Modules Removed

| File | Lines | Reason |
|------|-------|--------|
| `tasks/deploy.py` | 67 | All 4 tasks called non-existent modules (`danish_asr.promote_model`, `danish_asr.onnx_export`, dead `api.py`, non-existent `frontend/pages/home.py`). |
| `tasks/monitor.py` | 39 | Thin wrapper around the deleted monitoring stubs. |
| `tasks/eval.py` | 78 | `analyze` wrapped the stub analysis CLI. `benchmark`, `profile`, `model_info` all called non-existent modules (`danish_asr.inference_benchmark`, `danish_asr.model_info`). |
| `tasks/docker.py` | 97 | Docker image building for the non-functional API and Docker-based training. Training actually runs on HPC via LSF — Docker was never used. |
| `tasks/git_tasks.py` | 42 | Thin wrappers around git commands. The `commit` task ran `git add .` (dangerous). The `branch` task had a fragile cross-module import (`from tasks.quality import ruff`). |

### 3. Dead HPC Scripts Removed

| File | Reason |
|------|--------|
| `scripts/hpc/upload_step15k.py` | One-off upload script hardcoded to a specific historical W&B run (`gi46t3kp`). No reuse value. |
| `scripts/hpc/repair_parquet_schema.py` | One-time repair of legacy Parquet converter output. The data repair was completed long ago. |

### 4. Docker Infrastructure Removed

| File | Reason |
|------|--------|
| `dockerfiles/api.dockerfile` | Served the dead FastAPI app. |
| `dockerfiles/api.cloudrun.dockerfile` | Cloud Run variant of the dead API. |
| `dockerfiles/train.dockerfile` | CPU training image — training runs on HPC, not Docker. |
| `dockerfiles/train_cuda.dockerfile` | CUDA training image — same reason. |
| `.dockerignore` | No longer needed without dockerfiles. |

### 5. Dead Test Removed

| File | Reason |
|------|--------|
| `tests/test_api.py` | Tested the deleted `api.py` module. |

### 6. Junk Files Removed

Three tracked zero-byte files at the repo root (likely accidental shell pastes):
- `+`
- `+**Mandatory`
- `+**Every`

### 7. Classification Config Remnant Removed

| File | Reason |
|------|--------|
| `configs/model/default.yaml` | Contained `num_classes: 2` — a leftover from the classification template. The actual ASR model configs (`wav2vec2.yaml`, `whisper.yaml`) are separate. |

### 8. Dependency Cleanup

**16 unused packages removed from `pyproject.toml`:**

From `[project.dependencies]` (14 packages):

| Package | Reason |
|---------|--------|
| `fastapi` | Only used by deleted `api.py` |
| `uvicorn` | Only used by deleted `api.py` |
| `python-multipart` | Only used by deleted `api.py` |
| `httpx` | Only used by deleted `api.py` |
| `prometheus-fastapi-instrumentator` | Only used by deleted `api.py` |
| `prometheus-client` | Only used by deleted `api.py` |
| `psutil` | Only used by deleted `api.py` |
| `evidently` | Only used by deleted monitoring stubs |
| `scikit-learn` | Zero imports anywhere in codebase |
| `matplotlib` | Zero imports anywhere in codebase |
| `pandas` | Zero imports anywhere in codebase |
| `librosa` | Zero imports anywhere (`torchaudio` is used instead) |
| `scipy` | Zero imports anywhere in codebase |
| `bitsandbytes` | Zero imports anywhere in codebase |

From `[dependency-groups] dev` (3 packages):

| Package | Reason |
|---------|--------|
| `locust` | Load testing tool, never referenced |
| `tensorboard` | W&B is used for experiment tracking |
| `zensical` | Docs build tool, not actively used |

**Dead entry points removed from `[project.scripts]`:**
- `evaluate = "danish_asr.evaluate:evaluate_model"` (pointed to deleted stub)
- `api = "danish_asr.api:app"` (pointed to deleted API)

### 9. `tasks.py` Updated

Removed loading and registration of the 5 deleted task collections:
- `eval` (was `eval_mod`)
- `deploy`
- `docker`
- `monitor`
- `git` (was `git_tasks`)

**Remaining 7 namespaces:** `core`, `data`, `train`, `quality`, `dvc`, `docs`, `utils`

### 10. HPC Shell Script Deduplication

**Created `scripts/hpc/env.sh`** — a shared environment setup file sourced by all 16 job scripts.

Contains:
- All cache/scratch directory exports (`HF_HOME`, `HF_DATASETS_CACHE`, `FAIRSEQ2_CACHE_DIR`, `TMPDIR`, `WANDB_DIR`, `WANDB_DATA_DIR`, `WANDB_CACHE_DIR`)
- Directory creation (`mkdir -p` for tmp, logs/lsf, logs/python, wandb/cache)
- Project directory resolution and venv activation
- `setup_omniasr()` helper function for omniASR PYTHONPATH validation

**Updated all 16 job scripts** to replace their 10-20 line duplicated environment blocks with:
```bash
source "$(dirname "$0")/env.sh"
setup_omniasr  # only in scripts that need omnilingual-asr
```

**Scripts using `source env.sh` only** (no omniASR needed):
- `01_verify_data.sh`, `02_convert_fairseq2.sh`
- `07_train_wav2vec2.sh`, `08_train_whisper.sh`
- `09_smoke_wav2vec2.sh`, `10_smoke_whisper.sh`

**Scripts using `source env.sh` + `setup_omniasr`:**
- `03_train.sh`, `04_eval.sh`, `05_smoke_test.sh`
- `04_train_e2.sh`, `05_train_e3.sh`, `10_train_e5.sh`
- `11_eval_e2.sh`, `12_eval_e3.sh`
- `13_train_e7.sh`, `14_train_e6.sh`

**Special case:** `06_sweep.sh` additionally prepends `$PROJECT_DIR` to `PYTHONPATH` (needed for `scripts.hpc.common` imports).

**`submit_pipeline.sh`** was not modified — it's an orchestrator that doesn't run Python directly.

### 11. Fixed Script Permissions

Added execute bit (`chmod +x`) to 8 scripts that were missing it:
`04_train_e2.sh`, `05_train_e3.sh`, `10_train_e5.sh`, `11_eval_e2.sh`, `12_eval_e3.sh`, `13_train_e7.sh`, `14_train_e6.sh`, `env.sh`

---

## What Was NOT Changed

- **All active source modules** (`data.py`, `model.py`, `train.py`, `preprocessing.py`, `metrics.py`, `text.py`, `sweep_train.py`, `sweep_best.py`, `utils.py`, `__init__.py`)
- **All active HPC Python scripts** (`run_training.py`, `run_eval.py`, `common.py`, `train_common.py`, `train_wav2vec2.py`, `train_whisper.py`, `verify_data.py`, `convert_to_fairseq2.py`, `make_subset_tsv.py`, `patch_wer_calculator.py`, `sweep_agent_wrapper.py`)
- **All fairseq2 experiment configs** (left as-is — they serve as documentation of completed experiments)
- **DVC integration** (`dvc`/`dvc-gs` deps + `tasks/dvc_tasks.py` kept per user request)
- **Documentation** (`docs/`, `mkdocs.yaml`, `tasks/docs.py`)
- **All tests** (except `test_api.py` which tested deleted code)

---

## Verification

- All **92 tests pass** (`uv run pytest`)
- **Lint clean** (`invoke quality.ruff`)
- **`invoke --list`** shows 7 healthy namespaces (core, data, train, quality, dvc, docs, utils)
- **`uv.lock`** regenerated (1845 lines lighter)
