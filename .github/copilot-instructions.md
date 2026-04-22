# Danish ASR - Copilot Instructions — Claude Code Instructions

Fine-tuning `omniASR_CTC_300M_v2` (325M params, CTC) for Danish using CoRal-v3 (~710h). 5 ECTS special course at DTU, Feb–May 2026.

## MANDATORY — Execute Before Every Task

1. **Activate environment:** `source .venv/bin/activate`
2. **Use `uv run` for all Python execution:** `uv run python`, `uv run pytest`, etc.
3. **Use `uv add` to install packages.** NEVER use `pip install`.
4. **After any code change:** `invoke quality.ruff`
5. **After editing this file:** `invoke core.sync-ai-config` (NEVER edit `copilot-instructions.md` directly)

## Forbidden Actions

- Do NOT use HuggingFace Transformers for training — this project uses **fairseq2 + omnilingual-asr**
- Do NOT use `pip install` — always `uv add`
- Do NOT edit `copilot-instructions.md` — `CLAUDE.md` is the source of truth
- Do NOT import from or modify `src/danish_asr/train.py` — it is a broken classification template, not used
- Do NOT use `alexandrainst/coral` — the correct dataset is `CoRal-project/coral-v3`
- Do NOT resample audio to anything other than 16kHz

## Stop Conditions — Pause and Ask

- Architecture decisions involving fairseq2 configs or training hyperparameters
- Adding new dependencies not already in `pyproject.toml`
- Modifying DVC pipelines or W&B project config
- Changes affecting the HPC job submission scripts
- Two failed attempts to resolve the same error

## Stack

- Python 3.12, uv (packages), invoke (tasks)
- **fairseq2** + `omnilingual-asr` for training (NOT HF Transformers)
- torchaudio (audio), jiwer (WER/CER)
- W&B (tracking), DVC (data versioning)

## Model

- **`omniASR_CTC_300M_v2`** — CTC finetuning via fairseq2 recipe, full fine-tuning (no LoRA)
- Language code: `dan_Latn`
- `src/danish_asr/model.py` contains Wav2Vec2/Whisper baselines for comparison only

## Dataset

- **Source:** `CoRal-project/coral-v3` on HuggingFace
- **Subsets:** `read_aloud` + `conversation` (both used)
- **Splits:** `train` / `validation` / `test` — map `validation` → `dev` for fairseq2
- **Audio:** 48kHz native → resample to 16kHz
- **Metrics:** WER (primary), CER (secondary)

## Key Paths

| Path | Purpose |
|------|---------|
| `src/danish_asr/data.py` | CoRalDataset + CoRalDataModule (loads coral-v3) |
| `src/danish_asr/model.py` | Wav2Vec2/Whisper baselines only (LoRA comparison) |
| `src/danish_asr/metrics.py` | WER/CER via jiwer (reusable) |
| `src/danish_asr/train.py` | BROKEN — do not use |
| `configs/data/coral.yaml` | Data config |
| `tests/test_data.py` | Data pipeline tests |
| `tasks.py` | All invoke commands |

## Commands

```bash
invoke --list               # See all available tasks
invoke core.setup-dev       # One-time dev setup
invoke quality.ruff         # Lint + format (run after every code change)
invoke quality.test         # Run tests
invoke quality.ci           # Full CI pipeline
invoke data.download        # Download CoRal
invoke data.stats           # Dataset statistics
invoke utils.check-gpu      # GPU availability
```

## HPC Quick Reference

Target cluster: DTU HPC (LSF scheduler). Full reference: `docs/dtu_hpc/README.md`.

### HPC Job Script Standard

All `scripts/hpc/*.sh` BSUB scripts MUST include these directives:

```bash
#BSUB -B                                   # Email at job start
#BSUB -N                                   # Email at job end
#BSUB -u s204696@dtu.dk                    # Notification email
```

Place these after `-W` (walltime) and before `-o`/`-e` (log paths). When creating new HPC scripts, follow this template:

```bash
#!/bin/bash
#BSUB -J danish_asr_<name>
#BSUB -q gpua100                           # or hpc for CPU-only
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"  # omit for CPU-only
#BSUB -W 24:00
#BSUB -B
#BSUB -N
#BSUB -u s204696@dtu.dk
#BSUB -o /work3/s204696/logs/lsf/<name>_%J.out
#BSUB -e /work3/s204696/logs/lsf/<name>_%J.err

set -euo pipefail

# --- Environment ---
export HF_HOME=/work3/$USER/hf_cache
export HF_DATASETS_CACHE=/work3/$USER/hf_cache/datasets
export FAIRSEQ2_CACHE_DIR=/work3/$USER/fairseq2_cache
export TMPDIR=/work3/$USER/tmp
export WANDB_DIR=/work3/$USER/wandb
export WANDB_DATA_DIR=/work3/$USER/wandb
export WANDB_CACHE_DIR=/work3/$USER/wandb/cache
mkdir -p "$TMPDIR"
mkdir -p /work3/$USER/logs/lsf
mkdir -p /work3/$USER/logs/python
mkdir -p /work3/$USER/wandb/cache

PROJECT_DIR="${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}"
cd "$PROJECT_DIR"
source .venv/bin/activate

# ... job logic ...
```

### Essential LSF commands

| Command | Purpose |
|---------|---------|
| `bsub < job.sh` | Submit job |
| `bstat` | List your jobs |
| `bpeek -f <jobid>` | Stream live output |
| `bkill <jobid>` | Kill job |
| `bjobs -l <jobid>` | Detailed info + PENDING REASONS |
| `bhist -l <jobid>` | History + termination reason |
| `nodestat -G gpua100` | GPU availability |

### GPU queues

| Queue | GPU | VRAM | Max walltime |
|-------|-----|------|-------------|
| `gpua100` | A100 | 40/80 GB | 24h |
| `gpul40s` | L40s | 48 GB | 24h |
| `gpuv100` | V100 | 16/32 GB | 24h |

### Storage rules

- `$HOME` = 30 GB, backed up. Code and configs only.
- `/work3/$USER/` = large, NOT backed up. Datasets, checkpoints, caches, conda.
- NEVER install conda or store model weights in `$HOME`.
- Set `HF_HOME`, `FAIRSEQ2_CACHE_DIR`, `TMPDIR` to `/work3/` paths in every job script.

### Common failure → fix

| Symptom | Fix |
|---------|-----|
| `TERM_MEMLIMIT` | Increase `rusage[mem=...]` |
| CUDA OOM | Reduce batch size or request larger GPU |
| `TERM_RUNLIMIT` | Increase `-W` walltime (default is 15 min!) |
| `TERM_CWD_NOTEXIST` | `mkdir -p logs` before submitting |
| Job stuck in PEND | `bjobs -l <jobid>` for reasons; try different queue |
| **Exit code 120, no traceback** | **`/work3` NVME pool quota hit (350 GB hard limit). Run `getquota_work3.sh` — check storagepool 6. Clean `wandb/cache/` and old checkpoint dirs. Every training config MUST have `keep_last_n_checkpoints: 2` + `keep_best_n_checkpoints: 1`.** |

### Interactive GPU session

```bash
a100sh                      # shared A100 (quick dev)
# OR exclusive via scheduler:
bsub -Is -q gpua100 -n 4 -R "rusage[mem=16GB]" -R "span[hosts=1]" \
    -gpu "num=1:mode=exclusive_process" -W 1:00 bash
```

## Documentation

Detailed guides in `docs/`:

- `docs/project-roadmap.md` — phases, timelines, resource budget
- `docs/omnilingual-asr-overview.md` — model details, installation
- `docs/data-preparation.md` — CoRal-v3 → Parquet conversion
- `docs/finetuning-recipe.md` — configs, hyperparameters
- `docs/dtu_hpc/` — **complete HPC reference** (access, storage, LSF, environment, monitoring)
- `docs/coral-dataset.md` — splits, fields, demographics
