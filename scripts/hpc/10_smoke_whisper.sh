#!/bin/bash
#BSUB -J whisper_smoke
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=20GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 0:45
#BSUB -B
#BSUB -N
#BSUB -u s204696@dtu.dk
#BSUB -o /work3/s204696/logs/lsf/whisper_smoke_%J.out
#BSUB -e /work3/s204696/logs/lsf/whisper_smoke_%J.err
#
# Whisper smoke test — validates pipeline in <15 min.
#
# Usage: bsub < scripts/hpc/10_smoke_whisper.sh

set -euo pipefail

# --- Environment ---
export HF_HOME=/work3/$USER/hf_cache
export HF_DATASETS_CACHE=/work3/$USER/hf_cache/datasets
export TMPDIR=/work3/$USER/tmp
export WANDB_DIR=/work3/$USER/wandb
export WANDB_DATA_DIR=/work3/$USER/wandb
export WANDB_CACHE_DIR=/work3/$USER/wandb/cache
mkdir -p "$TMPDIR"
mkdir -p /work3/$USER/logs/lsf
mkdir -p /work3/$USER/wandb/cache

PROJECT_DIR="${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}"
cd "$PROJECT_DIR"
source .venv/bin/activate

python -m scripts.hpc.train_whisper \
    --config configs/hf_baseline/whisper_smoke.yaml \
    --wandb-resume never \
    --wandb-tags "smoke,hpc,a100,whisper"
