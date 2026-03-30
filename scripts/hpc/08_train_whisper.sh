#!/bin/bash
#BSUB -J whisper_full
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=20GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -B
#BSUB -N
#BSUB -u s204696@dtu.dk
#BSUB -o /work3/s204696/logs/lsf/whisper_full_%J.out
#BSUB -e /work3/s204696/logs/lsf/whisper_full_%J.err
#
# Whisper-large-v3 full seq2seq training on CoRal-v3 (~30k steps, ~17h).
# Should complete in a single 24h job.
#
# Usage: bsub < scripts/hpc/08_train_whisper.sh

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

# Fixed run directory — reused across sequential jobs so checkpoint resume works.
# Override with WHISPER_RUN_DIR to use a different path.
RUN_DIR="${WHISPER_RUN_DIR:-/work3/$USER/outputs/whisper_full}"
mkdir -p "$RUN_DIR"

python -m scripts.hpc.train_whisper \
    --config configs/hf_baseline/whisper_full.yaml \
    --output-dir "$RUN_DIR" \
    --wandb-tags "train,full,hpc,a100,whisper" \
    --wandb-resume allow \
    ${RESUME_CKPT:+--resume-from-checkpoint "$RESUME_CKPT"}
