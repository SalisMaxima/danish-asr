#!/bin/bash
#BSUB -J danish_asr_train_e6
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 10:00
#BSUB -B
#BSUB -N
#BSUB -u s204696@dtu.dk
#BSUB -o /work3/s204696/logs/lsf/train_e6_%J.out
#BSUB -e /work3/s204696/logs/lsf/train_e6_%J.err
#
# E6 experiment: fix shuffle windows (1→1000), lr=5e-5, 50k steps.
# Fresh run from scratch to test whether proper data shuffling improves
# generalisation (closes the ~21 loss unit train/val gap seen in E3/E5).
# See docs/experiment-plan.md §E6.
#
# Usage:
#   bsub < scripts/hpc/14_train_e6.sh
#
# Resume support: resubmitting this script will auto-resume from the latest
# checkpoint if training is interrupted.

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
mkdir -p /work3/$USER/wandb/cache

OMNI_ASR_DIR="/work3/$USER/omnilingual-asr"
if [ ! -d "$OMNI_ASR_DIR/workflows" ]; then
    echo "ERROR: omnilingual-asr repo not found at $OMNI_ASR_DIR" >&2
    echo "Clone it: git clone https://github.com/facebookresearch/omnilingual-asr.git $OMNI_ASR_DIR" >&2
    exit 1
fi
export PYTHONPATH="$OMNI_ASR_DIR:${PYTHONPATH:-}"

PROJECT_DIR="${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}"
cd "$PROJECT_DIR"
source .venv/bin/activate

RUN_DIR="${RESUME_DIR:-/work3/$USER/outputs/omniasr_e6}"
mkdir -p "$RUN_DIR"

python scripts/hpc/run_training.py \
    --config configs/fairseq2/ctc-finetune-hpc-e6.yaml \
    --output-dir "$RUN_DIR" \
    --wandb-tags "train,full,hpc,a100,e6,50k,lr5e-5,shuffle1000" \
    --wandb-resume allow
