#!/bin/bash
#BSUB -J danish_asr_train_e2
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 14:00
#BSUB -B
#BSUB -N
#BSUB -u s204696@dtu.dk
#BSUB -o /work3/s204696/logs/lsf/train_e2_%J.out
#BSUB -e /work3/s204696/logs/lsf/train_e2_%J.err
#
# E2 experiment: lr=3e-5, 30k steps, tri_stage scheduler.
# Fresh run — does NOT resume from the 1e-5 baseline (omniasr_20k).
# See docs/experiment-plan.md §E2.
#
# Usage:
#   bsub < scripts/hpc/04_train_e2.sh
#
# Resume support: resubmitting this script with the same RUN_DIR will
# auto-resume from the latest checkpoint if training is interrupted.

set -euo pipefail

# --- Environment ---
source "${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}/scripts/hpc/env.sh"
setup_omniasr

RUN_DIR="${RESUME_DIR:-/work3/$USER/outputs/omniasr_e2}"
mkdir -p "$RUN_DIR"

python scripts/hpc/run_training.py \
    --config configs/fairseq2/ctc-finetune-hpc-e2.yaml \
    --output-dir "$RUN_DIR" \
    --wandb-tags "train,full,hpc,a100,e2,30k,lr3e-5" \
    --wandb-resume allow
