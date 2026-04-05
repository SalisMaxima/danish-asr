#!/bin/bash
#BSUB -J danish_asr_train_e7
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 10:00
#BSUB -B
#BSUB -N
#BSUB -u s204696@dtu.dk
#BSUB -o /work3/s204696/logs/lsf/train_e7_%J.out
#BSUB -e /work3/s204696/logs/lsf/train_e7_%J.err
#
# E7 experiment: resume E3 to 55k steps.
# E3 (lr=5e-5, no freeze) was still converging at 30k steps (WER 35.78%).
# This job resumes from the E3 checkpoint by pointing at the same output_dir.
# fairseq2 auto-resumes from the latest checkpoint found in RUN_DIR.
# See docs/experiment-plan.md §E7.
#
# Usage:
#   bsub < scripts/hpc/13_train_e7.sh
#
# Resume support: resubmitting this script will auto-resume from the latest
# checkpoint if training is interrupted.

set -euo pipefail

# --- Environment ---
source "${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}/scripts/hpc/env.sh"
setup_omniasr

# Same dir as E3 — fairseq2 auto-resumes from existing step_30000 checkpoint.
RUN_DIR="${RESUME_DIR:-/work3/$USER/outputs/omniasr_e3}"
mkdir -p "$RUN_DIR"

python scripts/hpc/run_training.py \
    --config configs/fairseq2/ctc-finetune-hpc-e7.yaml \
    --output-dir "$RUN_DIR" \
    --wandb-tags "train,full,hpc,a100,e7,55k,lr5e-5,resume-e3" \
    --wandb-resume allow
