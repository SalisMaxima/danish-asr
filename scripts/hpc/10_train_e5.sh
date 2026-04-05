#!/bin/bash
#BSUB -J danish_asr_train_e5
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 8:00
#BSUB -B
#BSUB -N
#BSUB -u s204696@dtu.dk
#BSUB -o /work3/s204696/logs/lsf/train_e5_%J.out
#BSUB -e /work3/s204696/logs/lsf/train_e5_%J.err
#
# E5 experiment: combined best settings.
# lr=3e-5 (from E2) + freeze_encoder 2k steps + 40k total steps.
# Curves were still descending at E2 step 30k (WER 38.6%), so 40k gives more headroom.
# See docs/experiment-plan.md §E5.
#
# Usage:
#   bsub < scripts/hpc/10_train_e5.sh
#
# Resume support: resubmitting this script with the same RUN_DIR will
# auto-resume from the latest checkpoint if training is interrupted.

set -euo pipefail

# --- Environment ---
source "${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}/scripts/hpc/env.sh"
setup_omniasr

RUN_DIR="${RESUME_DIR:-/work3/$USER/outputs/omniasr_e5}"
mkdir -p "$RUN_DIR"

python scripts/hpc/run_training.py \
    --config configs/fairseq2/ctc-finetune-hpc-e5.yaml \
    --output-dir "$RUN_DIR" \
    --wandb-tags "train,full,hpc,a100,e5,40k,lr3e-5,freeze2k" \
    --wandb-resume allow
