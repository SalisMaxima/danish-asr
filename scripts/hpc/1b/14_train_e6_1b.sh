#!/bin/bash
#BSUB -J danish_asr_train_e6_1b
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 40:00
#BSUB -B
#BSUB -N
#BSUB -u s204696@dtu.dk
#BSUB -o /work3/s204696/logs/lsf/train_e6_1b_%J.out
#BSUB -e /work3/s204696/logs/lsf/train_e6_1b_%J.err
#
# 1B equivalent of E6: omniASR_CTC_1B_v2 with fixed shuffle windows,
# lr=5e-5, and the VRAM-probe-backed batch shape
# (max_num_elements=1.92M, grad_accum=8).
#
# Usage:
#   bsub < scripts/hpc/1b/14_train_e6_1b.sh
#
# Resume support: resubmitting this script will auto-resume from the latest
# checkpoint if training is interrupted.

set -euo pipefail

source "${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}/scripts/hpc/env.sh"
setup_omniasr

RUN_DIR="${RESUME_DIR:-/work3/$USER/outputs/omniasr_e6_1b}"
mkdir -p "$RUN_DIR"

python scripts/hpc/run_training.py \
    --config configs/fairseq2/1b/ctc-finetune-hpc-e6-1b.yaml \
    --output-dir "$RUN_DIR" \
    --wandb-tags "train,full,hpc,a100,1b,e6,50k,lr5e-5,shuffle1000" \
    --wandb-resume allow
