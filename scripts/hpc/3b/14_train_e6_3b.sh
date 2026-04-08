#!/bin/bash
#BSUB -J danish_asr_train_e6_3b
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -R "select[gpu80gb]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 64:00
#BSUB -B
#BSUB -N
#BSUB -u s204696@dtu.dk
#BSUB -o /work3/s204696/logs/lsf/train_e6_3b_%J.out
#BSUB -e /work3/s204696/logs/lsf/train_e6_3b_%J.err
#
# 3B equivalent of E6: omniASR_CTC_3B_v2 with fixed shuffle windows,
# lr=5e-5, 30k steps, and the VRAM-probe-backed batch shape
# (max_num_elements=960K, grad_accum=16).
#
# Timing note:
#   The successful 3B tiny VRAM probe averaged about 350s per 50 train steps,
#   which extrapolates to about 58.3h for 30k steps before validation/checkpoint
#   overhead. A 64h walltime leaves a reasonable buffer while staying below the
#   72h queue limit.
#
# Usage:
#   bsub < scripts/hpc/3b/14_train_e6_3b.sh
#
# Resume support: resubmitting this script will auto-resume from the latest
# checkpoint if training is interrupted.

set -euo pipefail

source "${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}/scripts/hpc/env.sh"
setup_omniasr

RUN_DIR="${RESUME_DIR:-/work3/$USER/outputs/omniasr_e6_3b}"
mkdir -p "$RUN_DIR"

python scripts/hpc/run_training.py \
    --config configs/fairseq2/3b/ctc-finetune-hpc-e6-3b.yaml \
    --output-dir "$RUN_DIR" \
    --wandb-tags "train,full,hpc,a100,80gb,3b,e6,30k,lr5e-5,shuffle1000" \
    --wandb-resume allow
