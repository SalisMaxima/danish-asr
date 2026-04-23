#!/bin/bash
#BSUB -J danish_asr_llm_1b_e1
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -R "select[gpu80gb]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 65:00
#BSUB -B
#BSUB -N
#BSUB -o /work3/%U/logs/lsf/llm_1b_e1_%J.out
#BSUB -e /work3/%U/logs/lsf/llm_1b_e1_%J.err
#
# E1 full finetune: omniASR_LLM_1B_v2 on CoRal-v3 Danish, 15k steps.
# Requires A100-80GB (2.3B total params + AdamW optimizer states + decoder
# activations exceed A100-40GB at max_audio_len=240k).
# Estimated runtime is about 55h from the 50-step smoke run, so this script
# requests 65h to leave room for validation and checkpoint overhead.
#
# Pre-requisites (run from login node before submitting):
#   invoke assets.pull-llm --size 1b
#   bsub < scripts/hpc/llm_1b/05_smoke_test.sh  # verify first
#
# Usage:
#   bsub < scripts/hpc/llm_1b/14_train_e1_1b.sh
#
# Resume support: resubmitting this script will auto-resume from the latest
# checkpoint if training is interrupted.

set -euo pipefail

source "${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}/scripts/hpc/env.sh"
setup_omniasr

RUN_DIR="${RESUME_DIR:-/work3/$USER/outputs/omniasr_llm_1b_e1_15k}"
mkdir -p "$RUN_DIR"

python scripts/hpc/run_training.py \
    --config configs/fairseq2/llm_1b/llm-finetune-hpc-e1-1b-15k.yaml \
    --output-dir "$RUN_DIR" \
    --wandb-project danish-asr-llm-v2 \
    --wandb-tags "train,full,hpc,a100,80gb,llm_1b_v2,e1,15k,lr5e-5,shuffle1000" \
    --wandb-resume allow
