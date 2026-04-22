#!/bin/bash
#BSUB -J danish_asr_llm_300m_e1
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -B
#BSUB -N
#BSUB -u s204696@dtu.dk
#BSUB -o /work3/s204696/logs/lsf/llm_300m_e1_%J.out
#BSUB -e /work3/s204696/logs/lsf/llm_300m_e1_%J.err
#
# E1 full finetune: omniASR_LLM_300M_v2 on CoRal-v3 Danish, 20k steps.
# A100-40GB is sufficient for LLM_300M_v2 at max_audio_len=240k (15s).
#
# Timing estimate: LLM decoder is ~30x slower per step than CTC (~96x RTF
# vs ~1x). Expect 18-22h for 20k steps. 24h walltime includes buffer.
#
# Pre-requisites (run from login node before submitting):
#   invoke assets.pull-llm --size 300m
#   bsub < scripts/hpc/llm_300m/05_smoke_test.sh  # verify first
#
# Usage:
#   bsub < scripts/hpc/llm_300m/14_train_e1.sh
#
# Resume support: resubmitting this script will auto-resume from the latest
# checkpoint if training is interrupted.

set -euo pipefail

source "${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}/scripts/hpc/env.sh"
setup_omniasr

RUN_DIR="${RESUME_DIR:-/work3/$USER/outputs/omniasr_llm_300m_e1}"
mkdir -p "$RUN_DIR"

python scripts/hpc/run_training.py \
    --config configs/fairseq2/llm_300m/llm-finetune-hpc-e1.yaml \
    --output-dir "$RUN_DIR" \
    --wandb-tags "train,full,hpc,a100,llm_300m_v2,e1,20k,lr5e-5,shuffle1000" \
    --wandb-resume allow
