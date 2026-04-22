#!/bin/bash
#BSUB -J danish_asr_llm_300m_smoke
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 2:00
#BSUB -B
#BSUB -N
#BSUB -u s204696@dtu.dk
#BSUB -o /work3/s204696/logs/lsf/llm_300m_smoke_%J.out
#BSUB -e /work3/s204696/logs/lsf/llm_300m_smoke_%J.err
#
# 50-step smoke test for omniASR_LLM_300M_v2.
# Validates model load, data pipeline, forward/backward, and checkpoint write
# before committing to the full 20k-step training run.
#
# Pre-requisites:
#   - Checkpoint pre-pulled: invoke assets.pull-llm --size 300m
#   - Parquet data on work3: scripts/hpc/data/02_convert_fairseq2.sh
#
# Usage:
#   bsub < scripts/hpc/llm_300m/05_smoke_test.sh

set -euo pipefail

source "${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}/scripts/hpc/env.sh"
setup_omniasr

python scripts/hpc/run_training.py \
    --config configs/fairseq2/llm_300m/llm-finetune-smoke.yaml \
    --wandb-resume never \
    --wandb-tags "smoke,hpc,a100,llm_300m_v2,50steps"
