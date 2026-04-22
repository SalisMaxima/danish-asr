#!/bin/bash
#BSUB -J danish_asr_llm_1b_smoke
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -R "select[gpu80gb]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 2:00
#BSUB -B
#BSUB -N
#BSUB -o /work3/%U/logs/lsf/llm_1b_smoke_%J.out
#BSUB -e /work3/%U/logs/lsf/llm_1b_smoke_%J.err
#
# 50-step smoke test for omniASR_LLM_1B_v2 (2.3B effective params).
# Requires A100-80GB: decoder activations + optimizer states exceed 40GB.
#
# Pre-requisites:
#   invoke assets.pull-llm --size 1b
#
# Usage:
#   bsub < scripts/hpc/llm_1b/05_smoke_test.sh

set -euo pipefail

source "${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}/scripts/hpc/env.sh"
setup_omniasr

python scripts/hpc/run_training.py \
    --config configs/fairseq2/llm_1b/llm-finetune-smoke.yaml \
    --wandb-project danish-asr-llm-v2 \
    --wandb-resume never \
    --wandb-tags "smoke,hpc,a100,80gb,llm_1b_v2,50steps"
