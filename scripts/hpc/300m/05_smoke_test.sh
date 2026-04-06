#!/bin/bash
#BSUB -J danish_asr_smoke
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 0:35
#BSUB -B
#BSUB -N
#BSUB -u s204696@dtu.dk
#BSUB -o /work3/s204696/logs/lsf/smoke_%J.out
#BSUB -e /work3/s204696/logs/lsf/smoke_%J.err
# Usage: invoke train.hpc-smoke
#   or:  bsub < scripts/hpc/05_smoke_test.sh

set -euo pipefail

# --- Environment ---
source "${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}/scripts/hpc/env.sh"
setup_omniasr

python scripts/hpc/run_training.py \
    --config configs/fairseq2/300m/ctc-finetune-smoke.yaml \
    --wandb-resume never \
    --wandb-tags "smoke,validation,hpc,a100"
