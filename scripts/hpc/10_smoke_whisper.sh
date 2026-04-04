#!/bin/bash
#BSUB -J whisper_smoke
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=20GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 0:45
#BSUB -B
#BSUB -N
#BSUB -u s204696@dtu.dk
#BSUB -o /work3/s204696/logs/lsf/whisper_smoke_%J.out
#BSUB -e /work3/s204696/logs/lsf/whisper_smoke_%J.err
#
# Whisper smoke test — validates HF Trainer pipeline in <15 min.
#
# Usage: bsub < scripts/hpc/10_smoke_whisper.sh

set -euo pipefail

# --- Environment ---
source "$(dirname "$0")/env.sh"

python -m scripts.hpc.train_whisper \
    --config configs/hf_baseline/whisper_smoke.yaml \
    --wandb-resume never \
    --wandb-tags "smoke,hpc,a100,whisper"
