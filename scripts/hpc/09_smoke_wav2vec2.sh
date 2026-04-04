#!/bin/bash
#BSUB -J w2v2_smoke
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 0:35
#BSUB -B
#BSUB -N
#BSUB -u s204696@dtu.dk
#BSUB -o /work3/s204696/logs/lsf/w2v2_smoke_%J.out
#BSUB -e /work3/s204696/logs/lsf/w2v2_smoke_%J.err
#
# Wav2Vec2 smoke test — validates HF Trainer pipeline in <10 min.
#
# Usage: bsub < scripts/hpc/09_smoke_wav2vec2.sh

set -euo pipefail

# --- Environment ---
source "$(dirname "$0")/env.sh"

python -m scripts.hpc.train_wav2vec2 \
    --config configs/hf_baseline/wav2vec2_smoke.yaml \
    --wandb-resume never \
    --wandb-tags "smoke,hpc,a100,wav2vec2"
