#!/bin/bash
#BSUB -J w2v2_full
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -B
#BSUB -N
#BSUB -u s204696@dtu.dk
#BSUB -o /work3/s204696/logs/lsf/w2v2_full_%J.out
#BSUB -e /work3/s204696/logs/lsf/w2v2_full_%J.err
#
# Wav2Vec2-XLS-R-300M full CTC training on CoRal-v3 (~100k steps, ~57h).
#
# For multi-day training, submit sequential jobs with checkpoint resume:
#   JOB1: bsub < scripts/hpc/07_train_wav2vec2.sh
#   JOB2: RESUME_CKPT=latest bsub < scripts/hpc/07_train_wav2vec2.sh
#   JOB3: RESUME_CKPT=latest bsub < scripts/hpc/07_train_wav2vec2.sh
#
# Usage: bsub < scripts/hpc/07_train_wav2vec2.sh

set -euo pipefail

# --- Environment ---
source "${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}/scripts/hpc/env.sh"

# Fixed run directory — reused across sequential jobs so checkpoint resume works.
# Override with WAV2VEC2_RUN_DIR to use a different path.
RUN_DIR="${WAV2VEC2_RUN_DIR:-/work3/$USER/outputs/wav2vec2_full}"
mkdir -p "$RUN_DIR"

python -m scripts.hpc.train_wav2vec2 \
    --config configs/hf_baseline/wav2vec2_full.yaml \
    --output-dir "$RUN_DIR" \
    --wandb-tags "train,full,hpc,a100,wav2vec2" \
    --wandb-resume allow \
    ${RESUME_CKPT:+--resume-from-checkpoint "$RESUME_CKPT"}
