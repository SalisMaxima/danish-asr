#!/bin/bash
#BSUB -J danish_asr_train_20k
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 20:00
#BSUB -B
#BSUB -N
#BSUB -u s204696@dtu.dk
#BSUB -o /work3/s204696/logs/lsf/train_%J.out
#BSUB -e /work3/s204696/logs/lsf/train_%J.err
#
# omniASR CTC 20k-step training on CoRal-v3 Danish (~8-16h on 1x A100).
#
# Resume support: fairseq2 auto-resumes when the output directory contains
# checkpoints. The fixed RUN_DIR below means resubmitting this script with the
# same RUN_DIR will automatically continue from the last checkpoint if one exists.
#
# Usage:
#   # Default: use fixed RUN_DIR. This will:
#   #   - start a fresh run if /work3/$USER/outputs/omniasr_20k is empty or new, or
#   #   - auto-resume if that directory already contains checkpoints.
#   bsub < scripts/hpc/03_train.sh
#
#   # Resume from (or start fresh in) a specific directory:
#   RESUME_DIR=/work3/$USER/outputs/omniasr_20k_run2 bsub < scripts/hpc/03_train.sh
#
#   # To force a truly fresh run when reusing the default path, either:
#   #   - delete or move /work3/$USER/outputs/omniasr_20k, or
#   #   - set RESUME_DIR to a new, unused output directory as above.

set -euo pipefail

# --- Environment ---
source "$(dirname "$0")/env.sh"
setup_omniasr

# Fixed run directory — reused across sequential jobs so checkpoint resume works.
# Override with RESUME_DIR to resume from a different path.
RUN_DIR="${RESUME_DIR:-/work3/$USER/outputs/omniasr_20k}"
mkdir -p "$RUN_DIR"

CONFIG="${TRAIN_CONFIG:-configs/fairseq2/ctc-finetune-hpc-20k.yaml}"

python scripts/hpc/run_training.py \
    --config "$CONFIG" \
    --output-dir "$RUN_DIR" \
    --wandb-tags "train,full,hpc,a100,20k" \
    --wandb-resume allow
