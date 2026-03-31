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
export HF_HOME=/work3/$USER/hf_cache
export HF_DATASETS_CACHE=/work3/$USER/hf_cache/datasets
export FAIRSEQ2_CACHE_DIR=/work3/$USER/fairseq2_cache
export TMPDIR=/work3/$USER/tmp
export WANDB_DIR=/work3/$USER/wandb
export WANDB_DATA_DIR=/work3/$USER/wandb
export WANDB_CACHE_DIR=/work3/$USER/wandb/cache
mkdir -p "$TMPDIR"
mkdir -p /work3/$USER/logs/lsf
mkdir -p /work3/$USER/wandb/cache

# The pip-installed omnilingual-asr package does not include the workflows/ recipe module
OMNI_ASR_DIR="/work3/$USER/omnilingual-asr"
if [ ! -d "$OMNI_ASR_DIR/workflows" ]; then
    echo "ERROR: omnilingual-asr repo not found at $OMNI_ASR_DIR" >&2
    echo "Clone it: git clone https://github.com/facebookresearch/omnilingual-asr.git $OMNI_ASR_DIR" >&2
    exit 1
fi
export PYTHONPATH="$OMNI_ASR_DIR:${PYTHONPATH:-}"

PROJECT_DIR="${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}"
cd "$PROJECT_DIR"
source .venv/bin/activate

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
