#!/bin/bash
#BSUB -J danish_asr_smoke_val
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 1:00
#BSUB -B
#BSUB -N
#BSUB -u s204696@dtu.dk
#BSUB -o /work3/s204696/logs/lsf/smoke_val_%J.out
#BSUB -e /work3/s204696/logs/lsf/smoke_val_%J.err
# Usage: bsub < scripts/hpc/06_smoke_test_val.sh
# 500-step smoke test with WER validation at step 500.
# CTC blank phase ends ~100-500 steps (Meta recommendation).

set -euo pipefail

# --- Environment ---
export HF_HOME=/work3/$USER/hf_cache
export HF_DATASETS_CACHE=/work3/$USER/hf_cache/datasets
export FAIRSEQ2_CACHE_DIR=/work3/$USER/fairseq2_cache
export TMPDIR=/work3/$USER/tmp
mkdir -p "$TMPDIR"

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

python scripts/hpc/run_training.py \
    --config configs/fairseq2/ctc-finetune-smoke-val.yaml \
    --wandb-resume never \
    --wandb-tags "smoke,validation,hpc,a100"
