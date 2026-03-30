#!/bin/bash
#BSUB -J danish_asr_sweep[1-4]
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -B
#BSUB -N
#BSUB -u s204696@dtu.dk
#BSUB -o /work3/s204696/logs/lsf/sweep_%J_%I.out
#BSUB -e /work3/s204696/logs/lsf/sweep_%J_%I.err
# Usage:
#   SWEEP_ID=entity/project/abc123 bsub < scripts/hpc/06_sweep.sh
#
# Each array element runs --count sequential training runs.
# 4 array elements x 5 runs = 20 total (matches run_cap in sweep config).

set -euo pipefail

# --- Validate SWEEP_ID ---
if [ -z "${SWEEP_ID:-}" ]; then
    echo "ERROR: SWEEP_ID env var is required." >&2
    echo "Usage: SWEEP_ID=entity/project/abc123 bsub < scripts/hpc/06_sweep.sh" >&2
    exit 1
fi

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
PROJECT_DIR="${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}"

# Project root needed for scripts.hpc.common imports; omnilingual-asr for workflows.recipes
export PYTHONPATH="$PROJECT_DIR:$OMNI_ASR_DIR:${PYTHONPATH:-}"

cd "$PROJECT_DIR"
source .venv/bin/activate

echo "=== Sweep agent starting ==="
echo "SWEEP_ID: $SWEEP_ID"
echo "Array index: ${LSB_JOBINDEX:-N/A}"
echo "Job ID: ${LSB_JOBID:-N/A}"

wandb agent --count 5 "$SWEEP_ID"
