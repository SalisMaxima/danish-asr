#!/bin/bash
#BSUB -J danish_asr_eval
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 4:00
#BSUB -o /work3/%U/logs/lsf/eval_%J.out
#BSUB -e /work3/%U/logs/lsf/eval_%J.err

set -euo pipefail

# --- Environment ---
export HF_HOME=/work3/$USER/hf_cache
export HF_DATASETS_CACHE=/work3/$USER/hf_cache/datasets
export FAIRSEQ2_CACHE_DIR=/work3/$USER/fairseq2_cache
export TMPDIR=/work3/$USER/tmp
mkdir -p "$TMPDIR"
mkdir -p /work3/$USER/logs/lsf
mkdir -p /work3/$USER/logs/python

# PyTorch cu128 bundles its own CUDA runtime — no module load needed
PROJECT_DIR="${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}"
cd "$PROJECT_DIR"
source .venv/bin/activate

echo "=== Job $LSB_JOBID: Evaluate omniASR ==="
echo "Started: $(date)"
echo "Node: $(hostname)"
nvidia-smi

if [ -z "${CHECKPOINT_DIR:-}" ]; then
    echo "ERROR: CHECKPOINT_DIR not set. Pass via env or submit_pipeline.sh."
    exit 1
fi

python scripts/hpc/run_eval.py \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --config configs/fairseq2/ctc-finetune-hpc.yaml

echo "Finished: $(date)"
