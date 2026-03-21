#!/bin/bash
#BSUB -J danish_asr_eval
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 4:00
#BSUB -o /work3/s204696/logs/lsf/eval_%J.out
#BSUB -e /work3/s204696/logs/lsf/eval_%J.err

set -euo pipefail

# --- Environment ---
export HF_HOME=/work3/s204696/hf_cache
export HF_DATASETS_CACHE=/work3/s204696/hf_cache/datasets
export FAIRSEQ2_CACHE_DIR=/work3/s204696/fairseq2_cache
export TMPDIR=/work3/s204696/tmp
mkdir -p "$TMPDIR"
mkdir -p /work3/s204696/logs/lsf
mkdir -p /work3/s204696/logs/python

module load cuda/11.7

cd "$HOME/danish_asr"
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
