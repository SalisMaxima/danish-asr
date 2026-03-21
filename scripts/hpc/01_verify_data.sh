#!/bin/bash
#BSUB -J danish_asr_verify
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 2:00
#BSUB -o /work3/$USER/logs/lsf/verify_%J.out
#BSUB -e /work3/$USER/logs/lsf/verify_%J.err

set -euo pipefail

# --- Environment ---
export HF_HOME=/work3/$USER/hf_cache
export HF_DATASETS_CACHE=/work3/$USER/hf_cache/datasets
export FAIRSEQ2_CACHE_DIR=/work3/$USER/fairseq2_cache
export TMPDIR=/work3/$USER/tmp
mkdir -p "$TMPDIR"
mkdir -p /work3/$USER/logs/lsf
mkdir -p /work3/$USER/logs/python

cd "$HOME/danish_asr"
source .venv/bin/activate

echo "=== Job $LSB_JOBID: Verify Data ==="
echo "Started: $(date)"
echo "Node: $(hostname)"

python scripts/hpc/verify_data.py --data-dir /work3/$USER/data/preprocessed

echo "Finished: $(date)"
