#!/bin/bash
#BSUB -J danish_asr_convert
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 0:30
#BSUB -o /work3/s204696/logs/lsf/convert_%J.out
#BSUB -e /work3/s204696/logs/lsf/convert_%J.err

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

echo "=== Job $LSB_JOBID: Convert to fairseq2 ==="
echo "Started: $(date)"
echo "Node: $(hostname)"

python scripts/hpc/convert_to_fairseq2.py \
    --universal-dir /work3/$USER/data/preprocessed \
    --fairseq2-dir /work3/$USER/data/parquet/version=0 \
    --subset all

echo "Finished: $(date)"
