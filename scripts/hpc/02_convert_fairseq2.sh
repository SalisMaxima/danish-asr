#!/bin/bash
#BSUB -J danish_asr_convert
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 8:00
#BSUB -o /work3/s204696/logs/lsf/convert_%J.out
#BSUB -e /work3/s204696/logs/lsf/convert_%J.err

set -euo pipefail

# --- Environment ---
export HF_HOME=/work3/s204696/hf_cache
export HF_DATASETS_CACHE=/work3/s204696/hf_cache/datasets
export FAIRSEQ2_CACHE_DIR=/work3/s204696/fairseq2_cache
export TMPDIR=/work3/s204696/tmp
mkdir -p "$TMPDIR"
mkdir -p /work3/s204696/logs/lsf
mkdir -p /work3/s204696/logs/python

cd "$HOME/danish_asr"
source .venv/bin/activate

echo "=== Job $LSB_JOBID: Convert to fairseq2 ==="
echo "Started: $(date)"
echo "Node: $(hostname)"

python scripts/hpc/convert_to_fairseq2.py \
    --universal-dir /work3/s204696/data/preprocessed \
    --fairseq2-dir /work3/s204696/data/parquet/version=0 \
    --subset all

echo "Finished: $(date)"
