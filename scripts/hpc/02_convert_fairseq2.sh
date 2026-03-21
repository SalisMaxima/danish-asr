#!/bin/bash
#BSUB -J danish_asr_convert
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 2:00
# -o/-e passed on bsub command line (shell variables are not expanded in #BSUB directives)
# Usage: submit via submit_pipeline.sh
#   or:  bsub -o /work3/$USER/logs/lsf/convert_%J.out -e /work3/$USER/logs/lsf/convert_%J.err < scripts/hpc/02_convert_fairseq2.sh

set -euo pipefail

# --- Environment ---
export HF_HOME=/work3/$USER/hf_cache
export HF_DATASETS_CACHE=/work3/$USER/hf_cache/datasets
export FAIRSEQ2_CACHE_DIR=/work3/$USER/fairseq2_cache
export TMPDIR=/work3/$USER/tmp
mkdir -p "$TMPDIR"
mkdir -p /work3/$USER/logs/lsf
mkdir -p /work3/$USER/logs/python

PROJECT_DIR="${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}"
cd "$PROJECT_DIR"
source .venv/bin/activate

echo "=== Job $LSB_JOBID: Convert to fairseq2 ==="
echo "Started: $(date)"
echo "Node: $(hostname)"

python scripts/hpc/convert_to_fairseq2.py \
    --universal-dir /work3/$USER/data/preprocessed \
    --fairseq2-dir /work3/$USER/data/parquet/version=0 \
    --subset all

echo "Finished: $(date)"
