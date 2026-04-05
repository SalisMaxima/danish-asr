#!/bin/bash
#BSUB -J danish_asr_convert
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 2:00
#BSUB -B
#BSUB -N
#BSUB -u s204696@dtu.dk
# -o/-e passed on bsub command line (shell variables are not expanded in #BSUB directives)
# Usage: submit via submit_pipeline.sh
#   or:  bsub -o /work3/$USER/logs/lsf/convert_%J.out -e /work3/$USER/logs/lsf/convert_%J.err < scripts/hpc/02_convert_fairseq2.sh

set -euo pipefail

# --- Environment ---
source "${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}/scripts/hpc/env.sh"

echo "=== Job $LSB_JOBID: Convert to fairseq2 ==="
echo "Started: $(date)"
echo "Node: $(hostname)"

python scripts/hpc/convert_to_fairseq2.py \
    --universal-dir /work3/$USER/data/preprocessed \
    --fairseq2-dir /work3/$USER/data/parquet/version=0 \
    --subset all

echo "Finished: $(date)"
