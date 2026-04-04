#!/bin/bash
#BSUB -J danish_asr_verify
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 2:00
#BSUB -B
#BSUB -N
#BSUB -u s204696@dtu.dk
# -o/-e passed on bsub command line (shell variables are not expanded in #BSUB directives)
# Usage: submit via submit_pipeline.sh
#   or:  bsub -o /work3/$USER/logs/lsf/verify_%J.out -e /work3/$USER/logs/lsf/verify_%J.err < scripts/hpc/01_verify_data.sh

set -euo pipefail

# --- Environment ---
source "$(dirname "$0")/env.sh"

echo "=== Job $LSB_JOBID: Verify Data ==="
echo "Started: $(date)"
echo "Node: $(hostname)"

python scripts/hpc/verify_data.py --data-dir /work3/$USER/data/preprocessed

echo "Finished: $(date)"
