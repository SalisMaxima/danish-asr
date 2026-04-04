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
source "$(dirname "$0")/env.sh"
setup_omniasr
# Project root also needed for scripts.hpc.common imports
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

echo "=== Sweep agent starting ==="
echo "SWEEP_ID: $SWEEP_ID"
echo "Array index: ${LSB_JOBINDEX:-N/A}"
echo "Job ID: ${LSB_JOBID:-N/A}"

wandb agent --count 5 "$SWEEP_ID"
