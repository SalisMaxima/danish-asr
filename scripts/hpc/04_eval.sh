#!/bin/bash
#BSUB -J danish_asr_eval
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 4:00
#BSUB -B
#BSUB -N
#BSUB -u s204696@dtu.dk
# -o/-e passed on bsub command line (shell variables are not expanded in #BSUB directives)
# Usage: submit via submit_pipeline.sh
#   or:  bsub -o /work3/$USER/logs/lsf/eval_%J.out -e /work3/$USER/logs/lsf/eval_%J.err < scripts/hpc/04_eval.sh

set -euo pipefail

# --- Environment ---
source "${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}/scripts/hpc/env.sh"
setup_omniasr

echo "=== Job $LSB_JOBID: Evaluate omniASR ==="
echo "Started: $(date)"
echo "Node: $(hostname)"
nvidia-smi

if [ -z "${CHECKPOINT_DIR:-}" ]; then
    echo "ERROR: CHECKPOINT_DIR not set. Pass via env or submit_pipeline.sh." >&2
    exit 1
fi

python scripts/hpc/run_eval.py \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --config configs/fairseq2/ctc-finetune-hpc.yaml

echo "Finished: $(date)"
