#!/bin/bash
#BSUB -J danish_asr_train_e6_3b_50k_resume
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -R "select[gpu80gb]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 48:00
#BSUB -B
#BSUB -N
#BSUB -u s204696@dtu.dk
#BSUB -o /work3/s204696/logs/lsf/train_e6_3b_50k_resume_%J.out
#BSUB -e /work3/s204696/logs/lsf/train_e6_3b_50k_resume_%J.err
#
# Resume the original 3B E6 workspace from 30k to 50k total steps.
#
# This script must point at the resolved fairseq2 workspace directory from the
# completed 30k run, e.g.:
#
#   ORIGINAL_WS_DIR=/work3/$USER/outputs/omniasr_e6_3b/ws_1.<hash> \
#       bsub < scripts/hpc/3b/15_train_e6_3b_50k_resume.sh
#
# Why this works:
# - The continuation config sets `common.no_sweep_dir: true`, so fairseq2 uses
#   the exact output directory below instead of creating a new `ws_*` child.
# - fairseq2 then auto-resumes from the latest full checkpoint already present
#   in that workspace. This script checks for `step_30000/trainer` up front to
#   ensure the workspace contains resumable state, not just model weights.

set -euo pipefail

source "${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}/scripts/hpc/env.sh"
setup_omniasr

RUN_DIR="${ORIGINAL_WS_DIR:-${RESUME_DIR:-}}"

if [ -z "$RUN_DIR" ]; then
    echo "ERROR: Set ORIGINAL_WS_DIR (or RESUME_DIR) to the original 3B workspace." >&2
    echo "ERROR: Example: ORIGINAL_WS_DIR=/work3/\$USER/outputs/omniasr_e6_3b/ws_1.<hash> \\" >&2
    echo "ERROR:            bsub < scripts/hpc/3b/15_train_e6_3b_50k_resume.sh" >&2
    exit 1
fi

if [ ! -d "$RUN_DIR" ]; then
    echo "ERROR: Workspace directory not found: $RUN_DIR" >&2
    exit 1
fi

if [ ! -d "$RUN_DIR/checkpoints/step_30000/trainer" ]; then
    echo "ERROR: Full resumable checkpoint not found at:" >&2
    echo "ERROR:   $RUN_DIR/checkpoints/step_30000/trainer" >&2
    echo "ERROR: This resume path requires the original full checkpoint state" >&2
    echo "ERROR: (trainer/model/optimizer/data_reader), not only model weights." >&2
    exit 1
fi

echo "=== Phase 10C: 3B Resume From 30k To 50k ==="
echo "Workspace: $RUN_DIR"
echo "Started:   $(date)"
echo "Node:      $(hostname)"
nvidia-smi

python scripts/hpc/run_training.py \
    --config configs/fairseq2/3b/ctc-finetune-hpc-e6-3b-50k.yaml \
    --output-dir "$RUN_DIR" \
    --wandb-tags "train,full,hpc,a100,80gb,3b,e6,50k,resume,from30k,lr5e-5,shuffle1000" \
    --wandb-resume allow
