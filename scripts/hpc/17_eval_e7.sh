#!/bin/bash
#BSUB -J danish_asr_eval_e7
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 1:00
#BSUB -B
#BSUB -N
#BSUB -u s204696@dtu.dk
#BSUB -o /work3/s204696/logs/lsf/eval_e7_%J.out
#BSUB -e /work3/s204696/logs/lsf/eval_e7_%J.err
#
# Evaluate the E7 checkpoint (true-firefly-27, crashed at step 53900, lr=5e-5, resume-e3)
# on the held-out CoRal-v3 TEST split (read_aloud + conversation combined).
#
# IMPORTANT: E7 crashed before completing 55k steps. Before submitting this script:
#   1. SSH to HPC and find the E7 workspace hash:
#        ls /work3/s204696/outputs/omniasr_e3/
#      (E7 created a new ws_1.XXXXXXXX directory distinct from E3's ws_1.88015460)
#   2. Update model.path in configs/fairseq2/ctc-eval-e7.yaml with the correct path,
#      e.g.: /work3/s204696/outputs/omniasr_e3/ws_1.XXXXXXXX/checkpoints/step_53000
#
# Usage:
#   bsub < scripts/hpc/17_eval_e7.sh

set -euo pipefail

# --- Environment ---
source "${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}/scripts/hpc/env.sh"
setup_omniasr

EVAL_OUT_DIR="${EVAL_OUT_DIR:-/work3/$USER/outputs/omniasr_e7_eval}"
if ! mkdir -p "$EVAL_OUT_DIR" 2>/dev/null; then
    echo "ERROR: Cannot create eval workspace: $EVAL_OUT_DIR" >&2
    echo "ERROR: Check /work3 quota with getquota_work3.sh" >&2
    exit 1
fi
if ! touch "$EVAL_OUT_DIR/.write_test" 2>/dev/null; then
    echo "ERROR: Cannot write to eval workspace: $EVAL_OUT_DIR" >&2
    echo "ERROR: Check /work3 quota with getquota_work3.sh" >&2
    exit 1
fi
rm -f "$EVAL_OUT_DIR/.write_test" || true

CONFIG="${EVAL_CONFIG:-configs/fairseq2/ctc-eval-e7.yaml}"
CHECKPOINT_DIR="/work3/$USER/outputs/omniasr_e3"  # E7 shares omniasr_e3 output dir

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "ERROR: Expected training workspace not found: $CHECKPOINT_DIR" >&2
    echo "ERROR: The evaluated model checkpoint is configured via model.path in $CONFIG." >&2
    exit 1
fi

echo "=== Evaluating E7 checkpoint ==="
echo "Training workspace (existence check only): $CHECKPOINT_DIR"
echo "Checkpoint source: hardcoded via model.path in $CONFIG"
echo "Eval workspace (--checkpoint-dir):         $EVAL_OUT_DIR"
echo "Config:     $CONFIG"
echo "Started:    $(date)"
echo "Node:       $(hostname)"
nvidia-smi

if ! python scripts/hpc/run_eval.py \
    --checkpoint-dir "$EVAL_OUT_DIR" \
    --config "$CONFIG" \
    --wandb-tags "e7,53k,lr5e-5,resume-e3,test"; then
    echo "ERROR: run_eval.py failed — see output above for details." >&2
    exit 1
fi

echo "Finished: $(date)"
