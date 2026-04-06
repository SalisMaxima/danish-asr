#!/bin/bash
#BSUB -J danish_asr_eval_e2
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 1:00
#BSUB -B
#BSUB -N
#BSUB -u s204696@dtu.dk
#BSUB -o /work3/s204696/logs/lsf/eval_e2_%J.out
#BSUB -e /work3/s204696/logs/lsf/eval_e2_%J.err
#
# Evaluate the E2 checkpoint (autumn-dawn, 30k steps, lr=3e-5) on the
# held-out CoRal-v3 TEST split (read_aloud + conversation combined).
#
# For per-subset evaluation, first generate subset TSVs on HPC:
#   python scripts/hpc/make_subset_tsv.py --subset read_aloud \
#       --output data/parquet/version=0/language_distribution_read_aloud.tsv
#   python scripts/hpc/make_subset_tsv.py --subset conversation \
#       --output data/parquet/version=0/language_distribution_conversation.tsv
# Then resubmit with CONFIG overridden to ctc-eval-e2-read-aloud.yaml or
# ctc-eval-e2-conversation.yaml.
#
# Usage:
#   bsub < scripts/hpc/11_eval_e2.sh

set -euo pipefail

# --- Environment ---
source "${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}/scripts/hpc/env.sh"
setup_omniasr

# Fresh output dir for eval — keeps eval workspace separate from training workspace.
# The recipe silently no-ops when run against a completed training workspace.
EVAL_OUT_DIR="${EVAL_OUT_DIR:-/work3/$USER/outputs/omniasr_e2_eval}"
# Catch /work3 quota exhaustion before burning GPU time.
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

CONFIG="${EVAL_CONFIG:-configs/fairseq2/300m/ctc-eval-e2.yaml}"
CHECKPOINT_DIR="/work3/$USER/outputs/omniasr_e2"  # training workspace, existence check only

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "ERROR: Expected training workspace not found: $CHECKPOINT_DIR" >&2
    echo "ERROR: The evaluated model checkpoint is configured via model.path in $CONFIG." >&2
    exit 1
fi

echo "=== Evaluating E2 checkpoint ==="
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
    --wandb-tags "e2,30k,lr3e-5,test"; then
    echo "ERROR: run_eval.py failed — see output above for details." >&2
    exit 1
fi

echo "Finished: $(date)"
