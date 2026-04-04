#!/bin/bash
#BSUB -J danish_asr_eval_e3
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 1:00
#BSUB -B
#BSUB -N
#BSUB -u s204696@dtu.dk
#BSUB -o /work3/s204696/logs/lsf/eval_e3_%J.out
#BSUB -e /work3/s204696/logs/lsf/eval_e3_%J.err
#
# Evaluate the E3 checkpoint (wobbly-pond-25, 30k steps, lr=5e-5) on the
# held-out CoRal-v3 TEST split (read_aloud + conversation combined).
#
# For per-subset evaluation, first generate subset TSVs on HPC:
#   python scripts/hpc/make_subset_tsv.py --subset read_aloud \
#       --output data/parquet/version=0/language_distribution_read_aloud.tsv
#   python scripts/hpc/make_subset_tsv.py --subset conversation \
#       --output data/parquet/version=0/language_distribution_conversation.tsv
# Then resubmit with CONFIG overridden to ctc-eval-e3-read-aloud.yaml or
# ctc-eval-e3-conversation.yaml.
#
# Usage:
#   bsub < scripts/hpc/12_eval_e3.sh

set -euo pipefail

# --- Environment ---
source "$(dirname "$0")/env.sh"
setup_omniasr

# Fresh output dir for eval — keeps eval workspace separate from training workspace.
# The recipe silently no-ops when run against a completed training workspace.
EVAL_OUT_DIR="${EVAL_OUT_DIR:-/work3/$USER/outputs/omniasr_e3_eval}"
mkdir -p "$EVAL_OUT_DIR"
CONFIG="${EVAL_CONFIG:-configs/fairseq2/ctc-eval-e3.yaml}"
CHECKPOINT_DIR="/work3/$USER/outputs/omniasr_e3"  # kept for existence check only

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "ERROR: Checkpoint directory not found: $CHECKPOINT_DIR" >&2
    exit 1
fi

echo "=== Evaluating E3 checkpoint ==="
echo "Training workspace (existence check only): $CHECKPOINT_DIR"
echo "Checkpoint source: hardcoded via model.path in $CONFIG"
echo "Eval workspace (--checkpoint-dir):         $EVAL_OUT_DIR"
echo "Config:     $CONFIG"
echo "Started:    $(date)"
echo "Node:       $(hostname)"
nvidia-smi

python scripts/hpc/run_eval.py \
    --checkpoint-dir "$EVAL_OUT_DIR" \
    --config "$CONFIG" \
    --wandb-tags "e3,30k,lr5e-5,test"

echo "Finished: $(date)"
