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
source "$(dirname "$0")/env.sh"
setup_omniasr

CHECKPOINT_DIR="/work3/$USER/outputs/omniasr_e2"
CONFIG="${EVAL_CONFIG:-configs/fairseq2/ctc-eval-e2.yaml}"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "ERROR: Checkpoint directory not found: $CHECKPOINT_DIR" >&2
    exit 1
fi

echo "=== Evaluating E2 checkpoint ==="
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Config:     $CONFIG"
echo "Started:    $(date)"
echo "Node:       $(hostname)"
nvidia-smi

python scripts/hpc/run_eval.py \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --config "$CONFIG" \
    --wandb-tags "e2,30k,lr3e-5,test"

echo "Finished: $(date)"
