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
export HF_HOME=/work3/$USER/hf_cache
export HF_DATASETS_CACHE=/work3/$USER/hf_cache/datasets
export FAIRSEQ2_CACHE_DIR=/work3/$USER/fairseq2_cache
export TMPDIR=/work3/$USER/tmp
export WANDB_DIR=/work3/$USER/wandb
export WANDB_DATA_DIR=/work3/$USER/wandb
export WANDB_CACHE_DIR=/work3/$USER/wandb/cache
mkdir -p "$TMPDIR"
mkdir -p /work3/$USER/logs/lsf
mkdir -p /work3/$USER/wandb/cache

OMNI_ASR_DIR="/work3/$USER/omnilingual-asr"
if [ ! -d "$OMNI_ASR_DIR/workflows" ]; then
    echo "ERROR: omnilingual-asr repo not found at $OMNI_ASR_DIR" >&2
    echo "Clone it: git clone https://github.com/facebookresearch/omnilingual-asr.git $OMNI_ASR_DIR" >&2
    exit 1
fi
export PYTHONPATH="$OMNI_ASR_DIR:${PYTHONPATH:-}"

PROJECT_DIR="${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}"
cd "$PROJECT_DIR"
source .venv/bin/activate

# Fresh output dir for eval — keeps eval workspace separate from training workspace.
# The recipe silently no-ops when run against a completed training workspace.
EVAL_OUT_DIR="${EVAL_OUT_DIR:-/work3/$USER/outputs/omniasr_e2_eval}"
mkdir -p "$EVAL_OUT_DIR"
CONFIG="${EVAL_CONFIG:-configs/fairseq2/ctc-eval-e2.yaml}"
CHECKPOINT_DIR="/work3/$USER/outputs/omniasr_e2"  # kept for existence check only

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "ERROR: Checkpoint directory not found: $CHECKPOINT_DIR" >&2
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

python scripts/hpc/run_eval.py \
    --checkpoint-dir "$EVAL_OUT_DIR" \
    --config "$CONFIG" \
    --wandb-tags "e2,30k,lr3e-5,test"

echo "Finished: $(date)"
