#!/bin/bash
#BSUB -J danish_asr_eval_e6_3b
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -R "select[gpu80gb]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 3:00
#BSUB -B
#BSUB -N
#BSUB -u s204696@dtu.dk
#BSUB -o /work3/s204696/logs/lsf/eval_e6_3b_%J.out
#BSUB -e /work3/s204696/logs/lsf/eval_e6_3b_%J.err
#
# Evaluate finetuned omniASR_CTC_3B_v2 (E6-3B, 30k steps) on the combined test split.
# This intentionally uses only the combined split because the current read_aloud /
# conversation configs do not truly filter the eval set for fairseq2.
#
# Usage:
#   bsub < scripts/hpc/3b/20_eval_e6_3b.sh
#
# Single-config override:
#   EVAL_CONFIG=configs/fairseq2/3b/ctc-eval-e6-3b.yaml \
#       bsub < scripts/hpc/3b/20_eval_e6_3b.sh

set -euo pipefail

source "${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}/scripts/hpc/env.sh"
setup_omniasr

EVAL_OUT_DIR="${EVAL_OUT_DIR:-/work3/$USER/outputs/omniasr_e6_3b_eval}"
if ! mkdir -p "$EVAL_OUT_DIR" 2>/dev/null; then
    echo "ERROR: Cannot create eval workspace: $EVAL_OUT_DIR" >&2
    echo "ERROR: Check /work3 quota with getquota_work3.sh" >&2
    exit 1
fi

CONFIG="${EVAL_CONFIG:-configs/fairseq2/3b/ctc-eval-e6-3b.yaml}"
TAGS="e6-3b,3b,30k,lr5e-5,shuffle1000,test,combined"

echo "=== Phase 10C: 3B Finetuned E6-3B Evaluation ==="
echo "Eval workspace:     $EVAL_OUT_DIR"
echo "Config:             $CONFIG"
echo "Started:            $(date)"
echo "Node:               $(hostname)"
nvidia-smi

if ! python scripts/hpc/run_eval.py \
    --checkpoint-dir "$EVAL_OUT_DIR" \
    --config "$CONFIG" \
    --wandb-tags "$TAGS"; then
    echo "ERROR: Eval failed for $CONFIG" >&2
    exit 1
fi

echo ""
echo "Finished: $(date)"
