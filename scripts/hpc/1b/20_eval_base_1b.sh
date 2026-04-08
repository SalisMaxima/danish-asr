#!/bin/bash
#BSUB -J danish_asr_eval_base_1b
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 1:00
#BSUB -B
#BSUB -N
#BSUB -u s204696@dtu.dk
#BSUB -o /work3/s204696/logs/lsf/eval_base_1b_%J.out
#BSUB -e /work3/s204696/logs/lsf/eval_base_1b_%J.err
#
# Zero-shot evaluation of pretrained omniASR_CTC_1B_v2 on 3 test splits.
# No checkpoint needed — model loaded from fairseq2 asset registry.
#
# Usage:
#   bsub < scripts/hpc/1b/20_eval_base_1b.sh
#
# Single-split override:
#   EVAL_CONFIG=configs/fairseq2/1b/ctc-eval-base-read-aloud.yaml \
#       bsub < scripts/hpc/1b/20_eval_base_1b.sh

set -euo pipefail

source "${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}/scripts/hpc/env.sh"
setup_omniasr

EVAL_OUT_DIR="${EVAL_OUT_DIR:-/work3/$USER/outputs/omniasr_base_1b_eval}"
if ! mkdir -p "$EVAL_OUT_DIR" 2>/dev/null; then
    echo "ERROR: Cannot create eval workspace: $EVAL_OUT_DIR" >&2
    echo "ERROR: Check /work3 quota with getquota_work3.sh" >&2
    exit 1
fi

echo "=== Phase 8B: 1B Base (Zero-Shot) Evaluation ==="
echo "Eval workspace: $EVAL_OUT_DIR"
echo "Started:        $(date)"
echo "Node:           $(hostname)"
nvidia-smi

CONFIGS=(
    "configs/fairseq2/1b/ctc-eval-base.yaml"
    "configs/fairseq2/1b/ctc-eval-base-read-aloud.yaml"
    "configs/fairseq2/1b/ctc-eval-base-conversation.yaml"
)
TAGS=(
    "base,1b,zero-shot,test,combined"
    "base,1b,zero-shot,test,read_aloud"
    "base,1b,zero-shot,test,conversation"
)

for i in "${!CONFIGS[@]}"; do
    CONFIG="${EVAL_CONFIG:-${CONFIGS[$i]}}"
    echo ""
    echo "--- Split $((i+1))/3: $CONFIG ---"

    if ! python scripts/hpc/run_eval.py \
        --checkpoint-dir "$EVAL_OUT_DIR" \
        --config "$CONFIG" \
        --wandb-tags "${TAGS[$i]}"; then
        echo "ERROR: Eval failed for $CONFIG" >&2
    fi

    if [ -n "${EVAL_CONFIG:-}" ]; then break; fi
done

echo ""
echo "Finished: $(date)"
