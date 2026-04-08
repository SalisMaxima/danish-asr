#!/bin/bash
#BSUB -J danish_asr_eval_e6_1b
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 2:00
#BSUB -B
#BSUB -N
#BSUB -u s204696@dtu.dk
#BSUB -o /work3/s204696/logs/lsf/eval_e6_1b_%J.out
#BSUB -e /work3/s204696/logs/lsf/eval_e6_1b_%J.err
#
# Evaluate finetuned omniASR_CTC_1B_v2 (E6-1B, 50k steps) on 3 test splits.
# Walltime 2:00 covers 3 sequential splits (~30-40 min each).
#
# Usage:
#   bsub < scripts/hpc/1b/21_eval_e6_1b.sh
#
# Single-split override:
#   EVAL_CONFIG=configs/fairseq2/1b/ctc-eval-e6-1b-read-aloud.yaml \
#       bsub < scripts/hpc/1b/21_eval_e6_1b.sh

set -euo pipefail

source "${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}/scripts/hpc/env.sh"
setup_omniasr

EVAL_OUT_DIR="${EVAL_OUT_DIR:-/work3/$USER/outputs/omniasr_e6_1b_eval}"
if ! mkdir -p "$EVAL_OUT_DIR" 2>/dev/null; then
    echo "ERROR: Cannot create eval workspace: $EVAL_OUT_DIR" >&2
    echo "ERROR: Check /work3 quota with getquota_work3.sh" >&2
    exit 1
fi

CHECKPOINT_DIR="/work3/$USER/outputs/omniasr_e6_1b"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "ERROR: Training workspace not found: $CHECKPOINT_DIR" >&2
    exit 1
fi

echo "=== Phase 8B: 1B Finetuned E6-1B Evaluation ==="
echo "Training workspace: $CHECKPOINT_DIR"
echo "Eval workspace:     $EVAL_OUT_DIR"
echo "Started:            $(date)"
echo "Node:               $(hostname)"
nvidia-smi

split_tag() {
    case "$1" in
        *read-aloud*)   echo "read_aloud" ;;
        *conversation*) echo "conversation" ;;
        *)              echo "combined" ;;
    esac
}

CONFIGS=(
    "configs/fairseq2/1b/ctc-eval-e6-1b.yaml"
    "configs/fairseq2/1b/ctc-eval-e6-1b-read-aloud.yaml"
    "configs/fairseq2/1b/ctc-eval-e6-1b-conversation.yaml"
)

for i in "${!CONFIGS[@]}"; do
    CONFIG="${EVAL_CONFIG:-${CONFIGS[$i]}}"
    TAGS="e6-1b,1b,50k,lr5e-5,shuffle1000,test,$(split_tag "$CONFIG")"
    echo ""
    echo "--- Split $((i+1))/3: $CONFIG ---"

    if ! python scripts/hpc/run_eval.py \
        --checkpoint-dir "$EVAL_OUT_DIR" \
        --config "$CONFIG" \
        --wandb-tags "$TAGS"; then
        echo "ERROR: Eval failed for $CONFIG" >&2
    fi

    if [ -n "${EVAL_CONFIG:-}" ]; then break; fi
done

echo ""
echo "Finished: $(date)"
