#!/bin/bash
#BSUB -J danish_asr_eval_base_300m
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 2:00
#BSUB -B
#BSUB -N
#BSUB -u s204696@dtu.dk
#BSUB -o /work3/s204696/logs/lsf/eval_base_300m_%J.out
#BSUB -e /work3/s204696/logs/lsf/eval_base_300m_%J.err
#
# Zero-shot evaluation of pretrained omniASR_CTC_300M_v2 on 3 test splits.
# No checkpoint needed — model loaded from fairseq2 asset registry.
# Walltime 2:00 covers 3 sequential splits (~30-40 min each).
#
# Usage:
#   bsub < scripts/hpc/300m/20_eval_base.sh
#
# Single-split override:
#   EVAL_CONFIG=configs/fairseq2/300m/ctc-eval-base-read-aloud.yaml \
#       bsub < scripts/hpc/300m/20_eval_base.sh

set -euo pipefail

source "${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}/scripts/hpc/env.sh"
setup_omniasr

EVAL_OUT_DIR="${EVAL_OUT_DIR:-/work3/$USER/outputs/omniasr_base_300m_eval}"
if ! mkdir -p "$EVAL_OUT_DIR" 2>/dev/null; then
    echo "ERROR: Cannot create eval workspace: $EVAL_OUT_DIR" >&2
    echo "ERROR: Check /work3 quota with getquota_work3.sh" >&2
    exit 1
fi

echo "=== Phase 8B: 300M Base (Zero-Shot) Evaluation ==="
echo "Eval workspace: $EVAL_OUT_DIR"
echo "Started:        $(date)"
echo "Node:           $(hostname)"
nvidia-smi

# Derive W&B split tag from the config filename suffix.
split_tag() {
    case "$1" in
        *read-aloud*)   echo "read_aloud" ;;
        *conversation*) echo "conversation" ;;
        *)              echo "combined" ;;
    esac
}

CONFIGS=(
    "configs/fairseq2/300m/ctc-eval-base.yaml"
    "configs/fairseq2/300m/ctc-eval-base-read-aloud.yaml"
    "configs/fairseq2/300m/ctc-eval-base-conversation.yaml"
)

for i in "${!CONFIGS[@]}"; do
    CONFIG="${EVAL_CONFIG:-${CONFIGS[$i]}}"
    TAGS="base,300m,zero-shot,test,$(split_tag "$CONFIG")"
    echo ""
    echo "--- Split $((i+1))/3: $CONFIG ---"

    if ! python scripts/hpc/run_eval.py \
        --checkpoint-dir "$EVAL_OUT_DIR" \
        --config "$CONFIG" \
        --wandb-tags "$TAGS"; then
        echo "ERROR: Eval failed for $CONFIG" >&2
    fi

    # If EVAL_CONFIG was set, run only that one config
    if [ -n "${EVAL_CONFIG:-}" ]; then break; fi
done

echo ""
echo "Finished: $(date)"
