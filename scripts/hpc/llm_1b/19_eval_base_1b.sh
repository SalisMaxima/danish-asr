#!/bin/bash
#BSUB -J danish_asr_eval_llm_1b_base
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -R "select[gpu80gb]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 16:00
#BSUB -B
#BSUB -N
#BSUB -u s204696@dtu.dk
#BSUB -o /work3/s204696/logs/lsf/eval_llm_1b_base_%J.out
#BSUB -e /work3/s204696/logs/lsf/eval_llm_1b_base_%J.err
#
# Zero-shot evaluation of pretrained omniASR_LLM_1B_v2 on 3 test splits
# using the same old fairseq2 evaluation path as the finetuned LLM V2 runs.
# No checkpoint path is needed; fairseq2 loads the pretrained asset by model.name.
#
# Usage:
#   bsub < scripts/hpc/llm_1b/19_eval_base_1b.sh
#
# Optional single-split override:
#   EVAL_CONFIG=configs/fairseq2/llm_1b/llm-eval-base-1b-read-aloud.yaml \
#       bsub < scripts/hpc/llm_1b/19_eval_base_1b.sh

set -euo pipefail

source "${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}/scripts/hpc/env.sh"
setup_omniasr

EVAL_OUT_DIR="${EVAL_OUT_DIR:-/work3/$USER/outputs/omniasr_llm_1b_base_eval}"
if ! mkdir -p "$EVAL_OUT_DIR" 2>/dev/null; then
    echo "ERROR: Cannot create eval workspace: $EVAL_OUT_DIR" >&2
    echo "ERROR: Check /work3 quota with getquota_work3.sh" >&2
    exit 1
fi

echo "=== LLM 1B V2 Base (Zero-Shot) Evaluation ==="
echo "Eval workspace: $EVAL_OUT_DIR"
echo "Started:        $(date)"
echo "Node:           $(hostname)"
nvidia-smi

PREPARED_CONFIG_DIR="$EVAL_OUT_DIR/prepared_configs"
SUBSET_ROOT_PARENT="$EVAL_OUT_DIR/parquet_subsets"
mkdir -p "$PREPARED_CONFIG_DIR" "$SUBSET_ROOT_PARENT"

had_failures=0

split_tag() {
    case "$1" in
        *read-aloud*)   echo "read_aloud" ;;
        *conversation*) echo "conversation" ;;
        *)              echo "combined" ;;
    esac
}

subset_corpus() {
    case "$1" in
        *read-aloud*)   echo "coral_v3_read_aloud" ;;
        *conversation*) echo "coral_v3_conversation" ;;
        *)              echo "" ;;
    esac
}

CONFIGS=(
    "configs/fairseq2/llm_1b/llm-eval-base-1b.yaml"
    "configs/fairseq2/llm_1b/llm-eval-base-1b-read-aloud.yaml"
    "configs/fairseq2/llm_1b/llm-eval-base-1b-conversation.yaml"
)

for i in "${!CONFIGS[@]}"; do
    CONFIG="${EVAL_CONFIG:-${CONFIGS[$i]}}"
    SPLIT_TAG="$(split_tag "$CONFIG")"
    CORPUS="$(subset_corpus "$CONFIG")"
    PREPARED_CONFIG="$PREPARED_CONFIG_DIR/$(basename "$CONFIG")"
    TAGS="llm_1b_v2,base,zero-shot,eval,test,$SPLIT_TAG"
    echo ""
    echo "--- Split $((i+1))/3: $CONFIG ---"

    PREPARE_ARGS=(
        --config "$CONFIG"
        --output-config "$PREPARED_CONFIG"
        --subset-root-parent "$SUBSET_ROOT_PARENT"
    )
    if [ -n "$CORPUS" ]; then
        PREPARE_ARGS+=(--subset-corpus "$CORPUS")
    fi
    python scripts/hpc/prepare_parquet_subset_eval.py "${PREPARE_ARGS[@]}"

    if ! python scripts/hpc/run_eval.py \
        --checkpoint-dir "$EVAL_OUT_DIR" \
        --config "$PREPARED_CONFIG" \
        --wandb-project danish-asr-llm-v2 \
        --wandb-tags "$TAGS"; then
        echo "ERROR: Eval failed for $CONFIG" >&2
        had_failures=1
    fi

    if [ -n "${EVAL_CONFIG:-}" ]; then break; fi
done

echo ""
echo "Finished: $(date)"

if [ "$had_failures" -ne 0 ]; then
    echo "One or more eval splits failed." >&2
    exit 1
fi
