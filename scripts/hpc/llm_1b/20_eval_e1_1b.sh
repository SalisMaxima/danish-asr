#!/bin/bash
#BSUB -J danish_asr_eval_llm_1b_e1
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -R "select[gpu80gb]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 4:00
#BSUB -B
#BSUB -N
#BSUB -u s204696@dtu.dk
#BSUB -o /work3/s204696/logs/lsf/eval_llm_1b_e1_%J.out
#BSUB -e /work3/s204696/logs/lsf/eval_llm_1b_e1_%J.err
#
# Evaluate finetuned omniASR_LLM_1B_v2 (E1, nominally 15k steps) on 3 test splits
# using the same old fairseq2 evaluation path as the greedy CTC runs.
#
# Usage:
#   bsub < scripts/hpc/llm_1b/20_eval_e1_1b.sh
#
# Optional overrides:
#   RUN_DIR=/work3/$USER/outputs/omniasr_llm_1b_e1_15k \
#       bsub < scripts/hpc/llm_1b/20_eval_e1_1b.sh
#   MODEL_PATH=/work3/$USER/outputs/omniasr_llm_1b_e1_15k/ws_1.<hash>/checkpoints/step_15000/model \
#       bsub < scripts/hpc/llm_1b/20_eval_e1_1b.sh
#   EVAL_CONFIG=configs/fairseq2/llm_1b/llm-eval-e1-1b-15k-read-aloud.yaml \
#       bsub < scripts/hpc/llm_1b/20_eval_e1_1b.sh

set -euo pipefail

source "${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}/scripts/hpc/env.sh"
setup_omniasr

RUN_DIR="${RUN_DIR:-/work3/$USER/outputs/omniasr_llm_1b_e1_15k}"
EXPECTED_STEP="${EXPECTED_STEP:-step_15000}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-/work3/$USER/outputs/omniasr_llm_1b_e1_eval}"

if ! mkdir -p "$EVAL_OUT_DIR" 2>/dev/null; then
    echo "ERROR: Cannot create eval workspace: $EVAL_OUT_DIR" >&2
    echo "ERROR: Check /work3 quota with getquota_work3.sh" >&2
    exit 1
fi

resolve_model_path() {
    if [ -n "${MODEL_PATH:-}" ]; then
        echo "$MODEL_PATH"
        return 0
    fi

    local ws_dir="${WORKSPACE_DIR:-}"
    if [ -z "$ws_dir" ]; then
        if compgen -G "$RUN_DIR/ws_1.*" > /dev/null; then
            ws_dir="$(ls -td "$RUN_DIR"/ws_1.* | head -n1)"
        fi
    fi

    if [ -z "$ws_dir" ] || [ ! -d "$ws_dir" ]; then
        echo "ERROR: Could not resolve a fairseq2 workspace under $RUN_DIR" >&2
        echo "ERROR: Set WORKSPACE_DIR=/work3/\$USER/outputs/omniasr_llm_1b_e1_15k/ws_1.<hash> if needed." >&2
        exit 1
    fi

    local candidate="$ws_dir/checkpoints/$EXPECTED_STEP/model"
    if [ -d "$candidate" ]; then
        echo "$candidate"
        return 0
    fi

    candidate="$(find "$ws_dir/checkpoints" -maxdepth 1 -type d -name 'step_*' | sort -V | tail -n1)/model"
    if [ -d "$candidate" ]; then
        echo "$candidate"
        return 0
    fi

    echo "ERROR: Could not resolve a checkpoint model directory under $ws_dir/checkpoints" >&2
    exit 1
}

render_config() {
    local template="$1"
    local model_path="$2"
    local rendered
    rendered="$(mktemp /tmp/llm_eval_1b_XXXXXX.yaml)"
    sed "s|__MODEL_PATH__|$model_path|g" "$template" > "$rendered"
    echo "$rendered"
}

echo "=== LLM 1B V2 E1 Evaluation (old fairseq2 path) ==="
echo "Eval workspace:     $EVAL_OUT_DIR"
echo "Run dir:            $RUN_DIR"
echo "Started:            $(date)"
echo "Node:               $(hostname)"
nvidia-smi

MODEL_PATH="$(resolve_model_path)"
echo "Resolved checkpoint: $MODEL_PATH"

had_failures=0

split_tag() {
    case "$1" in
        *read-aloud*)   echo "read_aloud" ;;
        *conversation*) echo "conversation" ;;
        *)              echo "combined" ;;
    esac
}

CONFIGS=(
    "configs/fairseq2/llm_1b/llm-eval-e1-1b-15k.yaml"
    "configs/fairseq2/llm_1b/llm-eval-e1-1b-15k-read-aloud.yaml"
    "configs/fairseq2/llm_1b/llm-eval-e1-1b-15k-conversation.yaml"
)

for i in "${!CONFIGS[@]}"; do
    CONFIG_TEMPLATE="${EVAL_CONFIG:-${CONFIGS[$i]}}"
    CONFIG_RENDERED="$(render_config "$CONFIG_TEMPLATE" "$MODEL_PATH")"
    TAGS="llm_1b_v2,e1,15k,eval,test,$(split_tag "$CONFIG_TEMPLATE")"

    echo ""
    echo "--- Split $((i+1))/3: $CONFIG_TEMPLATE ---"

    if ! python scripts/hpc/run_eval.py \
        --checkpoint-dir "$EVAL_OUT_DIR" \
        --config "$CONFIG_RENDERED" \
        --wandb-project danish-asr-llm-v2 \
        --wandb-tags "$TAGS"; then
        echo "ERROR: Eval failed for $CONFIG_TEMPLATE" >&2
        had_failures=1
    fi

    rm -f "$CONFIG_RENDERED"

    if [ -n "${EVAL_CONFIG:-}" ]; then break; fi
done

echo ""
echo "Finished: $(date)"

if [ "$had_failures" -ne 0 ]; then
    echo "One or more eval splits failed." >&2
    exit 1
fi
