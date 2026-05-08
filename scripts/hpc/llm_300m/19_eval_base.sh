#!/bin/bash
#BSUB -J danish_asr_eval_llm_300m_base
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 8:00
#BSUB -B
#BSUB -N
#BSUB -u s204696@dtu.dk
#BSUB -o /work3/s204696/logs/lsf/eval_llm_300m_base_%J.out
#BSUB -e /work3/s204696/logs/lsf/eval_llm_300m_base_%J.err
#
# Zero-shot evaluation of pretrained omniASR_LLM_300M_v2 on 3 test splits
# using the same old fairseq2 evaluation path as the finetuned LLM V2 runs.
# No checkpoint path is needed; fairseq2 loads the pretrained asset by model.name.
#
# Usage:
#   bsub < scripts/hpc/llm_300m/19_eval_base.sh
#
# Optional single-split override:
#   EVAL_CONFIG=configs/fairseq2/llm_300m/llm-eval-base-read-aloud.yaml \
#       bsub < scripts/hpc/llm_300m/19_eval_base.sh

set -euo pipefail

source "${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}/scripts/hpc/env.sh"
setup_omniasr

EVAL_OUT_DIR="${EVAL_OUT_DIR:-/work3/$USER/outputs/omniasr_llm_300m_base_eval}"
if ! mkdir -p "$EVAL_OUT_DIR" 2>/dev/null; then
    echo "ERROR: Cannot create eval workspace: $EVAL_OUT_DIR" >&2
    echo "ERROR: Check /work3 quota with getquota_work3.sh" >&2
    exit 1
fi

echo "=== LLM 300M V2 Base (Zero-Shot) Evaluation ==="
echo "Eval workspace: $EVAL_OUT_DIR"
echo "Started:        $(date)"
echo "Node:           $(hostname)"
nvidia-smi

# Generate per-corpus subset TSVs so the per-split configs can use valid_split: "test"
# with a corpus-filtered TSV. The "test_<corpus>" split-name suffix causes an immediate
# exit for wav2vec2_llama model families and is not used here.
MAIN_TSV="data/parquet/version=0/language_distribution_0.tsv"
{ head -1 "$MAIN_TSV"; awk -F'\t' 'NR>1 && $1 == "coral_v3_read_aloud"' "$MAIN_TSV"; } \
    > "data/parquet/version=0/language_distribution_read_aloud.tsv"
{ head -1 "$MAIN_TSV"; awk -F'\t' 'NR>1 && $1 == "coral_v3_conversation"' "$MAIN_TSV"; } \
    > "data/parquet/version=0/language_distribution_conversation.tsv"
echo "Generated per-corpus subset TSVs from $MAIN_TSV"

had_failures=0

split_tag() {
    case "$1" in
        *read-aloud*)   echo "read_aloud" ;;
        *conversation*) echo "conversation" ;;
        *)              echo "combined" ;;
    esac
}

CONFIGS=(
    "configs/fairseq2/llm_300m/llm-eval-base.yaml"
    "configs/fairseq2/llm_300m/llm-eval-base-read-aloud.yaml"
    "configs/fairseq2/llm_300m/llm-eval-base-conversation.yaml"
)

for i in "${!CONFIGS[@]}"; do
    CONFIG="${EVAL_CONFIG:-${CONFIGS[$i]}}"
    TAGS="llm_300m_v2,base,zero-shot,eval,test,$(split_tag "$CONFIG")"
    echo ""
    echo "--- Split $((i+1))/3: $CONFIG ---"

    if ! python scripts/hpc/run_eval.py \
        --checkpoint-dir "$EVAL_OUT_DIR" \
        --config "$CONFIG" \
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
