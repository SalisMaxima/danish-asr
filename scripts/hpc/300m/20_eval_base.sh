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
# Zero-shot greedy evaluation of pretrained omniASR_CTC_300M_v2 on 3 test splits.
# Uses the parquet harness (decode_ctc_with_lm.py) so per-corpus subset WER is
# correct — the fairseq2 recipe evaluates on the full combined set regardless of
# the split name suffix or dataset_summary_path TSV.
#
# Usage:
#   bsub < scripts/hpc/300m/20_eval_base.sh

set -euo pipefail

source "${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}/scripts/hpc/env.sh"
setup_omniasr

CHECKPOINT="/work3/$USER/fairseq2_cache/f33741966d852362f4d416d5/omniASR-CTC-300M-v2.pt"
DATASET_ROOT="/work3/$USER/data/parquet/version=0"
OUTPUT_ROOT="${OUTPUT_ROOT:-/work3/$USER/outputs/ctc_zero_shot}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DTYPE="${DTYPE:-bfloat16}"
OVERWRITE="${OVERWRITE:-false}"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT" >&2
    echo "ERROR: Re-run a zero-shot eval to repopulate the fairseq2 cache, or set CHECKPOINT=<path>." >&2
    exit 1
fi

echo "=== 300M Base (Zero-Shot) Greedy Evaluation ==="
echo "Checkpoint:  $CHECKPOINT"
echo "Output root: $OUTPUT_ROOT"
echo "Started:     $(date)"
echo "Node:        $(hostname)"
nvidia-smi

had_failures=0

run_split() {
    local split_label="$1"
    local dataset_split="$2"
    local run_dir="$OUTPUT_ROOT/300m_base/$split_label/greedy"

    if [[ -f "$run_dir/SUCCESS" && "$OVERWRITE" != "true" ]]; then
        echo "Skipping completed split: $split_label"
        return 0
    fi

    mkdir -p "$run_dir"
    rm -f "$run_dir/SUCCESS" "$run_dir/FAILED"

    echo ""
    echo "--- Split: $split_label ($dataset_split) ---"
    if python scripts/decode_ctc_with_lm.py \
        --checkpoint-path "$CHECKPOINT" \
        --model-arch 300m_v2 \
        --dataset-root "$DATASET_ROOT" \
        --dataset-split "$dataset_split" \
        --decoder greedy \
        --batch-size "$BATCH_SIZE" \
        --dtype "$DTYPE" \
        --output-dir "$run_dir" \
        > "$run_dir/run.log" 2>&1; then
        date > "$run_dir/SUCCESS"
    else
        echo "$?" > "$run_dir/FAILED"
        echo "ERROR: Eval failed for $split_label" >&2
        tail -40 "$run_dir/run.log" >&2 || true
        had_failures=1
    fi
}

run_split combined             test
run_split read_aloud           test_coral_v3_read_aloud
run_split conversation         test_coral_v3_conversation

echo ""
echo "Finished: $(date)"

if [ "$had_failures" -ne 0 ]; then
    echo "One or more eval splits failed." >&2
    exit 1
fi
