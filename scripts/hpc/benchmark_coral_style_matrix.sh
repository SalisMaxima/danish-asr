#!/usr/bin/env bash
# Run CoRal-style CER/WER benchmarks for the fixed 300M, 1B, and 3B fine-tuned checkpoints.
#
# Usage:
#   bash scripts/hpc/benchmark_coral_style_matrix.sh
#   MAX_SAMPLES=50 bash scripts/hpc/benchmark_coral_style_matrix.sh

set -euo pipefail

export DANISH_ASR_PROJECT_DIR="${DANISH_ASR_PROJECT_DIR:-/zhome/00/8/147167/danish_asr}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/work3/$USER/outputs/coral_style_benchmark}"
BATCH_SIZE="${BATCH_SIZE:-2}"
DTYPE="${DTYPE:-bfloat16}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

source "$DANISH_ASR_PROJECT_DIR/scripts/hpc/env.sh"
setup_omniasr
cd "$DANISH_ASR_PROJECT_DIR"

run_one() {
  local label="$1"
  local arch="$2"
  local checkpoint="$3"
  local subset="$4"

  local args=(
    uv run python scripts/hpc/benchmark_coral_style.py
    --checkpoint-path "$checkpoint"
    --model-arch "$arch"
    --subset "$subset"
    --batch-size "$BATCH_SIZE"
    --dtype "$DTYPE"
    --output-dir "$OUTPUT_ROOT/$label/$subset"
  )

  if [[ -n "$MAX_SAMPLES" ]]; then
    args+=(--max-samples "$MAX_SAMPLES")
  fi

  echo "Running $label on $subset"
  "${args[@]}"
}

for subset in read_aloud conversation; do
  run_one \
    "omniasr_ctc_300m_e6_50k" \
    "300m_v2" \
    "/work3/s204696/outputs/omniasr_e6/ws_1.0bb2600b/checkpoints/step_50000/model" \
    "$subset"

  run_one \
    "omniasr_ctc_1b_e6_50k" \
    "1b_v2" \
    "/work3/s204696/outputs/omniasr_e6_1b/ws_1.f85211dd/checkpoints/step_50000/model" \
    "$subset"

  run_one \
    "omniasr_ctc_3b_e6_30k" \
    "3b_v2" \
    "/work3/s204696/outputs/omniasr_e6_3b/ws_1.2172dba0/checkpoints/step_30000/model" \
    "$subset"
done
