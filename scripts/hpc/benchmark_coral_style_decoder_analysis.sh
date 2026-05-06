#!/usr/bin/env bash
# Run the decoder-analysis CoRal-style benchmark for the fixed 3B E6 checkpoint.
#
# This runner populates the secondary interpretability table with:
#   1. greedy
#   2. beam
#   3. beam + KenLM
#
# Usage:
#   KENLM_BINARY=/work3/$USER/artifacts/lm/danish_lm_v1_3gram.bin \
#   bash scripts/hpc/benchmark_coral_style_decoder_analysis.sh

set -euo pipefail

export DANISH_ASR_PROJECT_DIR="${DANISH_ASR_PROJECT_DIR:-/zhome/00/8/147167/danish_asr}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/work3/$USER/outputs/coral_style_decoder_analysis}"
BATCH_SIZE="${BATCH_SIZE:-2}"
DTYPE="${DTYPE:-bfloat16}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
KENLM_BINARY="${KENLM_BINARY:-}"
BEAM_WIDTH="${BEAM_WIDTH:-64}"
ALPHAS="${ALPHAS:-0.6}"
BETAS="${BETAS:-0.5}"
TOKENIZER_MODEL_PATH="${TOKENIZER_MODEL_PATH:-}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/work3/s204696/outputs/omniasr_e6_3b/ws_1.2172dba0/checkpoints/step_30000/model}"
MODEL_ARCH="${MODEL_ARCH:-3b_v2}"
MODEL_LABEL="${MODEL_LABEL:-omniasr_ctc_3b_e6_30k}"

if [[ -z "$KENLM_BINARY" ]]; then
  echo "KENLM_BINARY must be set for the beam + KenLM analysis rows."
  exit 1
fi

source "$DANISH_ASR_PROJECT_DIR/scripts/hpc/env.sh"
setup_omniasr
cd "$DANISH_ASR_PROJECT_DIR"

run_one() {
  local subset="$1"
  local decoder="$2"
  local report_label="$3"
  local output_leaf="$4"
  local alpha="${5:-}"
  local beta="${6:-}"

  local args=(
    uv run python scripts/hpc/benchmark_coral_style.py
    --checkpoint-path "$CHECKPOINT_PATH"
    --model-arch "$MODEL_ARCH"
    --subset "$subset"
    --decoder "$decoder"
    --report-label "$report_label"
    --batch-size "$BATCH_SIZE"
    --dtype "$DTYPE"
    --output-dir "$OUTPUT_ROOT/$MODEL_LABEL/$subset/$output_leaf"
  )

  if [[ -n "$MAX_SAMPLES" ]]; then
    args+=(--max-samples "$MAX_SAMPLES")
  fi
  if [[ -n "$TOKENIZER_MODEL_PATH" ]]; then
    args+=(--tokenizer-model-path "$TOKENIZER_MODEL_PATH")
  fi
  if [[ "$decoder" == "beam" && -n "$alpha" && -n "$beta" ]]; then
    args+=(
      --kenlm-binary "$KENLM_BINARY"
      --beam-width "$BEAM_WIDTH"
      --alpha "$alpha"
      --beta "$beta"
    )
  elif [[ "$decoder" == "beam" ]]; then
    args+=(--beam-width "$BEAM_WIDTH")
  fi

  echo "Running $MODEL_LABEL on $subset with $report_label"
  "${args[@]}"
}

for subset in read_aloud conversation; do
  run_one "$subset" "greedy" "greedy" "greedy"
  run_one "$subset" "beam" "beam" "beam"

  for alpha in $ALPHAS; do
    for beta in $BETAS; do
      run_one "$subset" "beam" "beam + KenLM" "beam_lm_a${alpha}_b${beta}" "$alpha" "$beta"
    done
  done
done
