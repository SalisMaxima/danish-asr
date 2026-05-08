#!/usr/bin/env bash
#BSUB -J coral_ctc_alexandra_matrix
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "span[hosts=1]"
#BSUB -R "select[gpu80gb]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -B
#BSUB -N
#BSUB -u s204696@dtu.dk
#BSUB -o /work3/s204696/logs/lsf/coral_ctc_alexandra_matrix_%J.out
#BSUB -e /work3/s204696/logs/lsf/coral_ctc_alexandra_matrix_%J.err
#
# Run Alexandra-aligned CoRal-style benchmarks for the fixed 300M, 1B, and 3B checkpoints.
#
# This runner populates the main public-comparison table with two decoder rows:
#   1. CTC no_lm       -> greedy CTC
#   2. CTC LM-enabled  -> beam + KenLM
#
# Usage:
#   bsub < scripts/hpc/benchmark_coral_style_alexandra_matrix.sh
#
# Smoke run:
#   MAX_SAMPLES=5 bsub < scripts/hpc/benchmark_coral_style_alexandra_matrix.sh

set -euo pipefail

export DANISH_ASR_PROJECT_DIR="${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/work3/$USER/outputs/coral_style_benchmark_alexandra}"
BATCH_SIZE="${BATCH_SIZE:-2}"
DTYPE="${DTYPE:-bfloat16}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
KENLM_BINARY="${KENLM_BINARY:-/work3/$USER/artifacts/lm/danish_lm_v1_3gram.bin}"
BEAM_WIDTH="${BEAM_WIDTH:-64}"
ALPHA="${ALPHA:-0.6}"
BETA="${BETA:-0.5}"
TOKENIZER_MODEL_PATH="${TOKENIZER_MODEL_PATH:-}"

if [[ ! -f "$KENLM_BINARY" ]]; then
  echo "KENLM_BINARY not found: $KENLM_BINARY" >&2
  echo "Build it first with:" >&2
  echo "  bsub < scripts/hpc/build_lm_corpus.sh" >&2
  echo "  bsub < scripts/hpc/build_kenlm.sh" >&2
  exit 1
fi

source "$DANISH_ASR_PROJECT_DIR/scripts/hpc/env.sh"
setup_omniasr
cd "$DANISH_ASR_PROJECT_DIR"

echo "=== Alexandra-aligned CTC CoRal-style matrix ==="
echo "Output root:   $OUTPUT_ROOT"
echo "KenLM binary:  $KENLM_BINARY"
echo "Beam width:    $BEAM_WIDTH"
echo "Alpha/Beta:    $ALPHA / $BETA"
echo "Max samples:   ${MAX_SAMPLES:-full}"
echo "Started:       $(date)"
echo "Node:          $(hostname)"
nvidia-smi

run_one() {
  local label="$1"
  local arch="$2"
  local checkpoint="$3"
  local subset="$4"
  local decoder="$5"
  local report_label="$6"
  local output_leaf="$7"

  local args=(
    uv run python scripts/hpc/benchmark_coral_style.py
    --checkpoint-path "$checkpoint"
    --model-arch "$arch"
    --subset "$subset"
    --decoder "$decoder"
    --report-label "$report_label"
    --batch-size "$BATCH_SIZE"
    --dtype "$DTYPE"
    --output-dir "$OUTPUT_ROOT/$label/$subset/$output_leaf"
  )

  if [[ -n "$MAX_SAMPLES" ]]; then
    args+=(--max-samples "$MAX_SAMPLES")
  fi
  if [[ -n "$TOKENIZER_MODEL_PATH" ]]; then
    args+=(--tokenizer-model-path "$TOKENIZER_MODEL_PATH")
  fi
  if [[ "$decoder" == "beam" ]]; then
    args+=(
      --kenlm-binary "$KENLM_BINARY"
      --beam-width "$BEAM_WIDTH"
      --alpha "$ALPHA"
      --beta "$BETA"
    )
  fi

  echo "Running $label on $subset with $report_label"
  "${args[@]}"
}

for subset in read_aloud conversation; do
  for spec in \
    "omniasr_ctc_300m_e6_50k 300m_v2 /work3/s204696/outputs/omniasr_e6/ws_1.0bb2600b/checkpoints/step_50000/model" \
    "omniasr_ctc_1b_e6_50k 1b_v2 /work3/s204696/outputs/omniasr_e6_1b/ws_1.f85211dd/checkpoints/step_50000/model" \
    "omniasr_ctc_3b_e6_30k 3b_v2 /work3/s204696/outputs/omniasr_e6_3b/ws_1.2172dba0/checkpoints/step_30000/model"
  do
    # shellcheck disable=SC2086
    set -- $spec
    label="$1"
    arch="$2"
    checkpoint="$3"

    run_one "$label" "$arch" "$checkpoint" "$subset" "greedy" "CTC no_lm" "ctc_no_lm"
    run_one "$label" "$arch" "$checkpoint" "$subset" "beam" "CTC LM-enabled" "ctc_lm_enabled"
  done
done
