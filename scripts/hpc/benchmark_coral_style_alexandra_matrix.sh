#!/bin/bash
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
# Run Alexandra-aligned CoRal-style benchmarks for fixed fine-tuned CTC checkpoints.
#
# Usage:
#   bsub < scripts/hpc/benchmark_coral_style_alexandra_matrix.sh
#
# Smoke run:
#   MAX_SAMPLES=5 bsub < scripts/hpc/benchmark_coral_style_alexandra_matrix.sh

set -euo pipefail

export DANISH_ASR_PROJECT_DIR="${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}"
MANIFEST="${CTC_KENLM_MANIFEST:-$DANISH_ASR_PROJECT_DIR/configs/eval/ctc_kenlm_finetuned_hpc.yaml}"
DTYPE="${DTYPE:-bfloat16}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
OVERWRITE="${OVERWRITE:-false}"
KENLM_BINARY="${KENLM_BINARY:-}"
BEAM_WIDTH="${BEAM_WIDTH:-}"
ALPHA="${ALPHA:-}"
BETA="${BETA:-}"
TOKENIZER_MODEL_PATH="${TOKENIZER_MODEL_PATH:-}"
UNIGRAMS_PATH="${UNIGRAMS_PATH:-}"
MIN_SECONDS="${MIN_SECONDS:-}"
MAX_SECONDS="${MAX_SECONDS:-}"

source "$DANISH_ASR_PROJECT_DIR/scripts/hpc/env.sh"
setup_omniasr
cd "$DANISH_ASR_PROJECT_DIR"

eval "$(python scripts/hpc/check_ctc_kenlm_eval_ready.py --manifest "$MANIFEST" --emit shell)"

OUTPUT_ROOT="${OUTPUT_ROOT:-$CTC_OUTPUT_ROOT_CORAL}"
KENLM_BINARY="${KENLM_BINARY:-$CTC_KENLM_BINARY}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$CTC_HF_CACHE_DIR}"
TOKENIZER_NAME="${TOKENIZER_NAME:-$CTC_TOKENIZER_NAME}"
TOKENIZER_MODEL_PATH="${TOKENIZER_MODEL_PATH:-$CTC_TOKENIZER_MODEL_PATH}"
BEAM_WIDTH="${BEAM_WIDTH:-$CTC_BEAM_WIDTH}"
ALPHA="${ALPHA:-$CTC_ALPHA}"
BETA="${BETA:-$CTC_BETA}"
MIN_SECONDS="${MIN_SECONDS:-$CTC_MIN_SECONDS}"
MAX_SECONDS="${MAX_SECONDS:-$CTC_MAX_SECONDS}"

python scripts/hpc/check_ctc_kenlm_eval_ready.py \
  --manifest "$MANIFEST" \
  --method coral \
  --print-quota

if ! mkdir -p "$OUTPUT_ROOT" 2>/dev/null; then
  echo "ERROR: Cannot create eval workspace: $OUTPUT_ROOT" >&2
  echo "ERROR: Check /work3 quota with getquota_work3.sh" >&2
  exit 1
fi

echo "=== Alexandra-aligned CTC CoRal-style matrix ==="
echo "Output root:   $OUTPUT_ROOT"
echo "KenLM binary:  $KENLM_BINARY"
echo "Unigrams:      ${UNIGRAMS_PATH:-none}"
echo "Beam width:    $BEAM_WIDTH"
echo "Alpha/Beta:    $ALPHA / $BETA"
echo "Duration:      $MIN_SECONDS < duration < $MAX_SECONDS"
echo "Max samples:   ${MAX_SAMPLES:-full}"
echo "Started:       $(date)"
echo "Node:          $(hostname)"
nvidia-smi

echo "=== Dependency versions ==="
python -c "
import sys, importlib
print('Python:', sys.version)
for pkg in ['torch', 'fairseq2', 'omnilingual_asr', 'pyctcdecode', 'kenlm', 'pyarrow', 'datasets']:
    try:
        m = importlib.import_module(pkg)
        print(f'{pkg}: {getattr(m, \"__version__\", \"(no __version__)\")!r}')
    except Exception as e:
        print(f'{pkg}: MISSING ({e})')
"
if [[ -f "$KENLM_BINARY" ]]; then
  echo "KenLM binary: $KENLM_BINARY ($(du -sh "$KENLM_BINARY" | cut -f1))"
else
  echo "KenLM binary: NOT FOUND ($KENLM_BINARY)"
fi
echo "==========================="

had_failures=0

run_one() {
  local label="$1"
  local arch="$2"
  local checkpoint="$3"
  local batch_size="$4"
  local subset="$5"
  local decoder="$6"
  local report_label="$7"
  local output_leaf="$8"
  local run_dir="$OUTPUT_ROOT/$label/$subset/$output_leaf"

  if [[ -f "$run_dir/SUCCESS" && "$OVERWRITE" != "true" ]]; then
    echo "Skipping completed run: $run_dir"
    return 0
  fi

  mkdir -p "$run_dir"
  rm -f "$run_dir/SUCCESS" "$run_dir/FAILED"

  local args=(
    python scripts/hpc/benchmark_coral_style.py
    --checkpoint-path "$checkpoint"
    --model-arch "$arch"
    --subset "$subset"
    --tokenizer-name "$TOKENIZER_NAME"
    --decoder "$decoder"
    --report-label "$report_label"
    --batch-size "${BATCH_SIZE:-$batch_size}"
    --dtype "$DTYPE"
    --min-seconds "$MIN_SECONDS"
    --max-seconds "$MAX_SECONDS"
    --cache-dir "$HF_CACHE_DIR"
    --output-dir "$run_dir"
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
    if [[ -n "$UNIGRAMS_PATH" ]]; then
      args+=(--unigrams-path "$UNIGRAMS_PATH")
    fi
  fi

  echo "Running $label on $subset with $report_label"
  printf "%q " "${args[@]}" > "$run_dir/command.txt"
  printf "\n" >> "$run_dir/command.txt"
  if "${args[@]}" > "$run_dir/run.log" 2>&1; then
    date > "$run_dir/SUCCESS"
  else
    local code=$?
    echo "$code" > "$run_dir/FAILED"
    echo "ERROR: Run failed: $run_dir" >&2
    tail -80 "$run_dir/run.log" >&2 || true
    had_failures=1
  fi
}

while IFS=$'\t' read -r subset; do
  echo ""
  echo "=== CoRal subset: $subset ==="
  while IFS=$'\t' read -r label arch checkpoint batch_size; do
    echo ""
    echo "--- Model: $label ($arch) ---"
    run_one "$label" "$arch" "$checkpoint" "$batch_size" "$subset" "greedy" "CTC no_lm" "greedy"
    run_one "$label" "$arch" "$checkpoint" "$batch_size" "$subset" "beam" "CTC LM-enabled" "beam_lm_a${ALPHA}_b${BETA}"
  done < <(python scripts/hpc/check_ctc_kenlm_eval_ready.py --manifest "$MANIFEST" --emit models)
done < <(python scripts/hpc/check_ctc_kenlm_eval_ready.py --manifest "$MANIFEST" --emit coral-subsets)

echo ""
echo "Finished: $(date)"

if [ "$had_failures" -ne 0 ]; then
  echo "One or more CoRal-style CTC + KenLM runs failed." >&2
  exit 1
fi
