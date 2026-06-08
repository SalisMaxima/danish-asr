#!/bin/bash
#BSUB -J ctc_kenlm_my_method
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
#BSUB -o /work3/s204696/logs/lsf/ctc_kenlm_my_method_%J.out
#BSUB -e /work3/s204696/logs/lsf/ctc_kenlm_my_method_%J.err

set -euo pipefail

export DANISH_ASR_PROJECT_DIR="${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}"
MANIFEST="${CTC_KENLM_MANIFEST:-$DANISH_ASR_PROJECT_DIR/configs/eval/ctc_kenlm_finetuned_hpc.yaml}"
DTYPE="${DTYPE:-bfloat16}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
OVERWRITE="${OVERWRITE:-false}"
DECODERS="${DECODERS:-greedy beam_no_lm beam_lm}"
ALPHAS="${ALPHAS:-}"
BETAS="${BETAS:-}"
BEAM_WIDTH="${BEAM_WIDTH:-}"
TOKENIZER_MODEL_PATH="${TOKENIZER_MODEL_PATH:-}"
UNIGRAMS_PATH="${UNIGRAMS_PATH:-}"
MODELS="${MODELS:-}"
SPLITS="${SPLITS:-}"

source "$DANISH_ASR_PROJECT_DIR/scripts/hpc/env.sh"
setup_omniasr
cd "$DANISH_ASR_PROJECT_DIR"

eval "$(python scripts/hpc/check_ctc_kenlm_eval_ready.py --manifest "$MANIFEST" --emit shell)"

KENLM_BINARY="${KENLM_BINARY:-$CTC_KENLM_BINARY}"
DATASET_ROOT="${DATASET_ROOT:-$CTC_PARQUET_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$CTC_OUTPUT_ROOT_MY}"
TOKENIZER_NAME="${TOKENIZER_NAME:-$CTC_TOKENIZER_NAME}"
TOKENIZER_MODEL_PATH="${TOKENIZER_MODEL_PATH:-$CTC_TOKENIZER_MODEL_PATH}"
BEAM_WIDTH="${BEAM_WIDTH:-$CTC_BEAM_WIDTH}"
ALPHAS="${ALPHAS:-$CTC_ALPHA}"
BETAS="${BETAS:-$CTC_BETA}"

python scripts/hpc/check_ctc_kenlm_eval_ready.py \
  --manifest "$MANIFEST" \
  --method my \
  --print-quota

if ! mkdir -p "$OUTPUT_ROOT" 2>/dev/null; then
  echo "ERROR: Cannot create eval workspace: $OUTPUT_ROOT" >&2
  echo "ERROR: Check /work3 quota with getquota_work3.sh" >&2
  exit 1
fi

echo "=== CTC + beam + KenLM my-method matrix ==="
echo "Output root: $OUTPUT_ROOT"
echo "Dataset root: $DATASET_ROOT"
echo "KenLM binary: $KENLM_BINARY"
echo "Unigrams: ${UNIGRAMS_PATH:-none}"
echo "Decoders: $DECODERS"
echo "Alphas: ${ALPHAS:-none}"
echo "Betas: ${BETAS:-none}"
echo "Max samples: ${MAX_SAMPLES:-full}"
echo "Overwrite: $OVERWRITE"
echo "Model filter: ${MODELS:-all}"
echo "Split filter: ${SPLITS:-all}"
echo "Started: $(date)"
echo "Node: $(hostname)"
nvidia-smi

echo "=== Dependency versions ==="
python -c "
import sys, importlib
print('Python:', sys.version)
for pkg in ['torch', 'fairseq2', 'omnilingual_asr', 'pyctcdecode', 'kenlm', 'pyarrow']:
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

has_decoder() {
  local wanted="$1"
  [[ " $DECODERS " == *" $wanted "* ]]
}

selected() {
  local choices="$1"
  local wanted="$2"

  [[ -z "$choices" || " $choices " == *" $wanted "* ]]
}

run_command() {
  local run_dir="$1"
  shift

  if [[ -f "$run_dir/SUCCESS" && "$OVERWRITE" != "true" ]]; then
    echo "Skipping completed run: $run_dir"
    return 0
  fi

  mkdir -p "$run_dir"
  rm -f "$run_dir/SUCCESS" "$run_dir/FAILED"
  printf "%q " "$@" > "$run_dir/command.txt"
  printf "\n" >> "$run_dir/command.txt"

  if "$@" > "$run_dir/run.log" 2>&1; then
    date > "$run_dir/SUCCESS"
  else
    local code=$?
    echo "$code" > "$run_dir/FAILED"
    echo "ERROR: Run failed: $run_dir" >&2
    tail -80 "$run_dir/run.log" >&2 || true
    had_failures=1
  fi
}

if has_decoder "beam_lm" && { [[ -z "$ALPHAS" ]] || [[ -z "$BETAS" ]]; }; then
  echo "ERROR: beam_lm requested but ALPHAS or BETAS is empty." >&2
  exit 2
fi

tokenizer_args=()
if [[ -n "$TOKENIZER_MODEL_PATH" ]]; then
  tokenizer_args+=(--tokenizer-model-path "$TOKENIZER_MODEL_PATH")
fi
unigram_args=()
if [[ -n "$UNIGRAMS_PATH" ]]; then
  unigram_args+=(--unigrams-path "$UNIGRAMS_PATH")
fi

while IFS=$'\t' read -r model_label model_arch checkpoint_path default_batch_size; do
  if ! selected "$MODELS" "$model_label"; then
    echo "Skipping model filtered out by MODELS: $model_label"
    continue
  fi

  batch_size="${BATCH_SIZE:-$default_batch_size}"
  echo ""
  echo "=== Model: $model_label ($model_arch) ==="

  while IFS=$'\t' read -r split_label dataset_split; do
    if ! selected "$SPLITS" "$split_label"; then
      echo "Skipping split filtered out by SPLITS: $split_label"
      continue
    fi

    echo ""
    echo "--- Split: $split_label ($dataset_split) ---"
    base_args=(
      python scripts/decode_ctc_with_lm.py
      --checkpoint-path "$checkpoint_path"
      --model-arch "$model_arch"
      --dataset-root "$DATASET_ROOT"
      --dataset-split "$dataset_split"
      --tokenizer-name "$TOKENIZER_NAME"
      --batch-size "$batch_size"
      --dtype "$DTYPE"
      "${tokenizer_args[@]}"
    )
    if [[ -n "$MAX_SAMPLES" ]]; then
      base_args+=(--max-samples "$MAX_SAMPLES")
    fi

    if has_decoder "greedy"; then
      run_command "$OUTPUT_ROOT/$model_label/$split_label/greedy" \
        "${base_args[@]}" \
        --decoder greedy \
        --output-dir "$OUTPUT_ROOT/$model_label/$split_label/greedy"
    fi

    if has_decoder "beam_no_lm"; then
      run_command "$OUTPUT_ROOT/$model_label/$split_label/beam_no_lm" \
        "${base_args[@]}" \
        --decoder beam \
        --beam-width "$BEAM_WIDTH" \
        --output-dir "$OUTPUT_ROOT/$model_label/$split_label/beam_no_lm"
    fi

    if has_decoder "beam_lm"; then
      for alpha in $ALPHAS; do
        for beta in $BETAS; do
          run_dir="$OUTPUT_ROOT/$model_label/$split_label/beam_lm_a${alpha}_b${beta}"
          run_command "$run_dir" \
            "${base_args[@]}" \
            --decoder beam \
            --kenlm-binary "$KENLM_BINARY" \
            "${unigram_args[@]}" \
            --beam-width "$BEAM_WIDTH" \
            --alpha "$alpha" \
            --beta "$beta" \
            --output-dir "$run_dir"
        done
      done
    fi
  done < <(python scripts/hpc/check_ctc_kenlm_eval_ready.py --manifest "$MANIFEST" --emit my-splits)
done < <(python scripts/hpc/check_ctc_kenlm_eval_ready.py --manifest "$MANIFEST" --emit models)

echo ""
echo "Finished: $(date)"

if [ "$had_failures" -ne 0 ]; then
  echo "One or more CTC + KenLM my-method runs failed." >&2
  exit 1
fi
