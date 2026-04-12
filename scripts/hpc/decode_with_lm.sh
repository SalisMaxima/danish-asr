#!/bin/bash
#BSUB -J danish_asr_decode_lm
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -R "select[gpu80gb]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00
#BSUB -o /work3/s204696/logs/lsf/decode_lm_%J.out
#BSUB -e /work3/s204696/logs/lsf/decode_lm_%J.err

# Standalone LM-decoding grid for OmniASR CTC checkpoints.
#
# Usage example:
#   KENLM_BINARY=/work3/$USER/artifacts/lm/danish_lm_v1_3gram.bin \
#   bsub < scripts/hpc/decode_with_lm.sh

set -euo pipefail

source "${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}/scripts/hpc/env.sh"
setup_omniasr

PROJECT_DIR="${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}"
OUTPUT_ROOT="${LM_DECODE_OUTPUT_DIR:-/work3/$USER/outputs/lm_decode_3b_e6_combined}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/fairseq2/3b/ctc-eval-e6-3b.yaml}"
KENLM_BINARY="${KENLM_BINARY:-}"
TOKENIZER_MODEL_PATH="${TOKENIZER_MODEL_PATH:-}"
BEAM_WIDTH="${BEAM_WIDTH:-64}"
ALPHAS="${ALPHAS:-0.3 0.6 0.9}"
BETAS="${BETAS:-0.0 0.5 1.0}"
BATCH_SIZE="${BATCH_SIZE:-2}"

mkdir -p "$OUTPUT_ROOT"

TOKENIZER_ARGS=()
if [ -n "$TOKENIZER_MODEL_PATH" ]; then
    TOKENIZER_ARGS+=(--tokenizer-model-path "$TOKENIZER_MODEL_PATH")
fi

python "$PROJECT_DIR/scripts/decode_ctc_with_lm.py" \
    --eval-config "$EVAL_CONFIG" \
    --decoder greedy \
    --batch-size "$BATCH_SIZE" \
    --output-dir "$OUTPUT_ROOT/greedy"

python "$PROJECT_DIR/scripts/evaluate_ctc_predictions.py" \
    --predictions "$OUTPUT_ROOT/greedy/predictions.txt" \
    --references "$OUTPUT_ROOT/greedy/references.txt" \
    --output-json "$OUTPUT_ROOT/greedy/score.json"

python "$PROJECT_DIR/scripts/decode_ctc_with_lm.py" \
    --eval-config "$EVAL_CONFIG" \
    --decoder beam \
    --beam-width "$BEAM_WIDTH" \
    --batch-size "$BATCH_SIZE" \
    "${TOKENIZER_ARGS[@]}" \
    --output-dir "$OUTPUT_ROOT/beam_no_lm"

python "$PROJECT_DIR/scripts/evaluate_ctc_predictions.py" \
    --predictions "$OUTPUT_ROOT/beam_no_lm/predictions.txt" \
    --references "$OUTPUT_ROOT/beam_no_lm/references.txt" \
    --output-json "$OUTPUT_ROOT/beam_no_lm/score.json"

if [ -z "$KENLM_BINARY" ]; then
    echo "KENLM_BINARY is unset; skipping LM-backed decoding grid."
    exit 0
fi

for ALPHA in $ALPHAS; do
    for BETA in $BETAS; do
        RUN_DIR="$OUTPUT_ROOT/beam_lm_a${ALPHA}_b${BETA}"

        python "$PROJECT_DIR/scripts/decode_ctc_with_lm.py" \
            --eval-config "$EVAL_CONFIG" \
            --decoder beam \
            --kenlm-binary "$KENLM_BINARY" \
            --beam-width "$BEAM_WIDTH" \
            --alpha "$ALPHA" \
            --beta "$BETA" \
            --batch-size "$BATCH_SIZE" \
            "${TOKENIZER_ARGS[@]}" \
            --output-dir "$RUN_DIR"

        python "$PROJECT_DIR/scripts/evaluate_ctc_predictions.py" \
            --predictions "$RUN_DIR/predictions.txt" \
            --references "$RUN_DIR/references.txt" \
            --output-json "$RUN_DIR/score.json"
    done
done
