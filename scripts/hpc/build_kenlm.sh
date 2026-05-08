#!/bin/bash
#BSUB -J danish_asr_build_kenlm
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 12:00
#BSUB -o /work3/s204696/logs/lsf/build_kenlm_%J.out
#BSUB -e /work3/s204696/logs/lsf/build_kenlm_%J.err
#
# Build the Alexandra-proxy 3-gram KenLM artifact from the prepared Danish LM
# text corpus on work3.
#
# Usage:
#   bsub < scripts/hpc/build_kenlm.sh
#
# Optional overrides:
#   LM_CONFIG=configs/lm/alexandra_proxy_hpc_s204696.yaml
#   LM_TEXT_PATH=/work3/$USER/data/lm/processed/danish_lm_alexandra_proxy.txt
#   KENLM_OUTPUT_PREFIX=/work3/$USER/artifacts/lm/danish_lm_alexandra_proxy_3gram
#   LMPLZ_BIN=/path/to/lmplz
#   BUILD_BINARY_BIN=/path/to/build_binary

set -euo pipefail

source "${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}/scripts/hpc/env.sh"
setup_omniasr

LM_CONFIG="${LM_CONFIG:-configs/lm/alexandra_proxy_hpc_s204696.yaml}"
LM_TEXT_PATH="${LM_TEXT_PATH:-/work3/$USER/data/lm/processed/danish_lm_alexandra_proxy.txt}"
KENLM_OUTPUT_PREFIX="${KENLM_OUTPUT_PREFIX:-/work3/$USER/artifacts/lm/danish_lm_alexandra_proxy_3gram}"
LMPLZ_BIN="${LMPLZ_BIN:-lmplz}"
BUILD_BINARY_BIN="${BUILD_BINARY_BIN:-build_binary}"

uv sync

uv run python scripts/lm/build_kenlm.py \
    --config "$LM_CONFIG" \
    --text-path "$LM_TEXT_PATH" \
    --output-prefix "$KENLM_OUTPUT_PREFIX" \
    --lmplz-bin "$LMPLZ_BIN" \
    --build-binary-bin "$BUILD_BINARY_BIN"
