#!/bin/bash
#BSUB -J danish_asr_build_lm_corpus
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=32GB]"
#BSUB -W 12:00
#BSUB -o /work3/s204696/logs/lsf/build_lm_corpus_%J.out
#BSUB -e /work3/s204696/logs/lsf/build_lm_corpus_%J.err
#
# Build the Alexandra-proxy Danish KenLM text corpus from public Danish
# ScandiWiki + ScandiReddit text, excluding CoRal-v3 test transcripts.
#
# Usage:
#   bsub < scripts/hpc/build_lm_corpus.sh

set -euo pipefail

source "${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}/scripts/hpc/env.sh"
setup_omniasr

uv sync

LM_CONFIG="${LM_CONFIG:-configs/lm/alexandra_proxy_hpc_s204696.yaml}"

uv run python scripts/lm/build_danish_lm_corpus.py \
    --config "$LM_CONFIG"
