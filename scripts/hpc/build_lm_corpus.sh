#!/bin/bash
#BSUB -J danish_asr_build_lm_corpus
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 02:00
#BSUB -o /work3/s204696/logs/lsf/build_lm_corpus_%J.out
#BSUB -e /work3/s204696/logs/lsf/build_lm_corpus_%J.err
#
# Build the iteration-1 Danish KenLM text corpus from CoRal v3 train transcripts
# already converted into local fairseq2 parquet shards on work3.
#
# Usage:
#   bsub < scripts/hpc/build_lm_corpus.sh

set -euo pipefail

source "${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}/scripts/hpc/env.sh"
setup_omniasr

uv sync

uv run python scripts/lm/build_danish_lm_corpus.py \
    --config configs/lm/coral_train_only_hpc_s204696.yaml
