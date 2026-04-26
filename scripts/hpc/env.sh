#!/bin/bash
# Common HPC environment setup — sourced by all job scripts.
# Usage: source "${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}/scripts/hpc/env.sh"

# --- Cache and scratch directories ---
export HF_HOME=/work3/$USER/hf_cache
export HF_DATASETS_CACHE=/work3/$USER/hf_cache/datasets
export FAIRSEQ2_CACHE_DIR=/work3/$USER/fairseq2_cache
export TMPDIR=/work3/$USER/tmp
export WANDB_DIR=/work3/$USER/wandb
export WANDB_DATA_DIR=/work3/$USER/wandb
export WANDB_CACHE_DIR=/work3/$USER/wandb/cache
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$TMPDIR"
mkdir -p /work3/$USER/logs/lsf
mkdir -p /work3/$USER/logs/python
mkdir -p /work3/$USER/wandb/cache

# --- Project ---
PROJECT_DIR="${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}"
cd "$PROJECT_DIR"
if [ ! -f ".venv/bin/activate" ]; then
    echo "ERROR: Python venv not found at $PROJECT_DIR/.venv" >&2
    echo "ERROR: Run 'invoke core.setup-dev' from the project root." >&2
    return 1 2>/dev/null || exit 1
fi
source .venv/bin/activate

# --- Helper: validate omnilingual-asr repo ---
setup_omniasr() {
    OMNI_ASR_DIR="/work3/$USER/omnilingual-asr"
    if [ ! -d "$OMNI_ASR_DIR/workflows" ]; then
        echo "ERROR: omnilingual-asr repo not found at $OMNI_ASR_DIR" >&2
        echo "Clone it: git clone https://github.com/facebookresearch/omnilingual-asr.git $OMNI_ASR_DIR" >&2
        exit 1
    fi
    export PYTHONPATH="$OMNI_ASR_DIR:${PYTHONPATH:-}"
}
