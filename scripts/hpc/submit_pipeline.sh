#!/bin/bash
# Orchestrator: chains HPC jobs via LSF dependencies.
#
# Usage:
#   ./scripts/hpc/submit_pipeline.sh
#   ./scripts/hpc/submit_pipeline.sh --skip-verify
#   ./scripts/hpc/submit_pipeline.sh --skip-verify --skip-convert
#   ./scripts/hpc/submit_pipeline.sh --checkpoint-dir /work3/$USER/outputs/omniasr_hpc_20260308

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Parse flags ---
SKIP_VERIFY=false
SKIP_CONVERT=false
CHECKPOINT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-verify) SKIP_VERIFY=true; shift ;;
        --skip-convert) SKIP_CONVERT=true; shift ;;
        --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# Ensure log dirs exist
mkdir -p /work3/$USER/logs/lsf
mkdir -p /work3/$USER/logs/python

PREV_JOB_ID=""

# --- Step 1: Verify data ---
if [ "$SKIP_VERIFY" = false ]; then
    echo "Submitting: 01_verify_data.sh"
    VERIFY_OUTPUT=$(bsub < "$SCRIPT_DIR/01_verify_data.sh")
    VERIFY_JOB_ID=$(echo "$VERIFY_OUTPUT" | grep -oP '(?<=<)\d+(?=>)')
    echo "  Job ID: $VERIFY_JOB_ID"
    PREV_JOB_ID="$VERIFY_JOB_ID"
fi

# --- Step 2: Convert to fairseq2 ---
if [ "$SKIP_CONVERT" = false ]; then
    echo "Submitting: 02_convert_fairseq2.sh"
    if [ -n "$PREV_JOB_ID" ]; then
        CONVERT_OUTPUT=$(bsub -w "done($PREV_JOB_ID)" < "$SCRIPT_DIR/02_convert_fairseq2.sh")
    else
        CONVERT_OUTPUT=$(bsub < "$SCRIPT_DIR/02_convert_fairseq2.sh")
    fi
    CONVERT_JOB_ID=$(echo "$CONVERT_OUTPUT" | grep -oP '(?<=<)\d+(?=>)')
    echo "  Job ID: $CONVERT_JOB_ID"
    PREV_JOB_ID="$CONVERT_JOB_ID"
fi

# --- Step 3: Train ---
echo "Submitting: 03_train.sh"
if [ -n "$PREV_JOB_ID" ]; then
    TRAIN_OUTPUT=$(bsub -w "done($PREV_JOB_ID)" < "$SCRIPT_DIR/03_train.sh")
else
    TRAIN_OUTPUT=$(bsub < "$SCRIPT_DIR/03_train.sh")
fi
TRAIN_JOB_ID=$(echo "$TRAIN_OUTPUT" | grep -oP '(?<=<)\d+(?=>)')
echo "  Job ID: $TRAIN_JOB_ID"

# --- Step 4: Eval (optional) ---
if [ -n "$CHECKPOINT_DIR" ]; then
    echo "Submitting: 04_eval.sh (checkpoint: $CHECKPOINT_DIR)"
    EVAL_OUTPUT=$(bsub -w "done($TRAIN_JOB_ID)" -env "CHECKPOINT_DIR=$CHECKPOINT_DIR" < "$SCRIPT_DIR/04_eval.sh")
    EVAL_JOB_ID=$(echo "$EVAL_OUTPUT" | grep -oP '(?<=<)\d+(?=>)')
    echo "  Job ID: $EVAL_JOB_ID"
fi

echo ""
echo "Pipeline submitted. Monitor with: bjobs"
