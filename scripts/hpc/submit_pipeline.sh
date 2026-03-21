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

# Ensure log dirs exist before bsub (LSF needs -o/-e paths at submission time)
LOG_DIR="/work3/$USER/logs/lsf"
mkdir -p "$LOG_DIR"
mkdir -p /work3/$USER/logs/python

PREV_JOB_ID=""

# Helper: extract and validate job ID from bsub output
extract_job_id() {
    local output="$1" label="$2"
    local job_id
    job_id=$(echo "$output" | grep -oP '(?<=<)\d+(?=>)')
    if [ -z "$job_id" ]; then
        echo "ERROR: Failed to get job ID for $label. bsub output: $output" >&2
        exit 1
    fi
    echo "$job_id"
}

# --- Step 1: Verify data ---
if [ "$SKIP_VERIFY" = false ]; then
    echo "Submitting: 01_verify_data.sh"
    VERIFY_OUTPUT=$(bsub -o "$LOG_DIR/verify_%J.out" -e "$LOG_DIR/verify_%J.err" < "$SCRIPT_DIR/01_verify_data.sh")
    VERIFY_JOB_ID=$(extract_job_id "$VERIFY_OUTPUT" "01_verify_data")
    echo "  Job ID: $VERIFY_JOB_ID"
    PREV_JOB_ID="$VERIFY_JOB_ID"
fi

# --- Step 2: Convert to fairseq2 ---
if [ "$SKIP_CONVERT" = false ]; then
    echo "Submitting: 02_convert_fairseq2.sh"
    if [ -n "$PREV_JOB_ID" ]; then
        CONVERT_OUTPUT=$(bsub -o "$LOG_DIR/convert_%J.out" -e "$LOG_DIR/convert_%J.err" -w "done($PREV_JOB_ID)" < "$SCRIPT_DIR/02_convert_fairseq2.sh")
    else
        CONVERT_OUTPUT=$(bsub -o "$LOG_DIR/convert_%J.out" -e "$LOG_DIR/convert_%J.err" < "$SCRIPT_DIR/02_convert_fairseq2.sh")
    fi
    CONVERT_JOB_ID=$(extract_job_id "$CONVERT_OUTPUT" "02_convert_fairseq2")
    echo "  Job ID: $CONVERT_JOB_ID"
    PREV_JOB_ID="$CONVERT_JOB_ID"
fi

# --- Step 3: Train ---
echo "Submitting: 03_train.sh"
if [ -n "$PREV_JOB_ID" ]; then
    TRAIN_OUTPUT=$(bsub -o "$LOG_DIR/train_%J.out" -e "$LOG_DIR/train_%J.err" -w "done($PREV_JOB_ID)" < "$SCRIPT_DIR/03_train.sh")
else
    TRAIN_OUTPUT=$(bsub -o "$LOG_DIR/train_%J.out" -e "$LOG_DIR/train_%J.err" < "$SCRIPT_DIR/03_train.sh")
fi
TRAIN_JOB_ID=$(extract_job_id "$TRAIN_OUTPUT" "03_train")
echo "  Job ID: $TRAIN_JOB_ID"

# --- Step 4: Eval (optional) ---
if [ -n "$CHECKPOINT_DIR" ]; then
    echo "Submitting: 04_eval.sh (checkpoint: $CHECKPOINT_DIR)"
    EVAL_OUTPUT=$(bsub -o "$LOG_DIR/eval_%J.out" -e "$LOG_DIR/eval_%J.err" -w "done($TRAIN_JOB_ID)" -env "CHECKPOINT_DIR=$CHECKPOINT_DIR" < "$SCRIPT_DIR/04_eval.sh")
    EVAL_JOB_ID=$(extract_job_id "$EVAL_OUTPUT" "04_eval")
    echo "  Job ID: $EVAL_JOB_ID"
fi

echo ""
echo "Pipeline submitted. Monitor with: bjobs"
