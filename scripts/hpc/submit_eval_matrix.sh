#!/bin/bash
# Phase 8B: Submit all 12 evaluation runs (4 models x 3 splits).
#
# PREREQUISITE: Subset TSVs must exist. Run from project root on HPC:
#   python scripts/hpc/make_subset_tsv.py --subset read_aloud
#   python scripts/hpc/make_subset_tsv.py --subset conversation
#
# Usage:
#   ./scripts/hpc/submit_eval_matrix.sh
#   ./scripts/hpc/submit_eval_matrix.sh --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}"
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# Verify subset TSVs exist
for subset in read_aloud conversation; do
    TSV="$PROJECT_DIR/data/parquet/version=0/language_distribution_${subset}.tsv"
    if [ ! -f "$TSV" ]; then
        echo "ERROR: Missing subset TSV: $TSV" >&2
        echo "Run: python scripts/hpc/make_subset_tsv.py --subset $subset" >&2
        exit 1
    fi
done

mkdir -p "/work3/$USER/logs/lsf"

SCRIPTS=(
    "$SCRIPT_DIR/300m/20_eval_base.sh"
    "$SCRIPT_DIR/300m/21_eval_e6_full.sh"
    "$SCRIPT_DIR/1b/20_eval_base_1b.sh"
    "$SCRIPT_DIR/1b/21_eval_e6_1b.sh"
)
LABELS=(
    "300M base (zero-shot)"
    "300M finetuned (E6, 50k)"
    "1B base (zero-shot)"
    "1B finetuned (E6-1B, 50k)"
)

echo "=== Phase 8B: Submitting Eval Matrix (4 jobs × 3 splits = 12 evals) ==="
echo ""

for i in "${!SCRIPTS[@]}"; do
    echo "  ${LABELS[$i]}:"
    if [ "$DRY_RUN" = true ]; then
        echo "    [dry-run] bsub < ${SCRIPTS[$i]}"
    else
        OUTPUT=$(bsub < "${SCRIPTS[$i]}" 2>&1)
        echo "    $OUTPUT"
    fi
done

echo ""
echo "Monitor with: bjobs"
