#!/bin/bash
# Phase 8B: Submit all 12 evaluation runs (4 models x 3 splits).
#
# No prerequisites — subset filtering uses fairseq2's "<split>_<corpus>" valid_split format.
#
# Usage:
#   ./scripts/hpc/submit_eval_matrix.sh
#   ./scripts/hpc/submit_eval_matrix.sh --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

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
