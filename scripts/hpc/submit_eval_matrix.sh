#!/bin/bash
# Submit the evaluation matrix:
# - 300M base: 3 configs
# - 300M E6: 3 configs
# - 1B base: 3 configs
# - 1B E6: 3 configs
# - 3B base: 3 configs
# - 3B E6-3B: 3 configs
#
# Total: 18 eval runs submitted across 6 batch scripts.
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
    "$SCRIPT_DIR/3b/19_eval_base_3b.sh"
    "$SCRIPT_DIR/3b/20_eval_e6_3b.sh"
)
LABELS=(
    "300M base (zero-shot)"
    "300M finetuned (E6, 50k)"
    "1B base (zero-shot)"
    "1B finetuned (E6-1B, 50k)"
    "3B base (zero-shot)"
    "3B finetuned (E6-3B, 30k)"
)

echo "=== Submitting Eval Matrix (6 jobs, 18 eval runs total) ==="
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
