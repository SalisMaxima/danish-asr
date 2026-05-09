#!/usr/bin/env bash
# Backward-compatible wrapper for the unified CTC + beam + KenLM submitter.
#
# Usage:
#   bash scripts/hpc/submit_coral_ctc_kenlm_eval.sh smoke
#   bash scripts/hpc/submit_coral_ctc_kenlm_eval.sh full
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-smoke}" in
  smoke) exec "$SCRIPT_DIR/submit_ctc_kenlm_eval.sh" coral-smoke ;;
  full) exec "$SCRIPT_DIR/submit_ctc_kenlm_eval.sh" coral-full ;;
  *) echo "Usage: $0 [smoke|full]" >&2; exit 2 ;;
esac
