#!/usr/bin/env bash
# Submit CTC + beam + KenLM evaluation jobs from a DTU HPC login node.
#
# This helper is not itself a BSUB job script. It validates the environment on
# the login node, then submits the real BSUB job scripts with `bsub < ...`.
#
# Usage:
#   bash scripts/hpc/submit_ctc_kenlm_eval.sh my-smoke
#   bash scripts/hpc/submit_ctc_kenlm_eval.sh coral-smoke
#   bash scripts/hpc/submit_ctc_kenlm_eval.sh all-full
#
# Direct BSUB alternatives:
#   MAX_SAMPLES=5 bsub < scripts/hpc/benchmark_ctc_kenlm_my_method.sh
#   MAX_SAMPLES=5 bsub < scripts/hpc/benchmark_coral_style_alexandra_matrix.sh

set -euo pipefail

MODE="${1:-all-smoke}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MANIFEST="${CTC_KENLM_MANIFEST:-$PROJECT_DIR/configs/eval/ctc_kenlm_finetuned_hpc.yaml}"

if [[ -n "${LSB_JOBID:-}" ]]; then
  echo "ERROR: submit_ctc_kenlm_eval.sh should be run on a login node, not submitted with bsub." >&2
  echo "Submit the actual job scripts directly instead:" >&2
  echo "  bsub < scripts/hpc/benchmark_ctc_kenlm_my_method.sh" >&2
  echo "  bsub < scripts/hpc/benchmark_coral_style_alexandra_matrix.sh" >&2
  exit 2
fi

if ! command -v bsub >/dev/null 2>&1; then
  echo "ERROR: bsub not found. Run this on the HPC login node." >&2
  exit 127
fi

export DANISH_ASR_PROJECT_DIR="${DANISH_ASR_PROJECT_DIR:-$PROJECT_DIR}"
export CTC_KENLM_MANIFEST="$MANIFEST"

mkdir -p "/work3/$USER/logs/lsf"

case "$MODE" in
  my-smoke)
    export MAX_SAMPLES="${MAX_SAMPLES:-5}"
    methods=(my)
    ;;
  my-full)
    unset MAX_SAMPLES
    methods=(my)
    ;;
  coral-smoke)
    export MAX_SAMPLES="${MAX_SAMPLES:-5}"
    methods=(coral)
    ;;
  coral-full)
    unset MAX_SAMPLES
    methods=(coral)
    ;;
  all-smoke)
    export MAX_SAMPLES="${MAX_SAMPLES:-5}"
    methods=(my coral)
    ;;
  all-full)
    unset MAX_SAMPLES
    methods=(my coral)
    ;;
  *)
    echo "Usage: $0 [my-smoke|my-full|coral-smoke|coral-full|all-smoke|all-full]" >&2
    exit 2
    ;;
esac

source "$PROJECT_DIR/scripts/hpc/env.sh"
setup_omniasr
cd "$PROJECT_DIR"

for method in "${methods[@]}"; do
  python scripts/hpc/check_ctc_kenlm_eval_ready.py \
    --manifest "$MANIFEST" \
    --method "$method" \
    --skip-cuda \
    --print-quota
done

echo "Submitting CTC + beam + KenLM eval mode: $MODE"
echo "Project dir: $PROJECT_DIR"
echo "Manifest:    $MANIFEST"
echo "Max samples: ${MAX_SAMPLES:-full}"

for method in "${methods[@]}"; do
  case "$method" in
    my)
      echo "Submitting my-methodology matrix..."
      bsub < "$PROJECT_DIR/scripts/hpc/benchmark_ctc_kenlm_my_method.sh"
      ;;
    coral)
      echo "Submitting CoRal/Alexandra matrix..."
      bsub < "$PROJECT_DIR/scripts/hpc/benchmark_coral_style_alexandra_matrix.sh"
      ;;
  esac
done

echo "Monitor with: bstat"
echo "Collect when done with: uv run python scripts/hpc/collect_ctc_kenlm_results.py"
