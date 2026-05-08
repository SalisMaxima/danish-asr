#!/usr/bin/env bash
# Submit the CoRal-style CTC benchmark with greedy and beam+KenLM decoding.
#
# Usage:
#   bash scripts/hpc/submit_coral_ctc_kenlm_eval.sh smoke
#   bash scripts/hpc/submit_coral_ctc_kenlm_eval.sh full
#
# Optional overrides:
#   KENLM_BINARY=/work3/$USER/artifacts/lm/danish_lm_v1_3gram.bin
#   BEAM_WIDTH=64
#   ALPHA=0.6
#   BETA=0.5

set -euo pipefail

MODE="${1:-smoke}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
JOB_SCRIPT="$PROJECT_DIR/scripts/hpc/benchmark_coral_style_alexandra_matrix.sh"

mkdir -p "/work3/$USER/logs/lsf"

export KENLM_BINARY="${KENLM_BINARY:-/work3/$USER/artifacts/lm/danish_lm_v1_3gram.bin}"
export BEAM_WIDTH="${BEAM_WIDTH:-64}"
export ALPHA="${ALPHA:-0.6}"
export BETA="${BETA:-0.5}"
export DANISH_ASR_PROJECT_DIR="${DANISH_ASR_PROJECT_DIR:-$PROJECT_DIR}"

if [[ ! -f "$KENLM_BINARY" ]]; then
  echo "KENLM_BINARY not found: $KENLM_BINARY" >&2
  echo "Build it first:" >&2
  echo "  bsub < scripts/hpc/build_lm_corpus.sh" >&2
  echo "  bsub < scripts/hpc/build_kenlm.sh" >&2
  exit 1
fi

case "$MODE" in
  smoke)
    export MAX_SAMPLES="${MAX_SAMPLES:-5}"
    ;;
  full)
    unset MAX_SAMPLES
    ;;
  *)
    echo "Usage: $0 [smoke|full]" >&2
    exit 2
    ;;
esac

echo "Submitting CoRal-style CTC benchmark ($MODE)"
echo "Project dir:   $PROJECT_DIR"
echo "KenLM binary:  $KENLM_BINARY"
echo "Beam width:    $BEAM_WIDTH"
echo "Alpha/Beta:    $ALPHA / $BETA"
echo "Max samples:   ${MAX_SAMPLES:-full}"

bsub < "$JOB_SCRIPT"
