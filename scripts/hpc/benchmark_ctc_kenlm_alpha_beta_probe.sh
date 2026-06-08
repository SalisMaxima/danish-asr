#!/bin/bash
#BSUB -J ctc_kenlm_ab_probe
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "span[hosts=1]"
#BSUB -R "select[gpu80gb]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 08:00
#BSUB -B
#BSUB -N
#BSUB -u s204696@dtu.dk
#BSUB -o /work3/s204696/logs/lsf/ctc_kenlm_ab_probe_%J.out
#BSUB -e /work3/s204696/logs/lsf/ctc_kenlm_ab_probe_%J.err
#
# Small alpha/beta grid for CTC beam + KenLM.
#
# This is a real BSUB script with the probe defaults defined inside the job, so
# the grid does not depend on LSF preserving shell exports from the login node.
#
# Usage:
#   bsub < scripts/hpc/benchmark_ctc_kenlm_alpha_beta_probe.sh
#
# Override example:
#   MODELS="1b_e6_50k" SPLITS="combined" MAX_SAMPLES=200 bsub < scripts/hpc/benchmark_ctc_kenlm_alpha_beta_probe.sh

set -euo pipefail

export DANISH_ASR_PROJECT_DIR="${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}"
export CTC_KENLM_MANIFEST="${CTC_KENLM_MANIFEST:-$DANISH_ASR_PROJECT_DIR/configs/eval/ctc_kenlm_finetuned_hpc.yaml}"

export OUTPUT_ROOT="${OUTPUT_ROOT:-/work3/$USER/outputs/ctc_kenlm_alpha_beta_probe}"
export MAX_SAMPLES="${MAX_SAMPLES:-500}"
export DECODERS="${DECODERS:-beam_lm}"
export ALPHAS="${ALPHAS:-0.05 0.1 0.2 0.3}"
export BETAS="${BETAS:-0.0 0.5 1.0}"
export OVERWRITE="${OVERWRITE:-true}"

# Keep the default probe small enough to finish quickly while still checking the
# two model sizes that matter most for the current presentation story.
export MODELS="${MODELS:-1b_e6_50k 3b_e6_30k}"
export SPLITS="${SPLITS:-read_aloud conversation}"

exec "$DANISH_ASR_PROJECT_DIR/scripts/hpc/benchmark_ctc_kenlm_my_method.sh"
