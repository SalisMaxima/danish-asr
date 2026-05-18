#!/bin/bash
#BSUB -J ctc_smoke_greedy
#BSUB -q gpua100
#BSUB -n 1
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -R "select[gpu80gb]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:30
#BSUB -B
#BSUB -N
#BSUB -u s204696@dtu.dk
#BSUB -o /work3/s204696/logs/lsf/ctc_smoke_greedy_%J.out
#BSUB -e /work3/s204696/logs/lsf/ctc_smoke_greedy_%J.err

set -euo pipefail

export SMOKE_DECODERS="greedy"
export DANISH_ASR_PROJECT_DIR="${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}"

bash "$DANISH_ASR_PROJECT_DIR/scripts/hpc/ctc_kenlm_smoke_common.sh"
