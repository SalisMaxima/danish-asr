#!/bin/bash
#BSUB -J vram_probe_1b_small
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 4:00
#BSUB -B
#BSUB -N
#BSUB -u s204696@dtu.dk
#BSUB -o /work3/s204696/logs/lsf/vram_probe_1b_small_%J.out
#BSUB -e /work3/s204696/logs/lsf/vram_probe_1b_small_%J.err
#
# VRAM probe: omniASR_CTC_1B_v2 at halved batch (max_num_elements=1.92M, grad_accum=8)

set -euo pipefail

source "${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}/scripts/hpc/env.sh"
setup_omniasr

CONFIG="configs/fairseq2/1b/vram-probe-1b-small.yaml"
RUN_DIR="/work3/$USER/outputs/vram_probe_1b_small_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

echo "=== VRAM Probe: 1B small batch ==="
echo "Config: $CONFIG"
echo "Output: $RUN_DIR"

python scripts/hpc/run_training.py \
    --config "$CONFIG" \
    --output-dir "$RUN_DIR" \
    --wandb-tags "vram-probe,1b-small"
