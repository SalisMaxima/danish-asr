#!/bin/bash
#BSUB -J vram_probe_3b_tiny
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -R "select[gpu80gb]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 4:00
#BSUB -B
#BSUB -N
#BSUB -u s204696@dtu.dk
#BSUB -o /work3/s204696/logs/lsf/vram_probe_3b_tiny_%J.out
#BSUB -e /work3/s204696/logs/lsf/vram_probe_3b_tiny_%J.err
#
# VRAM probe: omniASR_CTC_3B_v2 at minimal batch (max_num_elements=960K, grad_accum=16)
# Target 80GB nodes: bsub -m "n-62-18-8 n-62-18-9 n-62-18-10 n-62-18-11 n-62-18-12" < scripts/hpc/06e_vram_probe_3b_tiny.sh

set -euo pipefail

source "${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}/scripts/hpc/env.sh"
setup_omniasr

CONFIG="configs/fairseq2/vram-probe-3b-tiny.yaml"
RUN_DIR="/work3/$USER/outputs/vram_probe_3b_tiny_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

echo "=== VRAM Probe: 3B tiny batch ==="
echo "Config: $CONFIG"
echo "Output: $RUN_DIR"

python scripts/hpc/run_training.py \
    --config "$CONFIG" \
    --output-dir "$RUN_DIR" \
    --wandb-tags "vram-probe,3b-tiny"
