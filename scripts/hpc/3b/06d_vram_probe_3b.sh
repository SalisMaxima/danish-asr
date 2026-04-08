#!/bin/bash
#BSUB -J vram_probe_3b
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
#BSUB -o /work3/s204696/logs/lsf/vram_probe_3b_%J.out
#BSUB -e /work3/s204696/logs/lsf/vram_probe_3b_%J.err
#
# VRAM probe: omniASR_CTC_3B_v2 at conservative batch (max_num_elements=1.92M, grad_accum=8)
# Target 80GB nodes: bsub < scripts/hpc/3b/06d_vram_probe_3b.sh

set -euo pipefail

source "${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}/scripts/hpc/env.sh"
setup_omniasr

CONFIG="configs/fairseq2/3b/vram-probe-3b.yaml"
RUN_DIR="/work3/$USER/outputs/vram_probe_3b_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

echo "=== VRAM Probe: 3B conservative ==="
echo "Config: $CONFIG"
echo "Output: $RUN_DIR"

python scripts/hpc/run_training.py \
    --config "$CONFIG" \
    --output-dir "$RUN_DIR" \
    --wandb-tags "vram-probe,3b"
