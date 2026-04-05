#!/bin/bash
#BSUB -J danish_asr_vram_probe
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 4:00
#BSUB -B
#BSUB -N
#BSUB -u s204696@dtu.dk
#BSUB -o /work3/s204696/logs/lsf/vram_probe_%J.out
#BSUB -e /work3/s204696/logs/lsf/vram_probe_%J.err
#
# VRAM probe: test whether a given OmniASR CTC v2 model size fits on A100-40GB.
# Runs only 50 training steps — enough to confirm fit or trigger OOM.
#
# Usage:
#   # 1B with default batch size
#   PROBE_CONFIG=configs/fairseq2/vram-probe-1b.yaml bsub < scripts/hpc/06_vram_probe.sh
#
#   # 1B with halved batch (fallback if above OOMs)
#   PROBE_CONFIG=configs/fairseq2/vram-probe-1b-small.yaml bsub < scripts/hpc/06_vram_probe.sh
#
#   # 3B conservative
#   PROBE_CONFIG=configs/fairseq2/vram-probe-3b.yaml bsub < scripts/hpc/06_vram_probe.sh

set -euo pipefail

# --- Environment ---
source "${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}/scripts/hpc/env.sh"
setup_omniasr

# --- Config ---
CONFIG="${PROBE_CONFIG:?ERROR: Set PROBE_CONFIG to a vram-probe yaml, e.g. configs/fairseq2/vram-probe-1b.yaml}"

# Use a unique output dir per run to prevent fairseq2 checkpoint resume.
# VRAM probes must always start fresh — resuming would skip the memory-heavy
# model init and first forward pass, defeating the purpose.
CONFIG_BASENAME="$(basename "$CONFIG" .yaml)"
RUN_DIR="/work3/$USER/outputs/vram_probe_${CONFIG_BASENAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

echo "=== VRAM Probe ==="
echo "Config: $CONFIG"
echo "Output: $RUN_DIR"
echo "=================="

python scripts/hpc/run_training.py \
    --config "$CONFIG" \
    --output-dir "$RUN_DIR" \
    --wandb-tags "vram-probe,${CONFIG_BASENAME}"
