#!/bin/bash
#BSUB -J danish_asr_smoke
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 18GB
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 0:30
#BSUB -o /work3/s204696/logs/lsf/smoke_%J.out
#BSUB -e /work3/s204696/logs/lsf/smoke_%J.err

set -euo pipefail

# --- Environment ---
export HF_HOME=/work3/$USER/hf_cache
export HF_DATASETS_CACHE=/work3/$USER/hf_cache/datasets
export FAIRSEQ2_CACHE_DIR=/work3/$USER/fairseq2_cache
export TMPDIR=/work3/$USER/tmp
mkdir -p "$TMPDIR"
mkdir -p /work3/$USER/logs/lsf
mkdir -p /work3/$USER/logs/python

# PyTorch cu128 bundles its own CUDA runtime — no module load needed
cd "$HOME/danish_asr"
source .venv/bin/activate

echo "=== Job $LSB_JOBID: Smoke Test (50 steps) ==="
echo "Started: $(date)"
echo "Node: $(hostname)"
nvidia-smi

# Background GPU monitoring (every 30s) — trap ensures cleanup on any exit
OUTPUT_DIR="/work3/$USER/outputs"
mkdir -p "$OUTPUT_DIR"
NVIDIA_SMI_PID=""
trap '[[ -n "$NVIDIA_SMI_PID" ]] && kill "$NVIDIA_SMI_PID" 2>/dev/null || true' EXIT
nvidia-smi --query-gpu=index,timestamp,utilization.gpu,memory.total,memory.used,memory.free \
    --format=csv -l 30 > "$OUTPUT_DIR/gpu_stats_smoke_${LSB_JOBID}.csv" &
NVIDIA_SMI_PID=$!

python scripts/hpc/run_training.py \
    --config configs/fairseq2/ctc-finetune-smoke.yaml \
    --wandb-resume never \
    --wandb-tags "smoke,hpc,a100"

echo "Finished: $(date)"
