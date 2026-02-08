# DTU HPC Setup Guide

How to run the omnilingual ASR finetuning pipeline on DTU's HPC cluster.

References:
- [GPU nodes](https://www.hpc.dtu.dk/?page_id=2129)
- [Using GPUs under LSF10](https://www.hpc.dtu.dk/?page_id=2759)
- [DTU MLOps HPC guide](https://skaftenicki.github.io/dtu_mlops/s10_extra/high_performance_clusters/)

## Available GPU Resources

| GPU | VRAM | Nodes | GPUs/Node | LSF Queue |
|---|---|---|---|---|
| L40s PCIe | 48 GB | 6 | 2 | `gpul40s` |
| A100 PCIe (80GB) | 80 GB | 6 | 2 | `gpua100` |
| A100 PCIe (40GB) | 40 GB | 4 | 2 | `gpua100` |
| V100 | 16 GB | 6 | 2 | `gpuv100` |
| V100 (32GB) | 32 GB | 8 | 2 | `gpuv100` |
| V100 NVLink (32GB) | 32 GB | 3 | 4 | `gpuv100` |
| H100 PCIe | 80 GB | — | — | — |
| H100 SXM5 | 80 GB | — | — | — |

**Recommended for our project:** `gpua100` queue (40GB or 80GB A100). Alternatively `gpul40s` (48GB L40s).

**Walltime limit:** 24 hours per job.

## Cluster Access

```bash
# SSH into login node
ssh <dtu-username>@login.hpc.dtu.dk

# Or use ThinLinc for graphical access
```

## Environment Setup on HPC

### Option A: Miniconda (traditional)

```bash
# Install Miniconda (one-time)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# Create environment
conda create -n danish_asr python=3.12 -y
conda activate danish_asr

# Install our package
pip install -e ".[dev]"
pip install "omnilingual-asr[data]"
```

### Option B: uv (faster, matches local dev)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repo and setup
git clone <repo-url> ~/danish_asr
cd ~/danish_asr
uv sync
uv add "omnilingual-asr[data]"
```

### CUDA Module

```bash
# Check available CUDA versions
module avail cuda

# Load CUDA (A100 requires >=11.0)
module load cuda/12.1  # or latest available
```

## Job Scripts

### Single GPU Training

`scripts/hpc/train_single_gpu.sh`:

```bash
#!/bin/bash
#BSUB -J danish_asr_train
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -o logs/train_%J.out
#BSUB -e logs/train_%J.err

module load cuda/12.1

# Activate environment
source ~/miniconda3/bin/activate danish_asr
# Or: source ~/danish_asr/.venv/bin/activate

export OUTPUT_DIR="$HOME/danish_asr/outputs/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

python -m workflows.recipes.wav2vec2.asr $OUTPUT_DIR \
    --config-file configs/omniasr/ctc-finetune-danish.yaml

echo "Training complete. Output: $OUTPUT_DIR"
```

Submit: `bsub < scripts/hpc/train_single_gpu.sh`

### Dual GPU Training

`scripts/hpc/train_dual_gpu.sh`:

```bash
#!/bin/bash
#BSUB -J danish_asr_train_2gpu
#BSUB -q gpua100
#BSUB -n 8
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -o logs/train_%J.out
#BSUB -e logs/train_%J.err

module load cuda/12.1

source ~/miniconda3/bin/activate danish_asr

export OUTPUT_DIR="$HOME/danish_asr/outputs/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

# fairseq2 uses torch.distributed internally
python -m workflows.recipes.wav2vec2.asr $OUTPUT_DIR \
    --config-file configs/omniasr/ctc-finetune-danish-2gpu.yaml

echo "Training complete. Output: $OUTPUT_DIR"
```

### Data Conversion Job

`scripts/hpc/convert_data.sh`:

```bash
#!/bin/bash
#BSUB -J coral_convert
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 12:00
#BSUB -o logs/convert_%J.out
#BSUB -e logs/convert_%J.err

source ~/miniconda3/bin/activate danish_asr

python scripts/convert_coral_to_parquet.py \
    --output-dir data/parquet \
    --subset read_aloud \
    --num-workers 8

echo "Conversion complete."
```

### Evaluation Job

`scripts/hpc/evaluate.sh`:

```bash
#!/bin/bash
#BSUB -J danish_asr_eval
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 4:00
#BSUB -o logs/eval_%J.out
#BSUB -e logs/eval_%J.err

module load cuda/12.1
source ~/miniconda3/bin/activate danish_asr

CHECKPOINT_DIR="$1"  # pass as argument

python -m workflows.recipes.wav2vec2.asr.eval.recipe $CHECKPOINT_DIR \
    --config-file configs/omniasr/ctc-finetune-danish.yaml

echo "Evaluation complete."
```

Submit: `bsub < scripts/hpc/evaluate.sh`

## LSF Cheat Sheet

| Command | Description |
|---|---|
| `bsub < script.sh` | Submit job |
| `bjobs` | List your running jobs |
| `bjobs -l <jobid>` | Detailed job info |
| `bkill <jobid>` | Kill a job |
| `bqueues` | List available queues |
| `bstat` | Cluster statistics |
| `nodestat -F gpu` | GPU node availability |

## Resource Requests Explained

```bash
#BSUB -n 4                              # 4 CPU cores (use 4 per GPU)
#BSUB -R "rusage[mem=16GB]"             # 16GB RAM per core
#BSUB -R "span[hosts=1]"               # All resources on one node
#BSUB -gpu "num=1:mode=exclusive_process"  # 1 GPU, exclusive access
#BSUB -q gpua100                        # A100 queue
#BSUB -W 24:00                          # 24-hour walltime (maximum)
```

For 80GB A100 specifically:
```bash
#BSUB -R "select[gpu80gb]"
```

For 32GB V100:
```bash
#BSUB -q gpuv100
#BSUB -R "select[gpu32gb]"
```

## Storage Considerations

| Location | Size | Speed | Use for |
|---|---|---|---|
| `$HOME` | Limited (~10-50GB) | Slow | Code, configs, small files |
| `/work3/<username>` | Large (TB-scale) | Fast | Datasets, Parquet files, checkpoints |
| `/tmp` on compute node | Limited | Very fast | Temporary during training |

Recommended layout:
```
$HOME/danish_asr/              # Code repository
/work3/<user>/danish_asr/
├── data/parquet/              # Converted Parquet dataset
├── hf_cache/                  # HuggingFace cache (raw downloads)
├── fairseq2_cache/            # Model checkpoints cache
└── outputs/                   # Training outputs
```

Set cache paths before training:
```bash
export HF_HOME=/work3/<user>/danish_asr/hf_cache
export FAIRSEQ2_CACHE_DIR=/work3/<user>/danish_asr/fairseq2_cache
```

## Workflow Summary

1. **First time:** Set up environment + download model checkpoint
2. **Data conversion:** Submit CPU job to convert CoRal-v2 → Parquet (8-12h)
3. **Smoke test:** Interactive GPU session to verify training starts
4. **Training:** Submit GPU job (A100, 24h walltime)
5. **Evaluation:** Submit eval job after training completes
6. **Iterate:** Adjust hyperparameters based on validation loss/WER

### Interactive GPU Session (smoke test)

```bash
# Request interactive A100 session
linuxsh -s -l nodes=1:ppn=4:gpus=1 -q gpua100 -W 1:00

# Or use the dedicated interactive node
ssh a100sh
```
