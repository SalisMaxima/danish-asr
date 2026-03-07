# DTU HPC — Python Environment Setup

## Module system

DTU HPC uses **Environment Modules** to manage software versions.

```bash
module list                          # show currently loaded modules
module avail                         # list all available modules
module avail -t                      # terse listing (one per line, grep-friendly)
module avail cuda                    # filter by name
module load <module/version>         # load a module
module unload <module/version>       # unload a module
module switch old_mod new_mod        # swap versions
module purge                         # unload ALL modules
module whatis <name>                 # short description
module show <name>                   # show what env vars the module sets
```

**In batch scripts: always load modules explicitly.** Jobs start in a clean environment — modules from your interactive session are not inherited.

```bash
#!/bin/sh
#BSUB ...
module purge
module load cuda/11.7
python train.py
```

### Known modules

| Module | Notes |
|--------|-------|
| `cuda/11.6` | Confirmed in official GPU job template |
| `cuda/11.7` | Confirmed in DTU MLOps guide |
| `python3/3.10.7` | System Python 3.10 |
| `python3/3.11.7` | System Python 3.11 |
| `pandas/2.1.3-python-3.11.7` | Auto-loads python3/3.11.7 + numpy + deps |
| `mpi/4.1.4-gcc-12.2.0-binutils-2.39` | MPI for multi-node jobs |
| `gcc/12.2.0-binutils-2.39` | GNU compiler |

> Check the live list on the cluster: `module avail 2>&1 | grep -iE "cuda|torch|python"`

No PyTorch, cuDNN, or NCCL system modules are publicly documented — install these inside your own conda/venv environment.

---

## Software stacks

DTU HPC maintains two stacks:

### Default stack (evolving)
Active for any user with a fresh `.bashrc`. Continuously updated. Good for interactive use.

### DCC stack (versioned, reproducible)
Only available on LSF cluster nodes. Enables pinned, reproducible environments.

```bash
# Enable DCC stack (add to .bashrc for permanent activation):
source /dtu/sw/dcc/dcc-sw.bash

# Load a specific pinned snapshot:
module purge
module load dcc-setup/2023-aug      # other versions: 2021-nov, 2020-aug, 2019-jun
```

Optional pre-load variables:

```bash
export USER_COMPILER=...
export USER_CPUTYPE=avx2    # avx | avx2 | avx512
```

---

## Python setup — install to /work3, not home

**The 30 GB home quota fills instantly with conda envs and model weights.** Always install your environment on `/work3`.

### Option A: Miniconda (recommended for ML workloads)

```bash
# Download installer (run on login node or interactive node)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install to /work3 (not home!)
bash Miniconda3-latest-Linux-x86_64.sh -b -p /work3/$USER/danish_asr/conda

# Initialize conda (adds to ~/.bashrc)
/work3/$USER/danish_asr/conda/bin/conda init bash
source ~/.bashrc

# Create environment
conda create -n danish_asr python=3.11 -y
conda activate danish_asr
```

### Option B: Module + virtualenv

```bash
module load python3/3.11.7
python3 -m venv /work3/$USER/danish_asr/venv
source /work3/$USER/danish_asr/venv/bin/activate
pip install <packages>
```

### Option C: Module + pre-built scientific stack

```bash
module load pandas/2.1.3-python-3.11.7   # auto-loads python3/3.11.7, numpy, etc.
source /work3/$USER/danish_asr/venv/bin/activate  # additional packages on top
```

---

## PyTorch + CUDA installation

No PyTorch system module is available — install it in your environment.

```bash
# Match to the CUDA module you will load in your job scripts:

# For cuda/11.6:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116

# For cuda/11.7:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# Verify:
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**Key constraint:** A100 GPUs require CUDA 11.0 or newer. Always use `module load cuda/11.6` or higher when targeting `gpua100`.

---

## fairseq2 / omnilingual-ASR installation

fairseq2 ships CUDA-specific wheels — the CUDA version in the wheel must match the loaded CUDA module.

```bash
# First, load the matching CUDA module:
module load cuda/11.7

# Install fairseq2 matching cu117:
pip install fairseq2 --extra-index-url https://fair.pkg.anvil.fairseq2.org/whl/cu117/stable/

# Install omnilingual-asr:
pip install "omnilingual-asr[data]"
```

Adjust `cu117` to match your loaded CUDA version (e.g., `cu116` for `cuda/11.6`).

Do this **once** in an interactive session. Batch jobs just activate the environment.

---

## uv (alternative to conda/pip)

uv is not documented in DTU HPC official docs, but works fine:

```bash
# Install uv (run once on login node):
curl -LsSf https://astral.sh/uv/install.sh | sh

# Use uv from /work3 to avoid home quota:
export UV_CACHE_DIR=/work3/$USER/.uv-cache

# Clone and sync the project:
git clone <repo-url> /work3/$USER/danish_asr
cd /work3/$USER/danish_asr
uv sync
```

In job scripts, call `uv run python` instead of `python`.

---

## Activating your environment in job scripts

### conda

```bash
# Source conda, then activate:
source /work3/$USER/danish_asr/conda/etc/profile.d/conda.sh
conda activate danish_asr
python -u train.py
```

### venv

```bash
source /work3/$USER/danish_asr/venv/bin/activate
python -u train.py
```

### Direct path (avoids activation entirely)

```bash
/work3/$USER/danish_asr/conda/envs/danish_asr/bin/python train.py
```

> Use `python -u` (unbuffered) for real-time output in `bpeek` and log files.

---

## Checking available CUDA and Python modules on the cluster

Run this on the login node after connecting:

```bash
module avail 2>&1 | grep -iE "cuda"
module avail 2>&1 | grep -iE "python"
module avail 2>&1 | grep -iE "torch"
```

---

## Auto-loading modules on login

Edit `~/.gbarrc` to auto-load modules on login:

```
MODULES=python3/3.11.7
```

> Warning: auto-loaded modules can conflict with modules in specific job scripts. Prefer loading modules explicitly in each job script.

---

## Internet access on compute nodes

DTU HPC compute nodes have outbound internet access (unlike some other HPC clusters). You can `pip install` and access HuggingFace Hub from within running jobs. However, it is better practice to:

1. Download all datasets and model weights **before submitting the job** (in an interactive session)
2. Install all packages **before submitting the job** (in an interactive session or at environment creation time)

This avoids job failure due to network issues, and reduces job startup time.
