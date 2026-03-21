# DTU HPC Reference

Single reference for all DTU HPC operations. Replaces the previous multi-file documentation.

---

## Access

### SSH

```bash
# On-campus or VPN (password only):
ssh <userid>@login.hpc.dtu.dk

# Off-campus (SSH key + passphrase + DTU password):
ssh -i ~/.ssh/gbar <userid>@login.hpc.dtu.dk
```

SSH key setup (one-time, from DTU network):
```bash
ssh-keygen -t ed25519 -f ~/.ssh/gbar
ssh <userid>@transfer.gbar.dtu.dk mkdir -m 700 -p .ssh
scp ~/.ssh/gbar.pub <userid>@transfer.gbar.dtu.dk:.ssh/authorized_keys
ssh <userid>@transfer.gbar.dtu.dk chmod 600 .ssh/authorized_keys
```

Optional `~/.ssh/config`:
```
Host gbar1
    User <userid>
    IdentityFile ~/.ssh/gbar
    Hostname login1.gbar.dtu.dk
```

Login nodes: `login1.gbar.dtu.dk`, `login2.gbar.dtu.dk`. **Do NOT run applications on login nodes** — job submission and file management only.

### ThinLinc (graphical)

Server: `thinlinc.gbar.dtu.dk`. On-campus: password. External: SSH key auth.

### OS

All compute nodes: Scientific Linux 7.9. VS Code Remote last working version: 1.85.

---

## Storage

| Path | Quota | Backed up | Speed | Use for |
|------|-------|-----------|-------|---------|
| `$HOME` (`/zhome/<l>/<user>/`) | 30 GB | Yes | Moderate | Code, configs, scripts |
| `/work3/<user>` | By request | **No** | High (BeeGFS/Infiniband) | Datasets, checkpoints, caches, conda |
| `/work1/<user>` | By request | **No** | High | Same as /work3 |

**Critical:** Exceeding 30 GB home quota kills running jobs and can prevent login.

Check usage:
```bash
getquota_zhome.sh        # home
getquota_work3.sh        # scratch (also shows file-count limit: 2M files)
```

Request /work3 access: email support@cc.dtu.dk.

### Project directory layout

```
$HOME/danish_asr/                    # Git repo (backed up)

/work3/$USER/danish_asr/
├── data/parquet/                    # CoRal-v3 Parquet (fairseq2 input)
├── hf_cache/                        # HuggingFace cache
├── fairseq2_cache/                  # fairseq2 model/asset cache
├── conda/                           # Miniconda (NOT in $HOME)
└── outputs/run_YYYYMMDD_HHMMSS/     # Training runs
```

### Required environment variables (every job script)

```bash
export HF_HOME=/work3/$USER/danish_asr/hf_cache
export HF_DATASETS_CACHE=/work3/$USER/danish_asr/hf_cache/datasets
export FAIRSEQ2_CACHE_DIR=/work3/$USER/danish_asr/fairseq2_cache
export TMPDIR=/work3/$USER/tmp
mkdir -p $TMPDIR
```

### File transfer

Use `transfer.gbar.dtu.dk` (10 GbE + Infiniband), not login nodes:
```bash
rsync -avP local_data/ <userid>@transfer.gbar.dtu.dk:/work3/<userid>/danish_asr/data/
rsync -avP <userid>@transfer.gbar.dtu.dk:/work3/<userid>/danish_asr/outputs/ ./outputs/
```

### Data lifecycle

Account expiry = immediate data loss. `/work3` is not backed up. Export important outputs before expiry.

---

## GPU Queues

All GPU queues: 24h max walltime. Default walltime if omitted: **15 minutes**.

| Queue | GPU | Nodes | GPUs/node | VRAM |
|-------|-----|-------|-----------|------|
| `gpua100` | A100 PCIe | 4+6 | 2 | 40 GB / 80 GB |
| `gpul40s` | L40s PCIe | 6 | 2 | 48 GB |
| `gpuv100` | V100 PCIe | 6+8 | 2 | 16 GB / 32 GB |
| `gpuv100` | V100 NVLink SXM2 | 3 | 4 | 32 GB |
| `gpua10` | A10 PCIe | 1 | 2 | 24 GB |
| `gpua40` | A40 NVLink | 1 | 2 | 48 GB |

CPU queue `hpc`: 72h max, 100 cores/job.

### GPU selection flags

```bash
#BSUB -R "select[gpu80gb]"     # 80 GB A100
#BSUB -R "select[gpu32gb]"     # 32 GB V100
#BSUB -R "select[sxm2]"        # V100 NVLink SXM2
```

Always use `mode=exclusive_process`.

---

## Job Submission (LSF)

### BSUB directive reference

```bash
#BSUB -J name                              # Job name
#BSUB -q queue                             # Queue
#BSUB -n N                                 # CPU cores (use 4 per GPU)
#BSUB -R "span[hosts=1]"                   # Single node
#BSUB -R "rusage[mem=16GB]"                # RAM per core (no space in value)
#BSUB -M 18GB                              # Hard per-process memory ceiling
#BSUB -gpu "num=1:mode=exclusive_process"   # GPU count + exclusive
#BSUB -W 24:00                             # Walltime hh:mm
#BSUB -o logs/train_%J.out                 # Stdout (%J=jobid, %I=array index)
#BSUB -e logs/train_%J.err                 # Stderr
#BSUB -B                                   # Email at start
#BSUB -N                                   # Email at end
#BSUB -Ne                                  # Email on failure only
#BSUB -u email@dtu.dk                      # Email address
```

Memory is **per core**: 4 cores × `rusage[mem=16GB]` = 64 GB total.

### Complete job script (single A100)

```bash
#!/bin/sh
#BSUB -J danish_asr_train
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 18GB
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -o logs/train_%J.out
#BSUB -e logs/train_%J.err

module purge
module load cuda/11.7

export HF_HOME=/work3/$USER/danish_asr/hf_cache
export FAIRSEQ2_CACHE_DIR=/work3/$USER/danish_asr/fairseq2_cache
export TMPDIR=/work3/$USER/tmp
mkdir -p $TMPDIR logs

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.free \
    --format=csv -l 10 > logs/gpu_${LSB_JOBID}.csv &
NVPID=$!

source /work3/$USER/danish_asr/conda/etc/profile.d/conda.sh
conda activate danish_asr

OUTPUT_DIR="/work3/$USER/danish_asr/outputs/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

python -u train.py --output-dir $OUTPUT_DIR

kill $NVPID
```

### Dual GPU variant

Change these lines only:
```bash
#BSUB -n 8
#BSUB -gpu "num=2:mode=exclusive_process"
```

### CPU-only job (data conversion)

```bash
#!/bin/sh
#BSUB -J coral_convert
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 12:00
#BSUB -o logs/convert_%J.out
#BSUB -e logs/convert_%J.err

export HF_HOME=/work3/$USER/danish_asr/hf_cache
mkdir -p logs

source /work3/$USER/danish_asr/conda/etc/profile.d/conda.sh
conda activate danish_asr
python -u scripts/convert_coral_to_parquet.py --subset all --num-workers 8
```

### Job dependencies

```bash
#BSUB -w "done(44951)"                    # after job succeeds
#BSUB -w "ended(44951)"                   # after job ends (success or fail)
#BSUB -w "done(44951) && done(44952)"     # both succeed
```

Check: `bjdepinfo <jobid>` (what it depends on), `bjdepinfo -c <jobid>` (what depends on it).

### Job arrays

```bash
#BSUB -J My_array[1-25]       # 25 independent jobs
#BSUB -J My_array[1-20]%5     # max 5 concurrent
```

Index in shell: `$LSB_JOBINDEX`. Manage: `bkill <jobid>[3]`, `bjobs <jobid>[3]`.

### LSF environment variables (inside running jobs)

| Variable | Value |
|----------|-------|
| `$LSB_JOBID` | Job ID |
| `$LSB_JOBINDEX` | Array index |
| `$LSB_DJOB_NUMPROC` | Allocated CPU cores |

---

## Job Management Commands

| Command | Purpose |
|---------|---------|
| `bsub < job.sh` | Submit |
| `bstat` | Your jobs (compact) |
| `bstat -C <id>` | CPU efficiency |
| `bstat -M <id>` | Memory usage |
| `bjobs` | Your jobs (standard) |
| `bjobs -l <id>` | Verbose + PENDING REASONS |
| `bpeek <id>` | Stdout snapshot |
| `bpeek -f <id>` | Stream live output |
| `bhist -l <id>` | History + termination reason |
| `bhist -a` | All past jobs |
| `bacct <id>` | Accounting (CPU, wait, turnaround) |
| `showstart <id>` | Estimated start time |
| `bkill <id>` | Kill job |
| `bkill 0` | Kill ALL your jobs |
| `bnvtop <id>` | GPU utilization (batch jobs only) |

Job states: `PEND` (queued), `RUN`, `DONE` (success), `EXIT` (failed/killed), `SSUSP`/`USUSP` (suspended).

### Cluster status

```bash
nodestat -G gpua100        # GPU availability
nodestat -F hpc            # CPU nodes
bqueues gpua100            # queue stats
classstat hpc              # cluster-wide totals
```

---

## Debugging Failed Jobs

### Step 1: Termination reason

```bash
bjobs -l <id>     # if still in system
bhist -l <id>     # completed jobs
```

| Reason | Meaning | Fix |
|--------|---------|-----|
| `TERM_MEMLIMIT` | CPU OOM | Increase `rusage[mem=...]` |
| `TERM_RUNLIMIT` | Walltime exceeded | Increase `-W` (default 15 min!) |
| `TERM_CPULIMIT` | CPU time exceeded | Increase limit |
| `TERM_CWD_NOTEXIST` | Log dir missing | `mkdir -p logs` before submit |
| Log output mailed, not written to file | `$USER` not expanded in `#BSUB -o/-e` | Use hardcoded username (e.g. `s204696`) in `#BSUB` log path directives — LSF does not expand shell variables in `#BSUB` headers, only in the script body |
| `TERM_OWNER` | Killed by user | `bkill` was called |
| `TERM_UNKNOWN` | Unknown | Check `.err` file |

### Step 2: Read logs

```bash
bpeek -f <id>                   # while running
cat logs/train_<id>.err          # after completion
tail -100 logs/train_<id>.out
```

### Step 3: Common fixes

| Symptom | Fix |
|---------|-----|
| CUDA OOM in log | Reduce batch size or use larger GPU (`select[gpu80gb]`) |
| Job exits immediately | Script error — check `.err`; test interactively first |
| Job stuck PEND | `bjobs -l` for reasons; try different queue |
| Low CPU efficiency (`bstat -C`) | Reduce `-n` or fix parallelism |
| Memory used, GPU compute ≈ 0 | Data loading bottleneck — more workers, pin memory |

### Step 4: Smoke test before batch

```bash
bsub -Is -q gpua100 -n 4 -R "rusage[mem=16GB]" -R "span[hosts=1]" \
    -gpu "num=1:mode=exclusive_process" -W 1:00 bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -u train.py --max-steps 5
```

---

## Interactive Sessions

### Shared GPU nodes (always on, no queue)

```bash
a100sh     # 2× A100 40 GB — recommended
sxm2sh     # 4× V100 SXM2 32 GB
voltash    # 2× V100 16 GB
linuxsh    # CPU only
```

Shared — check `nvidia-smi` first. For dev and short tests only.

### Exclusive GPU via scheduler

```bash
# A100, 1h
bsub -Is -q gpua100 -n 4 -R "rusage[mem=16GB]" -R "span[hosts=1]" \
    -gpu "num=1:mode=exclusive_process" -W 1:00 bash

# V100 32GB, 2h
bsub -Is -q gpuv100 -n 4 -R "rusage[mem=8GB]" -R "span[hosts=1]" \
    -R "select[gpu32gb]" -gpu "num=1:mode=exclusive_process" -W 2:00 bash

# L40s 48GB, 1h
bsub -Is -q gpul40s -n 4 -R "rusage[mem=16GB]" -R "span[hosts=1]" \
    -gpu "num=1:mode=exclusive_process" -W 1:00 bash
```

Session killed at `-W` expiry. Default 15 min if omitted.

### Jupyter on GPU node

1. Get interactive session, note hostname (`hostname`)
2. Start Jupyter: `jupyter lab --no-browser --port=40000 --ip=$HOSTNAME`
3. Local tunnel: `ssh -N -L 40000:HOSTNAME:40000 <userid>@login.hpc.dtu.dk`
4. Open: `http://127.0.0.1:40000/?token=<token>`

---

## Monitoring

### GPU

```bash
nvidia-smi                                    # snapshot
watch -n 2 nvidia-smi                         # refresh (spot checks only on shared nodes)
bnvtop <jobid>                                # batch job GPU utilization
CUDA_VISIBLE_DEVICES=0 python train.py        # force specific GPU
```

### GPU logging in batch jobs

```bash
nvidia-smi --query-gpu=index,timestamp,utilization.gpu,memory.total,memory.used \
    --format=csv -l 10 > logs/gpu_${LSB_JOBID}.csv &
NVPID=$!
# ... training ...
kill $NVPID
```

### PyTorch memory profiling

```python
import torch
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Peak:      {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
torch.cuda.reset_peak_memory_stats()
```

### W&B

Works directly from compute nodes (outbound HTTPS):
```bash
export WANDB_API_KEY=<key>
python train.py    # wandb.init() syncs automatically
```

Offline fallback:
```bash
export WANDB_MODE=offline
# After job: wandb sync runs/<run-dir>
```

### TensorBoard via tunnel

On compute node: `tensorboard --logdir=logs --port=6006 --host=$HOSTNAME &`
Local: `ssh -N -L 6006:HOSTNAME:6006 <userid>@login.hpc.dtu.dk`

---

## Python Environment Setup

### Module system

```bash
module list                   # loaded modules
module avail cuda             # available CUDA versions
module load cuda/11.7         # load (always explicit in job scripts)
module purge                  # unload all
```

Jobs start clean — always `module purge` + `module load` in scripts.

### Known modules

| Module | Notes |
|--------|-------|
| `cuda/11.6`, `cuda/11.7` | Confirmed available |
| `python3/3.10.7`, `python3/3.11.7` | System Python |
| `pandas/2.1.3-python-3.11.7` | Auto-loads numpy + deps |

No PyTorch, cuDNN, or NCCL system modules — install in your own environment.

### Install Miniconda (to /work3, never $HOME)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /work3/$USER/danish_asr/conda
/work3/$USER/danish_asr/conda/bin/conda init bash
source ~/.bashrc
conda create -n danish_asr python=3.11 -y
conda activate danish_asr
```

### PyTorch + CUDA

```bash
# Match to loaded CUDA module:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

A100 requires CUDA ≥ 11.0.

### fairseq2 + omnilingual-ASR

```bash
module load cuda/11.7
pip install fairseq2 --extra-index-url https://fair.pkg.anvil.fairseq2.org/whl/cu117/stable/
pip install "omnilingual-asr[data]"
```

CUDA version in fairseq2 wheel MUST match loaded module.

### uv alternative

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export UV_CACHE_DIR=/work3/$USER/.uv-cache
cd /work3/$USER/danish_asr && uv sync
```

In job scripts: `uv run python` instead of `python`.

### Activating in job scripts

```bash
# conda:
source /work3/$USER/danish_asr/conda/etc/profile.d/conda.sh
conda activate danish_asr

# venv:
source /work3/$USER/danish_asr/venv/bin/activate

# direct (no activation):
/work3/$USER/danish_asr/conda/envs/danish_asr/bin/python train.py
```

Use `python -u` (unbuffered) for real-time output in `bpeek`.

### Internet on compute nodes

Available but unreliable. Download all datasets and install all packages **before** submitting jobs.

---

## Gotchas

| Issue | Details |
|-------|---------|
| Default walltime 15 min | Always set `#BSUB -W` |
| DOS line endings | Fix with `dos2unix job.sh` — Windows `\r\n` fails silently |
| Danish chars in paths | Avoid æ, ø, å in filenames |
| Home quota 30 GB | Conda + model weights overflow instantly — use /work3 |
| `LD_LIBRARY_PATH` in .bashrc | Never set this — causes subtle breakage |
| Account expiry | Immediate data loss — back up first |

---

## Official Links

- [DTU HPC main](https://www.hpc.dtu.dk/)
- [GPU nodes](https://www.hpc.dtu.dk/?page_id=2129)
- [GPU jobs under LSF10](https://www.hpc.dtu.dk/?page_id=2759)
- [Batch jobs guide](https://www.hpc.dtu.dk/?page_id=1416)
- [Managing jobs](https://www.hpc.dtu.dk/?page_id=1519)
- [Storage](https://www.hpc.dtu.dk/?page_id=59)
- [HPC best practice](https://www.hpc.dtu.dk/?page_id=4317)
- [DTU MLOps HPC guide](https://skaftenicki.github.io/dtu_mlops/s10_extra/high_performance_clusters/)
- Support: support@cc.dtu.dk
