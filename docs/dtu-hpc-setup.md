# DTU HPC Setup Guide

How to run the omnilingual ASR finetuning pipeline on DTU's HPC cluster.

References:
- [GPU nodes](https://www.hpc.dtu.dk/?page_id=2129)
- [Using GPUs under LSF10](https://www.hpc.dtu.dk/?page_id=2759)
- [GPU jobs: best practice guide](https://www.hpc.dtu.dk/?page_id=4976)
- [Batch Jobs under LSF10](https://www.hpc.dtu.dk/?page_id=1416)
- [Managing jobs under LSF10](https://www.hpc.dtu.dk/?page_id=1519)
- [Job workflow and monitoring](https://www.hpc.dtu.dk/?page_id=4204)
- [Monitoring jobs: advanced](https://www.hpc.dtu.dk/?page_id=2652)
- [Accessing the LSF10 cluster](https://www.hpc.dtu.dk/?page_id=2501)
- [AI/DL/ML corner](https://www.hpc.dtu.dk/?page_id=4788)
- [LSF termination reasons](https://www.hpc.dtu.dk/lsf931/lsf_admin/termination_reasons_lsf.html)
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
| `bsub < script.sh` | Submit batch job |
| `bsub -Is -q gpua100 -n 4 -R "rusage[mem=16GB]" -R "span[hosts=1]" -gpu "num=1:mode=exclusive_process" -W 1:00 bash` | Interactive GPU session |
| `bjobs` | List your running/pending jobs |
| `bjobs -l <jobid>` | Detailed job info + PENDING REASONS |
| `bpeek <jobid>` | View live stdout/stderr of running job |
| `bpeek -f <jobid>` | Follow/stream live output (like tail -f) |
| `bkill <jobid>` | Kill a job |
| `bkill 0` | Kill ALL your jobs |
| `bstat` | Your jobs, compact view |
| `bstat -C <jobid>` | CPU efficiency (EFFIC %) |
| `bstat -M <jobid>` | Memory usage (MEM / MAX / AVG / LIM) |
| `bhist -l <jobid>` | Full history + termination reason |
| `bhist -a` | All your past jobs |
| `bacct <jobid>` | Accounting summary (CPU, wait, turnaround) |
| `showstart <jobid>` | Estimated start time for pending job |
| `bqueues` | List available queues |
| `nodestat -G gpua100` | GPU node availability + specs |
| `classstat hpc` | Cluster-wide load overview |
| `bnvtop <jobid>` | Live GPU monitor for a batch job |
| `voltash` | Interactive V100 node (16 GB) |
| `sxm2sh` | Interactive V100 SXM2 node (32 GB) |
| `a100sh` | Interactive A100 node (40 GB) |

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

## Storage Systems

Sources: [DTU HPC Storages](https://www.hpc.dtu.dk/?page_id=59), [Disk Space and Quota FAQ](https://www.hpc.dtu.dk/?page_id=927)

### Overview

| Location | Quota | Filesystem | Backed up | Speed | Use for |
|---|---|---|---|---|---|
| `$HOME` (`/zhome/<u>/<username>`) | 30 GB (default) | Dell EMC + ZFS/NFS | Yes (tape + snapshots) | Moderate | Code, configs, results |
| `/work3/<username>` | By request | BeeGFS over Infiniband | No | High (parallel) | Datasets, checkpoints, HF cache |
| `/work1/<username>` | By request | BeeGFS over Infiniband | No | High (parallel) | Same as /work3 |

**Total home storage pool:** 220 TB
**Total scratch pool:** 98 TB + 142 TB = 240 TB combined

### Home Directory

- **Default quota:** 30 GB per user
- **Path format:** `/zhome/<first-letter>/<username>/` (i.e. `$HOME`)
- **Non-Linux access:** Windows: `\\home.cc.dtu.dk\<username>` / Mac: `smb://home.cc.dtu.dk/<username>`
- **Backup:** Hourly + daily snapshots; snapshot replication to secondary server; tape backup to remote DTU site
- **Check usage:**
  ```bash
  getquota_zhome.sh
  # Output: "You are using 12.34 GB of 30.00 GB."
  # Or manually:
  cd ~ && du -h --max-depth=1 .
  ```
- **Extra quota:** Can request a temporary increase by emailing support@cc.dtu.dk. Only for limited time.
- **Warning:** If you exceed quota, running jobs are killed and you may be unable to log in.

### /work3 Scratch Filesystem

- **How to get access:** Email support@cc.dtu.dk requesting scratch space; state the reason. A directory named after your username will be created at `/work3/<username>`.
- **Quota:** Assigned per request (shown in `getquota_work3.sh` output as `hard` limit)
- **Filesystem:** BeeGFS (distributed parallel filesystem) connected via Infiniband — high throughput, low latency
- **Accessible from:** All HPC nodes (login nodes and compute nodes)
- **NOT backed up** — scratch is for temporary/intermediate files only
- **Check usage:**
  ```bash
  getquota_work3.sh
  # Example output:
  # user/group     ||           size          ||    chunk files
  # name     |  id  ||    used    |    hard    ||  used   |  hard
  # ----------|------||------------|------------||---------|---------
  # s123456 |54321 ||  158.75 GiB|  400.00 GiB||  119758 |  2000000
  ```
  Note the `chunk files` column — this is the inode/file-count limit (shown above: 2,000,000 files hard limit).
- **On first access:** Read the `readme.txt` in your scratch directory before using it.
- **Performance warning:** Performance degrades when the filesystem is near capacity. Delete old/unnecessary files regularly.

### File Transfer

- **Dedicated transfer server:** `transfer.gbar.dtu.dk`
  - Connected to both 10 Gbit Ethernet AND Infiniband — faster than the login node
  - Can access home directory and all scratch filesystems
  - Use this instead of the login node for large transfers
- **Preferred tool for bulk transfers:** `rsync` (preserves metadata, resumable)
  ```bash
  # Upload CoRal data or Parquet files to /work3
  rsync -avP --progress local_data/ <username>@transfer.gbar.dtu.dk:/work3/<username>/danish_asr/data/

  # Download results
  rsync -avP <username>@transfer.gbar.dtu.dk:/work3/<username>/danish_asr/outputs/ ./outputs/
  ```
- **SCP also works** for single files:
  ```bash
  scp file.tar.gz <username>@transfer.gbar.dtu.dk:/work3/<username>/
  ```
- **SFTP clients** (FileZilla, Cyberduck): connect to `transfer.gbar.dtu.dk` with DTU credentials

### Data Lifecycle Warning

When your DTU account expires, all data becomes inaccessible immediately. Non-backed-up data (i.e., everything in `/work3`) cannot be recovered after account closure.

**Back up all important outputs to home or external storage before your account expires.**

### Recommended Directory Layout for This Project

```
$HOME/danish_asr/              # Git repo, configs, scripts (backed up)

/work3/<username>/danish_asr/
├── data/
│   ├── parquet/               # CoRal-v3 converted to Parquet (omnilingual ASR input)
│   └── raw/                   # Optional: raw HF downloads if not using hf_cache
├── hf_cache/                  # HuggingFace cache (coral-v3 downloads, ~100-200 GB)
├── fairseq2_cache/            # fairseq2 model/asset cache
└── outputs/                   # Training runs, checkpoints, logs
    └── run_YYYYMMDD_HHMMSS/
```

### Environment Variables — Set in Job Scripts

```bash
# Point HF and fairseq2 caches to /work3 (never fill $HOME with model weights)
export HF_HOME=/work3/<username>/danish_asr/hf_cache
export HF_DATASETS_CACHE=/work3/<username>/danish_asr/hf_cache/datasets
export FAIRSEQ2_CACHE_DIR=/work3/<username>/danish_asr/fairseq2_cache

# Useful to avoid accidental home-directory writes during jobs
export TMPDIR=/work3/<username>/tmp
mkdir -p $TMPDIR
```

Add these exports to the top of every `scripts/hpc/*.sh` job script, after the `#BSUB` directives.

## Interactive Sessions

### What interactive nodes are available

DTU HPC provides three dedicated shared interactive GPU nodes. They are always on and shared between users — check `nvidia-smi` before running anything intensive.

| Command  | Hardware                        | VRAM      |
|----------|---------------------------------|-----------|
| `voltash`  | 1 node, 2× V100 PCIe           | 16 GB each |
| `sxm2sh`   | 1 node, 4× V100-SXM2 NVLink   | 32 GB each |
| `a100sh`   | 1 node, 2× A100 PCIe NVLink   | 40 GB each |

These nodes are for **development, profiling, and short test jobs only** — not for full training runs. Walltime is not enforced but courtesy applies.

### How to reach an interactive node

```bash
# After SSH into login node:
ssh <dtu-username>@login.hpc.dtu.dk

# Option 1: generic interactive CPU node (no GPU)
linuxsh

# Option 2: jump directly to named interactive GPU nodes
voltash    # V100, 16 GB
sxm2sh     # V100 SXM2, 32 GB
a100sh     # A100, 40 GB  <-- use this for our project
```

`linuxsh` (and `voltash`/`sxm2sh`/`a100sh`) are shell wrappers that run an `ssh` or `bsub -Is` behind the scenes and land you in an interactive shell on the target node. The filesystem is shared, so your home directory and `/work3` are accessible from any node.

### bsub -Is: interactive batch job with GPU resources

Use `bsub -Is` to request a **dedicated** GPU via the scheduler (not the shared interactive nodes). This is the right approach for longer interactive testing that needs exclusive GPU access.

```bash
# Basic: 1 A100, 4 cores, 16 GB RAM, 1-hour interactive shell
bsub -Is -q gpua100 \
    -n 4 \
    -R "rusage[mem=16GB]" \
    -R "span[hosts=1]" \
    -gpu "num=1:mode=exclusive_process" \
    -W 1:00 \
    bash

# Longer session (2 hours) on V100 32 GB
bsub -Is -q gpuv100 \
    -n 4 \
    -R "rusage[mem=8GB]" \
    -R "span[hosts=1]" \
    -R "select[gpu32gb]" \
    -gpu "num=1:mode=exclusive_process" \
    -W 2:00 \
    bash

# L40s (48 GB, good VRAM-to-cost ratio)
bsub -Is -q gpul40s \
    -n 4 \
    -R "rusage[mem=16GB]" \
    -R "span[hosts=1]" \
    -gpu "num=1:mode=exclusive_process" \
    -W 1:00 \
    bash
```

Key flags for `-Is`:
- `-Is` — interactive shell (blocks until allocation is granted, then opens shell)
- `-W hh:mm` — walltime for the interactive session (default 15 min if omitted)
- `-gpu "num=N:mode=exclusive_process"` — always use `exclusive_process` to prevent GPU memory conflicts between users

After the `-W` walltime expires your session is killed automatically.

### Check which GPUs are free before jumping on an interactive node

```bash
# On the interactive node, check occupancy
nvidia-smi

# On the login node, check GPU node availability
nodestat -g gpua100
nodestat -g gpuv100
nodestat -G gpua100    # more detailed GPU info
```

---

## Running Jupyter Notebook on a GPU Node

The GPU nodes are firewalled — they are not directly reachable from your laptop. You need a **two-hop SSH tunnel**: laptop -> login node -> compute node.

### Step 1: Get an interactive GPU session and note the hostname

```bash
# On HPC login node — use a100sh or bsub -Is
a100sh
# You are now on e.g. n-62-20-1

# Or with bsub -Is (to get it through the scheduler):
bsub -Is -q gpua100 -n 4 -R "rusage[mem=16GB]" -R "span[hosts=1]" \
    -gpu "num=1:mode=exclusive_process" -W 2:00 bash
# Check what node you landed on:
hostname   # e.g. gpunode042
```

### Step 2: Start Jupyter on the compute node

```bash
# On the GPU compute node:
module load cuda/12.1
source ~/danish_asr/.venv/bin/activate    # or conda activate danish_asr

# Start Jupyter — bind to the node's hostname, pick a port (40000-49999)
jupyter notebook --no-browser --port=40000 --ip=$HOSTNAME

# Or JupyterLab:
jupyter lab --no-browser --port=40000 --ip=$HOSTNAME
```

Jupyter will print a URL like:
```
http://gpunode042:40000/?token=401bb4a3a4faeafd2fd948a137b0f6857ba4aa6e6fc47b7d
```

Note the **hostname** (`gpunode042`) and **port** (`40000`) and **token**.

### Step 3: Set up the SSH tunnel from your local machine

```bash
# On your LOCAL machine (Linux/macOS):
# Replace HOSTNAME with the compute node name, PORT with Jupyter port, USERNAME with your DTU ID
ssh -N -L 40000:HOSTNAME:40000 USERNAME@login.hpc.dtu.dk

# Example:
ssh -N -L 40000:gpunode042:40000 s123456@login.hpc.dtu.dk

# Run in background with -f:
ssh -Nf -L 40000:gpunode042:40000 s123456@login.hpc.dtu.dk
```

For the shared interactive nodes (voltash/a100sh), the hostname is fixed — e.g. `n-62-20-1` for voltash. Typical ports used are 40000, 40001, 40002 (incremented per concurrent user).

```bash
# Alternative gbar login node:
ssh -N -L 40000:n-62-20-1:40000 s123456@login.gbar.dtu.dk
```

### Step 4: Open in your browser

Navigate to `http://127.0.0.1:40000/?token=<your-token>` in your local browser.

### Windows (PuTTY)

In PuTTY: Connection > SSH > Tunnels > Source port: `40000`, Destination: `HOSTNAME:40000`, type Local. Then connect to `login.hpc.dtu.dk`.

### VS Code alternative

VS Code's Remote SSH extension can forward ports directly through the tunnel — connect to the login node and then use the "Ports" tab to forward `HOSTNAME:40000`.

---

## Monitoring GPU Usage

### nvidia-smi

```bash
# Snapshot of all GPUs
nvidia-smi

# Compact format every 2 seconds
watch -n 2 nvidia-smi

# Specific GPU only
nvidia-smi -i 0

# Query specific metrics in CSV (good for logging in a batch job)
nvidia-smi --query-gpu=index,timestamp,utilization.gpu,memory.total,memory.used,memory.free \
    --format=csv -l 2

# Show which processes are using each GPU and how much memory
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# Select a specific GPU for your process (0-indexed)
CUDA_VISIBLE_DEVICES=0 python train.py
```

**Important:** DTU HPC docs warn **do not** run `nvidia-smi` with `-l` (loop) or under `watch` for extended periods on shared interactive nodes — it consumes resources. Use it briefly to check, then stop.

### bnvtop (DTU-specific, batch jobs only)

```bash
# Open live GPU monitor for a running batch job
bnvtop <JOBID>
```

`bnvtop` is a DTU wrapper around `nvtop` — it shows a graphical terminal view of GPU memory and compute utilization over time. It attaches to the node running your job. Only works for **batch jobs** (not interactive sessions). Do not run it for more than a few minutes — it is for spot-checking, not continuous monitoring.

Warning signs to look for in bnvtop:
- Zero memory AND zero compute = GPU not being used at all
- Memory used but compute near zero = data loading bottleneck, GPU idle waiting for CPU
- Compute spiky/bursty = CPU is the bottleneck feeding data to GPU

### Logging GPU stats during a batch job

Add this to your job script to capture a GPU log in the background:

```bash
# In your job script, before your training command:
nvidia-smi --query-gpu=index,timestamp,utilization.gpu,memory.total,memory.used,memory.free \
    --format=csv -l 5 > $OUTPUT_DIR/gpu_stats_${LSB_JOBID}.csv &
NVIDIA_SMI_PID=$!

# ... your training command ...

# After training:
kill $NVIDIA_SMI_PID
```

---

## Job Monitoring Commands

### bjobs — list running/pending jobs

```bash
bjobs                    # all your jobs (RUN + PEND)
bjobs <jobid>            # specific job
bjobs -l <jobid>         # verbose: walltime, memlimit, resource usage, PENDING REASONS
bjobs -u all             # all users' jobs (see cluster load)
```

Output columns: `JOBID | USER | QUEUE | JOB_NAME | NALLOC (slots) | STAT | START_TIME | TIME_LEFT`

Job states:

| STAT   | Meaning |
|--------|---------|
| PEND   | Queued, waiting for resources |
| RUN    | Currently executing |
| DONE   | Completed successfully |
| EXIT   | Failed / killed |
| SSUSP  | Suspended by system |
| USUSP  | Suspended by user |

For pending jobs, `bjobs -l <jobid>` shows a **PENDING REASONS** section explaining exactly why the job hasn't started (e.g., not enough free GPUs, memory unavailable on any node).

### bstat — DTU's compact status tool

```bash
bstat                    # all your jobs, compact
bstat <jobid>            # specific job
bstat -C <jobid>         # CPU efficiency (shows EFFIC %; aim for ~100%)
bstat -M <jobid>         # memory usage (MEM, MAX, AVG, LIM per host)
bstat -u <username>      # filter by user
bstat -q <queue>         # filter by queue
```

CPU efficiency well below 100% means you requested too many cores or there's a parallelism issue. Memory `MAX` much lower than `LIM` means you over-requested RAM.

### bpeek — view live stdout/stderr of a running job

```bash
bpeek <jobid>            # print current stdout+stderr up to now
bpeek -f <jobid>         # follow mode (like tail -f), streams output continuously
```

This is the primary debugging tool for batch jobs — use it instead of waiting for the job to finish and checking the `.out` file.

### bhist — job history

```bash
bhist -a                        # compact list of all your past jobs
bhist -t -T .-2,                # jobs from the last 2 days
bhist -l <jobid>                # detailed report for one job (CPU time, memory, termination reason)
```

`bhist -l` shows:
- Submission time, dispatch time, start time, end time
- CPU time consumed vs. run time (ratio ≈ cores actually used)
- Peak memory usage (`MAX_MEM`)
- Termination reason (see below)

### bacct — accounting summary for completed jobs

```bash
bacct <jobid>            # CPU time, wait time, turnaround, hog factor, expansion factor
```

### showstart — estimate when a pending job will start

```bash
showstart <jobid>
```

### bkill — cancel jobs

```bash
bkill <jobid>            # graceful kill (SIGTERM then SIGKILL)
bkill -s SIGTERM <jobid> # send specific signal
bkill 0                  # kill ALL your jobs
```

---

## Cluster Status Commands

### nodestat — per-node status

```bash
nodestat hpc             # all nodes: status, cores used/free
nodestat -F hpc          # include CPU model, memory, features
nodestat -g gpuv100      # GPU details for a specific queue
nodestat -G gpuv100      # more verbose GPU info (model, count)
nodestat -g gpua100
```

### classstat — cluster-wide overview

```bash
classstat hpc            # total/used/available cores, pending jobs
```

### bqueues — queue configuration

```bash
bqueues                  # list all queues, status, job counts
bqueues gpua100          # specific queue details
```

---

## Job Failure Debugging

### Step 1: Check exit reason

```bash
bjobs -l <jobid>         # shows termination reason if still in system
bhist -l <jobid>         # shows termination reason for completed jobs
```

### LSF termination reasons

| Reason | Code | Meaning |
|--------|------|---------|
| `TERM_MEMLIMIT` | 16 | Job killed — exceeded memory limit (`-R rusage[mem=...]` or `-M`) |
| `TERM_RUNLIMIT` | 5 | Job killed — exceeded walltime (`-W`) — exit code 140 |
| `TERM_CPULIMIT` | 12 | Job killed — exceeded CPU time limit |
| `TERM_OWNER` | 14 | Killed by user (`bkill`) |
| `TERM_ADMIN` | 15 | Killed by admin |
| `TERM_FORCE_OWNER` | 8 | Killed by owner with no cleanup time |
| `TERM_PREEMPT` | 1 | Killed due to preemption |
| `TERM_LOAD` | 3 | Killed due to load threshold |
| `TERM_CWD_NOTEXIST` | 25 | Working directory doesn't exist — check `#BSUB -cwd` |
| `TERM_UNKNOWN` | 0 | LSF can't determine reason — check your `.err` log |

### Step 2: Read the output logs

```bash
# While job is running:
bpeek -f <jobid>

# After job finishes (output files from #BSUB -o and -e):
cat logs/train_<jobid>.out
cat logs/train_<jobid>.err
tail -100 logs/train_<jobid>.err
```

### Step 3: Common failure causes and fixes

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `TERM_MEMLIMIT` | OOM on CPU RAM | Increase `#BSUB -R "rusage[mem=...]"` |
| CUDA OOM in log | GPU VRAM exhausted | Reduce batch size, or use larger GPU (`gpua100` 80GB) |
| `TERM_RUNLIMIT` | Job too slow | Increase `-W`, optimize code, use more cores/GPUs |
| `TERM_CWD_NOTEXIST` | Log dir missing | Add `mkdir -p logs` before `bsub` |
| Job EXIT immediately | Script error | Check `.err` file; test interactively first |
| Job stuck in PEND | No resources free | `bjobs -l` for PENDING REASONS; try different queue/GPU |
| Low CPU EFFIC in bstat | Too many cores requested | Reduce `-n`, or fix parallelism |
| GPU memory used, compute=0 | Data loading bottleneck | Use more DataLoader workers, pre-fetch to SSD |

### Step 4: Test interactively before submitting batch

```bash
# Get interactive session
bsub -Is -q gpua100 -n 4 -R "rusage[mem=16GB]" -R "span[hosts=1]" \
    -gpu "num=1:mode=exclusive_process" -W 1:00 bash

# Validate GPU is visible
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Run a 1-step smoke test
python train.py --max-steps 5
```

---

## GPU Memory Profiling

### PyTorch built-in memory stats

```python
import torch

# After a forward pass:
print(torch.cuda.memory_allocated() / 1e9, "GB allocated")
print(torch.cuda.max_memory_allocated() / 1e9, "GB peak allocated")
print(torch.cuda.memory_reserved() / 1e9, "GB reserved by caching allocator")

# Reset peak stats between runs
torch.cuda.reset_peak_memory_stats()
```

### PyTorch memory snapshot (detailed)

```python
# At start of training
torch.cuda.memory._record_memory_history(max_entries=100000)

# After a few steps
snapshot = torch.cuda.memory._snapshot()
import pickle
with open("mem_snapshot.pkl", "wb") as f:
    pickle.dump(snapshot, f)

# Stop recording
torch.cuda.memory._record_memory_history(enabled=None)
```

Upload `mem_snapshot.pkl` to https://pytorch.org/memory_viz to get an interactive flame graph.

### nvidia-smi during training

```bash
# Quick check while training runs:
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
```

---

## TensorBoard and W&B via SSH Tunnel

Both tools need a port forwarded from the compute node to your laptop.

### TensorBoard

**On the compute node (in your job or interactive session):**

```bash
tensorboard --logdir=$OUTPUT_DIR/logs --port=6006 --host=$HOSTNAME &
```

**On your local machine:**

```bash
ssh -N -L 6006:COMPUTE_NODE_HOSTNAME:6006 USERNAME@login.hpc.dtu.dk
# Then open: http://localhost:6006
```

To combine with an existing SSH session (if you already have a tunnel to the login node):

```bash
# Two-hop manual tunnel:
ssh -N -L 6006:localhost:6006 USERNAME@login.hpc.dtu.dk -t \
    ssh -N -L 6006:localhost:6006 COMPUTE_NODE_HOSTNAME
```

### Weights & Biases (W&B)

W&B does **not** require SSH tunneling — it communicates outbound from the cluster to `wandb.ai` servers over HTTPS (port 443). As long as the cluster has outbound internet (DTU HPC does), W&B just works in your job script:

```bash
# In your job script or interactive session:
export WANDB_API_KEY=<your-key>
# Or run once: wandb login

python train.py  # wandb.init() will sync to cloud automatically
```

If outbound internet is blocked on compute nodes (check with `curl https://api.wandb.ai`), use **offline mode**:

```bash
export WANDB_MODE=offline
python train.py

# After job completes, sync from login node:
wandb sync runs/<run-dir>
```

### Port forwarding shorthand for multiple services

```bash
# Forward Jupyter (40000) + TensorBoard (6006) + W&B UI (8080) in one command:
ssh -N \
    -L 40000:COMPUTE_NODE:40000 \
    -L 6006:COMPUTE_NODE:6006 \
    USERNAME@login.hpc.dtu.dk
```

---

## Workflow Summary

1. **First time:** Set up environment + download model checkpoint
2. **Data conversion:** Submit CPU job to convert CoRal-v3 → Parquet (8-12h)
3. **Smoke test:** Interactive GPU session to verify training starts
4. **Training:** Submit GPU job (A100, 24h walltime)
5. **Evaluation:** Submit eval job after training completes
6. **Iterate:** Adjust hyperparameters based on validation loss/WER

### Interactive GPU Session (smoke test)

```bash
# Quickest: jump directly to shared A100 interactive node
a100sh

# Or request a dedicated GPU via scheduler (exclusive access):
bsub -Is -q gpua100 -n 4 -R "rusage[mem=16GB]" -R "span[hosts=1]" \
    -gpu "num=1:mode=exclusive_process" -W 1:00 bash
```
