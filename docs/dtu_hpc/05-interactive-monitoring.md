# DTU HPC — Interactive Sessions, Monitoring, and Debugging

## Interactive nodes

### Shared GPU nodes (always on, no scheduler)

Jump directly from the login node:

```bash
voltash    # 2× V100 PCIe, 16 GB each
sxm2sh     # 4× V100-SXM2 NVLink, 32 GB each
a100sh     # 2× A100 PCIe, 40 GB each  ← recommended for this project
```

These nodes are **shared** — other users may be on them. Check GPU occupancy first:

```bash
nvidia-smi
```

> For development and short tests only — not for full training runs.

### Generic interactive compute node (no GPU)

```bash
linuxsh    # lands you on a shared CPU node, dynamically assigned
```

### Exclusive GPU via scheduler (bsub -Is)

Request a dedicated GPU through LSF. This queues like a normal job but opens an interactive shell instead of running a script.

```bash
# 1 A100, 4 cores, 16 GB RAM, 1-hour session
bsub -Is -q gpua100 \
    -n 4 \
    -R "rusage[mem=16GB]" \
    -R "span[hosts=1]" \
    -gpu "num=1:mode=exclusive_process" \
    -W 1:00 \
    bash

# V100 32 GB, 2-hour session
bsub -Is -q gpuv100 \
    -n 4 \
    -R "rusage[mem=8GB]" \
    -R "span[hosts=1]" \
    -R "select[gpu32gb]" \
    -gpu "num=1:mode=exclusive_process" \
    -W 2:00 \
    bash

# L40s 48 GB
bsub -Is -q gpul40s \
    -n 4 \
    -R "rusage[mem=16GB]" \
    -R "span[hosts=1]" \
    -gpu "num=1:mode=exclusive_process" \
    -W 1:00 \
    bash
```

> The session is killed automatically when `-W` walltime expires. Default is 15 minutes if `-W` is omitted.

---

## Running Jupyter on a GPU node

GPU nodes are firewalled — a two-hop SSH tunnel is required: **laptop → login node → compute node**.

### Step 1: Get an interactive GPU session and note the hostname

```bash
# Quickest: shared A100 node
a100sh
hostname   # e.g. n-62-20-1

# Or exclusive via scheduler:
bsub -Is -q gpua100 -n 4 -R "rusage[mem=16GB]" -R "span[hosts=1]" \
    -gpu "num=1:mode=exclusive_process" -W 2:00 bash
hostname   # e.g. gpunode042
```

### Step 2: Start Jupyter on the compute node

```bash
module load cuda/11.7
source /work3/$USER/danish_asr/conda/etc/profile.d/conda.sh
conda activate danish_asr

# Bind to the node hostname, pick a port (40000–49999)
jupyter notebook --no-browser --port=40000 --ip=$HOSTNAME
# or JupyterLab:
jupyter lab --no-browser --port=40000 --ip=$HOSTNAME
```

Jupyter prints a URL like:
```
http://gpunode042:40000/?token=401bb4a3a4faeafd2fd948a137b0f6857ba4aa6e6fc47b7d
```

Note the hostname, port, and token.

### Step 3: SSH tunnel from your local machine

```bash
# Replace HOSTNAME with compute node name, USERNAME with your DTU ID
ssh -N -L 40000:HOSTNAME:40000 USERNAME@login.hpc.dtu.dk

# Example:
ssh -N -L 40000:gpunode042:40000 s123456@login.hpc.dtu.dk

# Run in background:
ssh -Nf -L 40000:gpunode042:40000 s123456@login.hpc.dtu.dk

# Via gbar login node:
ssh -N -L 40000:n-62-20-1:40000 s123456@login.gbar.dtu.dk
```

**PuTTY (Windows):** Connection > SSH > Tunnels > Source port: `40000`, Destination: `HOSTNAME:40000`, type Local. Connect to `login.hpc.dtu.dk`.

### Step 4: Open in browser

Navigate to `http://127.0.0.1:40000/?token=<your-token>`.

---

## GPU monitoring

### nvidia-smi

```bash
# Snapshot of all GPUs
nvidia-smi

# Compact view, refresh every 2 seconds
watch -n 2 nvidia-smi

# Query specific metrics (good for brief checks)
nvidia-smi --query-gpu=index,utilization.gpu,memory.total,memory.used,memory.free \
    --format=csv,noheader

# Show which processes are using GPU memory
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# Force one specific GPU
CUDA_VISIBLE_DEVICES=0 python train.py
```

> **Warning:** Do not run `nvidia-smi -l` (loop) or `watch nvidia-smi` for extended periods on shared interactive nodes — it wastes resources. Use it for spot checks only.

### bnvtop (batch jobs only)

```bash
bnvtop <JOBID>
```

DTU wrapper around `nvtop` — shows a graphical terminal view of GPU memory and compute utilization for your running batch job. **Only works for batch jobs**, not interactive sessions. For spot checks only (not continuous monitoring).

What to look for:
- Zero memory AND zero compute → GPU not being used at all
- Memory used, compute near zero → data loading bottleneck, GPU idle
- Spiky/bursty compute → CPU is the bottleneck feeding data

### Log GPU stats inside a batch job

Add to your job script to capture a continuous GPU log:

```bash
nvidia-smi --query-gpu=index,timestamp,utilization.gpu,memory.total,memory.used \
    --format=csv -l 10 > logs/gpu_${LSB_JOBID}.csv &
NVPID=$!

# ... training ...

kill $NVPID
```

---

## Job monitoring commands

### bstat (DTU custom — preferred)

```bash
bstat                      # all your jobs, compact
bstat <jobid>              # specific job
bstat -C <jobid>           # CPU efficiency (EFFIC %; aim for ~100%)
bstat -M <jobid>           # memory: MEM / MAX / AVG / LIM per host
bstat -u <username>        # filter by user
bstat -q <queue>           # filter by queue
```

CPU efficiency well below 100% → too many cores requested, or parallelism issue. Memory MAX much lower than LIM → over-requested RAM.

### bjobs (standard LSF)

```bash
bjobs                      # all your jobs
bjobs <jobid>              # specific job
bjobs -l <jobid>           # verbose: walltime, memlimit, PENDING REASONS
bjobs -u all               # all users' jobs (to see cluster load)
```

Job states:

| STAT | Meaning |
|------|---------|
| `PEND` | Queued, waiting for resources |
| `RUN` | Currently executing |
| `DONE` | Completed successfully |
| `EXIT` | Failed or killed |
| `SSUSP` | Suspended by system |
| `USUSP` | Suspended by user |

For pending jobs, `bjobs -l <jobid>` shows a **PENDING REASONS** section explaining exactly why the job hasn't started.

### bpeek — live job output

```bash
bpeek <jobid>              # print current stdout+stderr snapshot
bpeek -f <jobid>           # follow mode (streams continuously, like tail -f)
```

Primary debugging tool for batch jobs.

### bhist — job history

```bash
bhist -a                   # compact list of all past jobs
bhist -t -T .-2,           # jobs from last 2 days
bhist -l <jobid>           # detailed: timing, CPU time, peak memory, termination reason
```

### bacct — accounting summary

```bash
bacct <jobid>              # CPU time, wait, turnaround, hog factor, expansion factor
```

### showstart — estimated start time

```bash
showstart <jobid>
```

### bkill

```bash
bkill <jobid>              # graceful kill (SIGTERM → SIGKILL)
bkill -s SIGTERM <jobid>   # send specific signal
bkill 0                    # kill ALL your jobs
```

---

## Cluster status commands

```bash
nodestat hpc               # all CPU nodes: status, cores used/free
nodestat -F hpc            # add CPU model, memory, feature flags
nodestat -g gpua100        # GPU nodes: GPU summary
nodestat -G gpua100        # GPU nodes: verbose (model, count)
classstat hpc              # cluster-wide: total/used/available cores, pending
bqueues                    # all queues: name, status, job counts
bqueues gpua100            # specific queue
```

---

## Debugging failed jobs

### Step 1: Check the termination reason

```bash
bjobs -l <jobid>           # if job is still in the system
bhist -l <jobid>           # for completed jobs
```

LSF termination reasons:

| Reason | Code | Meaning |
|--------|------|---------|
| `TERM_MEMLIMIT` | 16 | Exceeded memory limit — increase `rusage[mem=...]` or `-M` |
| `TERM_RUNLIMIT` | 5 | Exceeded walltime — increase `-W`, exit code 140 |
| `TERM_CPULIMIT` | 12 | Exceeded CPU time limit |
| `TERM_OWNER` | 14 | Killed by user (`bkill`) |
| `TERM_ADMIN` | 15 | Killed by admin |
| `TERM_PREEMPT` | 1 | Preempted |
| `TERM_CWD_NOTEXIST` | 25 | Working directory doesn't exist — create log dir before `bsub` |
| `TERM_UNKNOWN` | 0 | LSF can't determine reason — check your `.err` file |

### Step 2: Read the logs

```bash
# While running:
bpeek -f <jobid>

# After completion:
cat logs/train_<jobid>.err
tail -100 logs/train_<jobid>.out
```

### Step 3: Common failures and fixes

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `TERM_MEMLIMIT` | CPU OOM | Increase `#BSUB -R "rusage[mem=...]"` |
| CUDA OOM in log | GPU VRAM exhausted | Reduce batch size, or use larger GPU |
| `TERM_RUNLIMIT` | Job too slow | Increase `-W`, optimize code |
| `TERM_CWD_NOTEXIST` | Log dir missing | `mkdir -p logs` before submitting |
| Job exits immediately | Script error | Check `.err` file; test interactively first |
| Job stuck in PEND | No free resources | `bjobs -l` for PENDING REASONS; try different queue |
| Low CPU EFFIC | Too many cores | Reduce `-n`, or fix parallelism |
| Memory used, compute≈0 | Data loading bottleneck | More DataLoader workers, pin memory |

### Step 4: Interactive smoke test before batch submission

```bash
# Get interactive session
bsub -Is -q gpua100 -n 4 -R "rusage[mem=16GB]" -R "span[hosts=1]" \
    -gpu "num=1:mode=exclusive_process" -W 1:00 bash

# Validate GPU
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Run a short smoke test
python -u train.py --max-steps 5
```

---

## GPU memory profiling (PyTorch)

```python
import torch

# After a forward pass:
print(f"Allocated:  {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Peak:       {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
print(f"Reserved:   {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Reset peak tracking between experiments:
torch.cuda.reset_peak_memory_stats()
```

### Full memory snapshot (flame graph)

```python
# At start of training:
torch.cuda.memory._record_memory_history(max_entries=100000)

# After a few steps:
import pickle
snapshot = torch.cuda.memory._snapshot()
with open("mem_snapshot.pkl", "wb") as f:
    pickle.dump(snapshot, f)

torch.cuda.memory._record_memory_history(enabled=None)
```

Upload `mem_snapshot.pkl` to https://pytorch.org/memory_viz for an interactive flame graph.

---

## TensorBoard and W&B via SSH tunnel

### TensorBoard

On compute node:

```bash
tensorboard --logdir=$OUTPUT_DIR/logs --port=6006 --host=$HOSTNAME &
```

On your local machine:

```bash
ssh -N -L 6006:COMPUTE_NODE_HOSTNAME:6006 USERNAME@login.hpc.dtu.dk
# Open: http://localhost:6006
```

### Weights & Biases (W&B)

W&B communicates **outbound** over HTTPS — no tunnel needed. It works directly from compute nodes if outbound internet is available (DTU HPC has it):

```bash
export WANDB_API_KEY=<your-key>
python train.py    # wandb.init() syncs to cloud automatically
```

If outbound is blocked, use offline mode:

```bash
export WANDB_MODE=offline
python train.py

# Sync after job from login node:
wandb sync runs/<run-dir>
```

### Forwarding multiple ports at once

```bash
ssh -N \
    -L 40000:COMPUTE_NODE:40000 \
    -L 6006:COMPUTE_NODE:6006 \
    USERNAME@login.hpc.dtu.dk
```
