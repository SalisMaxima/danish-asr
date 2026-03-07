# DTU HPC — LSF Job Submission

## Submitting a job

```bash
bsub < job.sh
# Returns: Job <44951> is submitted to queue <gpua100>
```

---

## Job script structure

```bash
#!/bin/sh
### --- Job directives (#BSUB) ---
#BSUB -J MyJob                       # Job name (default: NONAME)
#BSUB -q gpua100                     # Queue name
#BSUB -n 4                           # Total CPU cores
#BSUB -R "span[hosts=1]"             # All cores on one node
#BSUB -R "rusage[mem=16GB]"          # Memory per core (16 GB × 4 = 64 GB total)
#BSUB -M 18GB                        # Hard per-process memory ceiling
#BSUB -gpu "num=1:mode=exclusive_process"  # 1 GPU, exclusive
#BSUB -W 24:00                       # Walltime hh:mm (default: 0:15!)
#BSUB -o logs/train_%J.out           # Stdout (%J = job ID)
#BSUB -e logs/train_%J.err           # Stderr
#BSUB -B                             # Email at job start
#BSUB -N                             # Email at job end
#BSUB -Ne                            # Email only on failure
#BSUB -u s123456@student.dtu.dk      # Email address

### --- Job body ---
module purge
module load cuda/11.7

source /work3/$USER/danish_asr/conda/etc/profile.d/conda.sh
conda activate danish_asr

python -u train.py
```

---

## All `#BSUB` options

### Identification

| Option | Effect |
|--------|--------|
| `-J name` | Job name; default `NONAME`; also used for arrays: `-J My_array[1-25]` |
| `-q queue` | Target queue |

### CPU and node allocation

| Option | Effect |
|--------|--------|
| `-n N` | Total CPU cores |
| `-R "span[hosts=1]"` | All N cores on one node (required for shared-memory / OpenMP) |
| `-R "span[ptile=N]"` | N cores per node, distributed (MPI multi-node) |
| `-R "select[model == XeonE5_2660v3]"` | Target specific CPU model |
| `-R "select[avx2]"` | Require AVX2 instruction support |

Check CPU models: `nodestat -F hpc`

### Memory

| Option | Effect |
|--------|--------|
| `-R "rusage[mem=X]"` | Memory **per core** reserved for scheduling. No space (use `4GB`, not `4 GB`). |
| `-M limit` | Hard per-process limit — job killed if exceeded. Set slightly above rusage. |

Default when omitted: `1024MB` per core.

> Memory is **per core**. For 4 cores needing 64 GB total: `rusage[mem=16GB]` + `-M 18GB`.

### Walltime

| Option | Effect |
|--------|--------|
| `-W hh:mm` | Wall-clock limit. **Default is 0:15 (15 minutes)**. Always set this. |

### Output files

| Option | Effect |
|--------|--------|
| `-o filename` | Append stdout to file |
| `-oo filename` | Overwrite stdout file |
| `-e filename` | Append stderr to file |
| `%J` in filename | Expands to job ID |
| `%I` in filename | Expands to array index (job arrays) |

### Notifications

| Option | Effect |
|--------|--------|
| `-u email` | Email address |
| `-B` | Email at job start |
| `-N` | Email at completion |
| `-Ne` | Email only on failure |

---

## Queues and limits

### GPU queues (all have 24h max walltime)

| Queue | GPU | Nodes | GPUs/node | VRAM |
|-------|-----|-------|-----------|------|
| `gpua100` | A100 PCIe | 4 | 2 | 40 GB |
| `gpua100` | A100 PCIe | 6 | 2 | 80 GB |
| `gpul40s` | L40s PCIe | 6 | 2 | 48 GB |
| `gpuv100` | V100 PCIe | 6 | 2 | 16 GB |
| `gpuv100` | V100 PCIe | 8 | 2 | 32 GB |
| `gpuv100` | V100 NVLink SXM2 | 3 | 4 | 32 GB |
| `gpua10` | A10 PCIe | 1 | 2 | 24 GB |
| `gpua40` | A40 NVLink | 1 | 2 | 48 GB |
| `gpuamd` | AMD Radeon Instinct MI50 | 1 | 2 | 16 GB |

Retired (do not use): `gputitanxpascal`, `gpuk80`, `gpuk40`

### General CPU queues

| Queue | Max walltime | Default walltime | Max cores/job |
|-------|-------------|-----------------|--------------|
| `hpc` | 72 hours | **15 minutes** | 100 |
| ThinLinc app-node | 48 hours | 48 hours | 1 |

---

## GPU resource requests

```bash
# Single GPU, exclusive mode (always use exclusive_process)
#BSUB -gpu "num=1:mode=exclusive_process"

# Two GPUs
#BSUB -gpu "num=2:mode=exclusive_process"

# Select 32 GB V100 specifically (within gpuv100 queue)
#BSUB -R "select[gpu32gb]"

# Select 80 GB A100 specifically
#BSUB -R "select[gpu80gb]"

# Select NVLink SXM2 nodes (V100 32GB NVLink)
#BSUB -R "select[sxm2]"
```

> Always use `mode=exclusive_process` to prevent GPU memory conflicts with other users.

> A100 GPUs require CUDA 11.0 or newer — use `module load cuda/11.7` or higher.

---

## Job arrays

Run many independent jobs with one submission:

```bash
#BSUB -J My_array[1-25]    # 25 jobs, indexed 1–25
#BSUB -o Output_%J_%I.out  # %I = array index in directives

# The array index is available in the shell as:
echo $LSB_JOBINDEX
./my_program --input data_${LSB_JOBINDEX}.txt
```

### Array range syntax

| Syntax | Meaning |
|--------|---------|
| `[1-25]` | Jobs 1 through 25 |
| `[1,23,45-67]` | Job 1, job 23, jobs 45–67 |
| `[1-21:2]` | Odd jobs: 1, 3, 5, ... 21 |
| `[1-20]%5` | 20 jobs, max 5 running simultaneously |

### Managing arrays

```bash
bjobs <jobid>              # status of entire array
bjobs <jobid>[3]           # element 3 only
bkill <jobid>              # kill entire array
bkill <jobid>[3]           # kill one element
bkill <jobid>[1-5,212]     # kill selected elements
```

Jobs in an array are completely independent — do not rely on execution order.

---

## Job dependencies

```bash
#BSUB -w "done(44951)"              # start after job 44951 completes successfully
#BSUB -w "ended(44951)"             # start when job ends (DONE or EXIT)
#BSUB -w "done(44951) && done(44952)"  # both must complete
#BSUB -w "done(44951) || done(44952)"  # either completes
#BSUB -w "done(44951[3])"           # specific array element
```

Dependency conditions:

| Condition | Meaning |
|-----------|---------|
| `done(id)` | Completed successfully (DONE state) |
| `ended(id)` | In EXIT or DONE state |
| `exit(id)` | Exited (failed) |
| `exit(id, code)` | Exited with specific exit code |
| `started(id)` | In RUN, DONE, EXIT, or SUSP state |

```bash
bjdepinfo <jobid>      # what jobs this depends on
bjdepinfo -c <jobid>   # what jobs depend on this job
```

---

## LSF environment variables (available inside running jobs)

| Variable | Value |
|----------|-------|
| `$LSB_JOBID` | Job ID |
| `$LSB_JOBINDEX` | Array index (job arrays only) |
| `$LSB_DJOB_NUMPROC` | Total allocated CPU cores |

Usage:

```bash
OMP_NUM_THREADS=$LSB_DJOB_NUMPROC    # OpenMP thread count
mpirun -np $LSB_DJOB_NUMPROC ./prog   # MPI process count
python3 train.py > out_$LSB_JOBID.txt  # per-job output file
```

---

## Complete example scripts

### Single GPU training (A100)

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
#BSUB -N
#BSUB -u s123456@student.dtu.dk

module purge
module load cuda/11.7

export HF_HOME=/work3/$USER/danish_asr/hf_cache
export FAIRSEQ2_CACHE_DIR=/work3/$USER/danish_asr/fairseq2_cache
export TMPDIR=/work3/$USER/tmp
mkdir -p $TMPDIR logs

# Log GPU stats in background
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.free \
    --format=csv -l 10 > logs/gpu_${LSB_JOBID}.csv &
NVPID=$!

source /work3/$USER/danish_asr/conda/etc/profile.d/conda.sh
conda activate danish_asr

OUTPUT_DIR="/work3/$USER/danish_asr/outputs/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

python -u train.py --output-dir $OUTPUT_DIR

kill $NVPID
echo "Done. Output: $OUTPUT_DIR"
```

### Dual GPU training (A100)

```bash
#!/bin/sh
#BSUB -J danish_asr_2gpu
#BSUB -q gpua100
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -o logs/train_%J.out
#BSUB -e logs/train_%J.err

module purge
module load cuda/11.7

source /work3/$USER/danish_asr/conda/etc/profile.d/conda.sh
conda activate danish_asr

python -u train.py --devices 2
```

### CPU-only data conversion

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

python -u scripts/convert_coral_to_parquet.py \
    --subset all \
    --output-dir /work3/$USER/danish_asr/data/parquet \
    --cache-dir /work3/$USER/danish_asr/hf_cache

echo "Conversion complete."
```

### V100 32GB specifically

```bash
#!/bin/sh
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[gpu32gb]"
#BSUB -W 24:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o logs/train_%J.out
#BSUB -e logs/train_%J.err

module purge
module load cuda/11.7
source /work3/$USER/danish_asr/conda/etc/profile.d/conda.sh
conda activate danish_asr
python -u train.py
```

### Multi-node MPI (for reference)

```bash
#BSUB -n 32
#BSUB -R "span[ptile=8]"    # 8 cores/node → 4 nodes

module load mpi/4.1.4-gcc-12.2.0-binutils-2.39
mpirun -np $LSB_DJOB_NUMPROC ./my_mpi_program
```

---

## Checking queue and node availability

```bash
bqueues                    # all queues: name, status, job counts
bqueues gpua100            # specific queue
nodestat -g gpua100        # GPU nodes: free/used GPUs
nodestat -G gpua100        # verbose GPU info (model, count, memory)
nodestat -F hpc            # all CPU nodes: model, cores, memory, features
classstat hpc              # cluster-wide totals: used/free/pending
```
