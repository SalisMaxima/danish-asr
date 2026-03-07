# DTU HPC Documentation

Comprehensive reference for using DTU's HPC cluster with GPUs and Python.

## Contents

| File | Topic |
|------|-------|
| [01-access.md](01-access.md) | SSH, VPN, keys, ThinLinc, login nodes |
| [02-storage.md](02-storage.md) | Home, /work3, scratch — quotas, file transfer |
| [03-lsf-jobs.md](03-lsf-jobs.md) | LSF job submission, GPU queues, arrays, dependencies |
| [04-python-environment.md](04-python-environment.md) | Modules, conda, uv, PyTorch + CUDA setup |
| [05-interactive-monitoring.md](05-interactive-monitoring.md) | Interactive GPU sessions, Jupyter, monitoring, debugging |

## Quick Reference

### Cluster entry points

```bash
ssh <userid>@login.hpc.dtu.dk       # login node
linuxsh                              # interactive CPU node
a100sh                               # shared interactive A100 (40 GB)
sxm2sh                               # shared interactive V100 SXM2 (32 GB)
voltash                              # shared interactive V100 (16 GB)
```

### GPU queues

| Queue | GPU | VRAM | Max walltime |
|-------|-----|------|-------------|
| `gpua100` | A100 PCIe | 40 GB / 80 GB | 24h |
| `gpul40s` | L40s PCIe | 48 GB | 24h |
| `gpuv100` | V100 | 16 GB / 32 GB | 24h |

### Essential commands

```bash
bsub < job.sh              # submit job
bstat                      # list your jobs
bpeek -f <jobid>           # stream job output live
bkill <jobid>              # cancel job
nodestat -G gpua100        # GPU availability
showstart <jobid>          # estimated start time
```

### Storage at a glance

| Path | Quota | Backed up |
|------|-------|-----------|
| `$HOME` | 30 GB | Yes |
| `/work3/<user>` | By request | No |
| `/work1/<user>` | By request | No |

### Official links

- [DTU HPC main](https://www.hpc.dtu.dk/)
- [GPU nodes](https://www.hpc.dtu.dk/?page_id=2129)
- [GPU jobs under LSF10](https://www.hpc.dtu.dk/?page_id=2759)
- [Batch jobs guide](https://www.hpc.dtu.dk/?page_id=1416)
- [Managing jobs](https://www.hpc.dtu.dk/?page_id=1519)
- [Storage](https://www.hpc.dtu.dk/?page_id=59)
- [HPC best practice](https://www.hpc.dtu.dk/?page_id=4317)
- [DTU MLOps HPC guide](https://skaftenicki.github.io/dtu_mlops/s10_extra/high_performance_clusters/)

**Support:** support@cc.dtu.dk | Building 324, Room 280, DTU Compute
