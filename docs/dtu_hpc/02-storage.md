# DTU HPC — Storage Systems

## Overview

| Path | Quota | Filesystem | Backed up | Speed | Use for |
|------|-------|-----------|-----------|-------|---------|
| `$HOME` (`/zhome/<l>/<user>/`) | **30 GB** | Dell EMC + ZFS/NFS | Yes | Moderate | Code, configs, small results |
| `/work3/<username>` | By request | BeeGFS over Infiniband | **No** | High | Datasets, checkpoints, HF cache |
| `/work1/<username>` | By request | BeeGFS over Infiniband | **No** | High | Same as /work3 |

**Pool sizes:** Home = 220 TB total. work1 = 98 TB. work3 = 142 TB.

---

## Home directory

- **Quota:** 30 GB per user — this fills quickly with conda environments and model weights
- **Path format:** `/zhome/<first-letter-of-username>/<username>/` (also `$HOME`)
- **Backup:** Hourly + daily snapshots, replicated to a secondary server, tape backup to remote DTU site
- **Windows access:** `\\home.cc.dtu.dk\<username>`
- **macOS access:** `smb://home.cc.dtu.dk/<username>`

Check usage:

```bash
getquota_zhome.sh
# Output: "You are using 12.34 GB of 30.00 GB."

# Or manually:
du -h --max-depth=1 ~
```

Request a temporary increase: email support@cc.dtu.dk. Only granted for limited durations.

> **Warning:** Exceeding the 30 GB quota kills running jobs and can prevent you from logging in.

---

## GOTCHA: /work3 NVME pool has a 350 GB hard quota

`/work3` spans multiple storage pools. The **NVME pool (pool 6)** has a **350 GB hard limit**.
When this is hit, all `write()` calls fail silently — training jobs exit with code 120 and leave
no traceback, making the root cause invisible.

Check your usage before and after every training run:

```bash
getquota_work3.sh   # look at storagepool 6 "NVMEs added in 2024"
```

**Sources of runaway disk usage:**
- fairseq2 checkpoints: ~4 GB per checkpoint × 30 steps = 120 GB without pruning
- W&B artifact cache (`/work3/$USER/wandb/cache/`): `log_artifact()` copies every uploaded `.pt` file locally before uploading — **`run_training.py` disables all checkpoint artifact uploads** to prevent this; W&B only receives metrics, config, and logs

**Mandatory mitigations for every training config:**
```yaml
regime:
  keep_last_n_checkpoints: 2    # keep only 2 most recent
  keep_best_n_checkpoints: 1    # also keep best WER checkpoint
```

**`run_training.py` checkpoint upload policy:** checkpoint artifact uploads are fully disabled.
W&B receives metrics (loss/WER/CER), the config YAML, and the log file only.
Checkpoints live exclusively on HPC scratch — retrieve them with `rsync`/`scp`.

**Emergency cleanup:**
```bash
rm -rf /work3/$USER/wandb/cache/
rm -rf /work3/$USER/wandb/run-*/
rm -rf /work3/$USER/outputs/<run_dir>/   # only if you don't need checkpoints
```

---

## /work3 scratch filesystem

Work3 (and work1) use **BeeGFS** — a distributed parallel filesystem over Infiniband, designed for high-throughput I/O with large datasets.

### Getting access

Email support@cc.dtu.dk requesting scratch space; include the reason. A directory at `/work3/<username>` will be created for you.

### Usage

```bash
getquota_work3.sh
# Example output:
# user/group     ||           size          ||    chunk files
# name     |  id  ||    used    |    hard    ||  used   |  hard
# ----------|------||------------|------------||---------|---------
# s123456 |54321 ||  158.75 GiB|  400.00 GiB||  119758 |  2000000
```

The `chunk files` column is the **inode/file-count limit** — hard limit of 2,000,000 files per user. Python packages (conda envs with many small files) can exhaust this quickly.

- Read `readme.txt` in your scratch directory on first access.
- **NOT backed up** — scratch is for intermediate/temporary data only.
- Performance degrades when the filesystem is near capacity — delete old files regularly.
- Accessible from all nodes: login nodes and compute nodes.

---

## File transfer

Use the dedicated transfer server **`transfer.gbar.dtu.dk`** for bulk transfers — it has 10 GbE and Infiniband connectivity, significantly faster than login nodes.

```bash
# Upload a directory (resumable, progress bar)
rsync -avP --progress local_data/ s123456@transfer.gbar.dtu.dk:/work3/s123456/danish_asr/data/

# Download results
rsync -avP s123456@transfer.gbar.dtu.dk:/work3/s123456/danish_asr/outputs/ ./outputs/

# Single file
scp file.tar.gz s123456@transfer.gbar.dtu.dk:/work3/s123456/

# Interactive SFTP
sftp s123456@transfer.gbar.dtu.dk
```

GUI clients (FileZilla, Cyberduck) also work — connect to `transfer.gbar.dtu.dk`.

---

## Recommended layout for this project

```
$HOME/danish_asr/                    # Git repo, configs, scripts (30 GB home — backed up)

/work3/<username>/danish_asr/
├── data/
│   ├── parquet/                     # CoRal-v3 → Parquet (omnilingual ASR input)
│   └── raw/                         # Optional: raw HF downloads
├── hf_cache/                        # HuggingFace cache (~100-200 GB)
├── fairseq2_cache/                  # fairseq2 model/asset cache
├── conda/                           # Miniconda installation (NOT in home — too large)
└── outputs/                         # Training runs, checkpoints, logs
    └── run_YYYYMMDD_HHMMSS/
```

Set these environment variables in every job script:

```bash
# In scripts/hpc/*.sh — after #BSUB directives:
export HF_HOME=/work3/$USER/danish_asr/hf_cache
export HF_DATASETS_CACHE=/work3/$USER/danish_asr/hf_cache/datasets
export FAIRSEQ2_CACHE_DIR=/work3/$USER/danish_asr/fairseq2_cache

# Prevent accidental home quota exhaustion from temp files
export TMPDIR=/work3/$USER/tmp
mkdir -p $TMPDIR
```

---

## Data lifecycle warning

When your DTU account expires, **all access is terminated immediately**. Data in `/work3` (not backed up) cannot be recovered after account closure.

Back up all important outputs to home or external storage before your account expires.
