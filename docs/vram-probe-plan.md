# VRAM Probe Plan: OmniASR CTC v2 Model Scaling on A100-40GB

## Goal

Determine which OmniASR CTC v2 model sizes can train on a single A100-PCIE-40GB GPU, and find the largest viable batch configuration for each.

## Background

The 300M model trains comfortably at `max_num_elements=3,840,000` using ~20-25 GB VRAM. The roadmap targets 1B next, then 3B if warranted. Before committing to long training runs, we need to know what fits.

| Model | Params | Download Size | Est. Training VRAM |
|-------|--------|---------------|-------------------|
| `omniASR_CTC_300M_v2` | 325M | 1.3 GiB | ~20-25 GB (confirmed) |
| `omniASR_CTC_1B_v2` | 975M | 3.7 GiB | ~18-22 GB (estimated) |
| `omniASR_CTC_3B_v2` | 3.1B | 12.0 GiB | ~40-50 GB (estimated) |

## Method

Run 500-step smoke tests (matching the existing smoke test config) with decreasing batch sizes until training succeeds or all options are exhausted. 500 steps is the minimum needed for one validation pass (CTC blank phase lasts ~100-500 steps). OOM typically happens within the first few steps; a successful run takes ~1-2 hours on A100.

All configs use E6 best-practice settings: `shuffle_window=1000`, `bf16`, `lr=5e-5`.

## Probe Configs

| Config | Model | max_num_elements | grad_accum | Effective batch ratio |
|--------|-------|-----------------|------------|----------------------|
| `vram-probe-1b.yaml` | 1B | 3,840,000 | 4 | 1x (same as 300M) |
| `vram-probe-1b-small.yaml` | 1B | 1,920,000 | 8 | 1x (halved + 2x accum) |
| `vram-probe-1b-tiny.yaml` | 1B | 960,000 | 16 | 1x (quartered + 4x accum) |
| `vram-probe-3b.yaml` | 3B | 1,920,000 | 8 | conservative start |
| `vram-probe-3b-tiny.yaml` | 3B | 960,000 | 16 | minimal |

## Execution Order

Submit one at a time. Move to the next only if the previous OOMs.

```bash
# --- 1B probes (optimistic → conservative) ---

# Try 1: same batch as 300M
PROBE_CONFIG=configs/fairseq2/vram-probe-1b.yaml bsub < scripts/hpc/06_vram_probe.sh

# Try 2: halved batch (only if Try 1 OOMs)
PROBE_CONFIG=configs/fairseq2/vram-probe-1b-small.yaml bsub < scripts/hpc/06_vram_probe.sh

# Try 3: quartered batch (only if Try 2 OOMs)
PROBE_CONFIG=configs/fairseq2/vram-probe-1b-tiny.yaml bsub < scripts/hpc/06_vram_probe.sh

# --- 3B probes (conservative → minimal) ---

# Try 1: halved batch
PROBE_CONFIG=configs/fairseq2/vram-probe-3b.yaml bsub < scripts/hpc/06_vram_probe.sh

# Try 2: quartered batch (only if Try 1 OOMs)
PROBE_CONFIG=configs/fairseq2/vram-probe-3b-tiny.yaml bsub < scripts/hpc/06_vram_probe.sh
```

## How to Check Results

- **Live output:** `bpeek -f <jobid>`
- **After completion:** check `/work3/$USER/logs/lsf/vram_probe_<jobid>.err` for CUDA OOM
- **W&B:** filter by tag `vram-probe`
- **LSF status:** `TERM_MEMLIMIT` in `bhist -l <jobid>` means host RAM OOM (increase `rusage[mem=...]`)

## OOM Indicators

| Symptom | Meaning |
|---------|---------|
| `CUDA out of memory` in stderr | GPU VRAM exceeded — try smaller config |
| `TERM_MEMLIMIT` from LSF | Host RAM exceeded — increase `-R "rusage[mem=...]"` |
| Job completes 500 steps + validation | Config fits — this is the answer |

## Results Table

Fill in after running probes:

| Config | Model | max_num_elements | grad_accum | Result | Peak VRAM | Notes |
|--------|-------|-----------------|------------|--------|-----------|-------|
| `vram-probe-1b` | 1B | 3,840,000 | 4 | | | |
| `vram-probe-1b-small` | 1B | 1,920,000 | 8 | | | |
| `vram-probe-1b-tiny` | 1B | 960,000 | 16 | | | |
| `vram-probe-3b` | 3B | 1,920,000 | 8 | | | |
| `vram-probe-3b-tiny` | 3B | 960,000 | 16 | | | |

## Decision Criteria

- If 1B fits at `max_num_elements=3,840,000` → use same config as 300M, just swap model name.
- If 1B only fits at a smaller batch → increase `grad_accumulation` proportionally to preserve effective batch size.
- If 3B fits at any batch size → it's viable on A100-40GB but will train slower.
- If 3B doesn't fit at `960,000` → it requires A100-80GB or multi-GPU (out of scope for now).
