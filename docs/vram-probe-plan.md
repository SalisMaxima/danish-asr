# VRAM Probe Plan: OmniASR CTC v2 Model Scaling on A100

## Goal

Determine which OmniASR CTC v2 model sizes can train on the `gpua100` queue, which contains both A100-40GB and A100-80GB nodes. Find the largest viable batch configuration for each model on each GPU variant.

## Background

The 300M model trains comfortably at `max_num_elements=3,840,000` using ~20-25 GB VRAM. The roadmap targets 1B next, then 3B if warranted. Before committing to long training runs, we need to know what fits.

The `gpua100` queue has two GPU variants (confirmed via `bqueues -l gpua100`):

| Node range | GPU | VRAM |
|------------|-----|------|
| n-62-12-* | A100-PCIE | 40 GB |
| n-62-18-* | A100-PCIE | 80 GB |

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

### Phase 1: 1B probes (any A100 node)

Submit all three — they can run on either 40GB or 80GB nodes:

```bash
bsub < scripts/hpc/1b/06a_vram_probe_1b.sh
bsub < scripts/hpc/1b/06b_vram_probe_1b_small.sh
bsub < scripts/hpc/1b/06c_vram_probe_1b_tiny.sh
```

Note which node each job lands on (`bjobs -l <jobid>`) to know if the result applies to 40GB or 80GB.

### Phase 2: 3B probes (target 80GB nodes)

The 3B model is ~12 GiB just for weights. Training on 40GB is unlikely but possible at minimal batch. On 80GB it should fit comfortably.

The 3B scripts include `#BSUB -R "select[gpu80gb]"` to target 80GB nodes:

```bash
# 3B on 80GB node — conservative batch
bsub < scripts/hpc/3b/06d_vram_probe_3b.sh

# 3B on 80GB node — minimal batch (only if above OOMs)
bsub < scripts/hpc/3b/06e_vram_probe_3b_tiny.sh
```

## How to Check Results

- **Live output:** `bpeek -f <jobid>`
- **Which node:** `bjobs -l <jobid>` — check the host name to know 40GB vs 80GB
- **After completion:** check `/work3/$USER/logs/lsf/vram_probe_<jobid>.err` for CUDA OOM
- **W&B:** filter by tag `vram-probe`
- **LSF status:** `TERM_MEMLIMIT` in `bhist -l <jobid>` means host RAM OOM (increase `rusage[mem=...]`)

## OOM Indicators

| Symptom | Meaning |
|---------|---------|
| `CUDA out of memory` in stderr | GPU VRAM exceeded — try smaller config or larger GPU |
| `TERM_MEMLIMIT` from LSF | Host RAM exceeded — increase `-R "rusage[mem=...]"` |
| Job completes 500 steps + validation | Config fits — this is the answer |

## Results Table

Fill in after running probes. Record which GPU variant (40GB/80GB) each job ran on.

| Config | Model | max_num_elements | grad_accum | GPU (40/80) | Result | Peak VRAM | Notes |
|--------|-------|-----------------|------------|-------------|--------|-----------|-------|
| `vram-probe-1b` | 1B | 3,840,000 | 4 | | | | |
| `vram-probe-1b-small` | 1B | 1,920,000 | 8 | | | | |
| `vram-probe-1b-tiny` | 1B | 960,000 | 16 | | | | |
| `vram-probe-3b` (80GB) | 3B | 1,920,000 | 8 | 80 | | | |
| `vram-probe-3b-tiny` (80GB) | 3B | 960,000 | 16 | 80 | | | |
| `vram-probe-3b-tiny` (40GB) | 3B | 960,000 | 16 | 40 | | | |

## Decision Criteria

- If 1B fits at `max_num_elements=3,840,000` on 40GB → use same config as 300M, just swap model name.
- If 1B only fits at a smaller batch → increase `grad_accumulation` proportionally to preserve effective batch size.
- If 3B fits on 80GB → viable for training, but jobs must target 80GB nodes with `-m` flag.
- If 3B fits on 40GB at minimal batch → viable on any node but slower throughput.
- If 3B doesn't fit on 80GB → skip 3B entirely.

## Prerequisites for 3B probes

The 3B model checkpoint (~12 GiB) must be downloaded before submitting 3B jobs:

```bash
source scripts/hpc/env.sh && setup_omniasr
python -c "from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline; ASRInferencePipeline(model_card='omniASR_CTC_3B_v2')"
```
