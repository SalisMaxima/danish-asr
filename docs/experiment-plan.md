# Short-Term Experiment Plan

Based on the first full training runs (effortless-wildflower-13, lively-surf-16, warm-frost-17)
observed at step ~10k with val/WER ~47–49% and curves still clearly descending.

**Target:** CER < 6% on CoRal-v3 read-aloud, CER < 14% on conversation
(matching `roest-wav2vec2-315m-v3` — the best published CTC result at this model scale).

---

## Current Baseline

| Run | Steps | LR | Scheduler | grad_accum | Encoder freeze | Shuffle window | Final WER | Status |
|-----|-------|----|-----------|------------|----------------|----------------|-----------|--------|
| effortless-wildflower-13 | 5k | 1e-5 | none | 4 | 0 | 1 | ~49% | stopped early |
| warm-frost-17 | 20k | 1e-5 | tri_stage (10% warmup → cosine) | 4 | 0 | 1 | ~47.5% | completed |
| autumn-dawn (E2) | 30k | 3e-5 | tri_stage (10% warmup → cosine) | 4 | 0 | 1 | **38.6%** | completed ✓ |
| wobbly-pond (E3) | 30k | 5e-5 | tri_stage (10% warmup → cosine) | 4 | 0 | 1 | **35.8%** | completed ✓ |
| rose-dream (E5) | 40k | 3e-5 | tri_stage (10% warmup → cosine) | 4 | 2k | 1 | **37.2%** | completed ✓ |
| bumbling-dawn (E6) | 50k | 5e-5 | tri_stage (10% warmup → cosine) | 4 | 0 | **1000** | **32.7%** | completed ✓ |
| true-firefly (E7) | 53.9k | 5e-5 | tri_stage (10% warmup → cosine) | 4 | 0 | 1 | **32.6%** | crashed @ 53900 |
| **doctor-voyager (E6-1B)** | **50k** | **5e-5** | **tri_stage (10% warmup → cosine)** | **8** | **0** | **1000** | **25.2%** | **completed ✓** |

**E2 result (2026-04-02):** val/WER 38.6%, UER 15.2% at step 30k. No divergence — lr=3e-5 is safe.

**E3 result (2026-04-03):** val/WER 35.8%, UER 14.1% at step 30k. lr=5e-5 beats lr=3e-5 by 2.8pp.

**E5 result (2026-04-03):** val/WER 37.2%, UER 14.6% at step 40k. Freeze + more steps did NOT beat E3 — encoder freeze likely hurt with lr=3e-5.

**E6 result (2026-04-04):** val/WER **32.7%**, UER 12.9% at step 50k. **Key finding:** fixing shuffle_window (1→1000) dropped WER by 3pp vs E3 at same lr=5e-5. The no-shuffle setting in E2/E3/E5 caused overfitting — train/val loss gap was ~21 units; with proper shuffle it closes. Runtime: 7.7h.

**E7 result (2026-04-04):** val/WER **32.6%**, UER 12.9% at step 53900 (crashed). E7 resumed E3 (no shuffle fix) and matched E6 — suggesting more steps compensate partly for no shuffle, but the shuffle fix is cleaner. Best checkpoint at step ~53000 in `/work3/s204696/outputs/omniasr_e3/`.

**E6-1B result (2026-04-08):** val/WER **25.2%**, UER 9.97% at step 50k. **7.5pp improvement over 300M E6** (32.7% → 25.2%). Same hyperparameters as 300M E6 (lr=5e-5, shuffle1000, 50k steps), but with `omniASR_CTC_1B_v2` model, `max_num_elements=1.92M` (vs 2.56M for 300M), and `grad_accum=8` (vs 4). Runtime: 38.2h on A100-40GB. Peak reserved device memory: 96%. Best WER 25.21% at step 49k (plateau — only 0.07pp oscillation to final step). Config: `configs/fairseq2/1b/ctc-finetune-hpc-e6-1b.yaml`. Checkpoint: `/work3/s204696/outputs/omniasr_e6_1b/ws_1.f85211dd/checkpoints/step_50000/model`. W&B: doctor-voyager-51.

---

## Pre-flight Checklist (run before every experiment)

1. **Check scratch quota:** `getquota_work3.sh` — storagepool 6 must be under 200 GB hard limit
2. **Clean old outputs:** `rm -rf /work3/$USER/outputs/<old_run>/` (W&B cache no longer accumulates — checkpoint artifact uploads are disabled in `run_training.py`)
3. **Verify config has checkpoint pruning:**
   ```yaml
   regime:
     keep_last_n_checkpoints: 2
     keep_best_n_checkpoints: 1
   ```
   Without this, 30k steps × ~4 GB/checkpoint = 120 GB, which reliably hits the 200 GB quota and kills the job silently with exit code 120.

**Current quota status (2026-04-03):** 87.35 / 200 GB used on storagepool 6 → 112 GB free.
With pruning enabled, each 30k run uses ~7.4 GB (measured from E2). E3 + E5 + eval ≈ 25 GB total,
leaving ~87 GB headroom. Safe to queue all three simultaneously.

---

## Experiment Queue

Run in this order. Each changes **one variable** from the settled baseline (warm-frost-17).

### E0 — Wait for warm-frost-17 to finish ✓ DONE

Completed. Final WER ~47.5% — LR too low, steps too few. E1 and E2 submitted in parallel.

---

### E1 — More steps (extend warm-frost-17 or resubmit)

**Hypothesis:** Curves are still descending at 10k. The model needs more time, not different hyperparameters.

```yaml
regime:
  num_steps: 40_000          # double from 20k

lr_scheduler:
  name: "tri_stage"
  config:
    stage_ratio: [0.05, 0.0, 0.95]   # 5% warmup (2k steps), 95% cosine decay
    start_lr_scale: 0.01
    final_lr_scale: 0.05
```

**Expected outcome:** Lower WER floor. If the slope at step 20k is still significant, this is the fix.

---

### E2 — Higher learning rate ✓ DONE (autumn-dawn, 2026-04-02)

**Result:** val/WER **38.6%** at step 30k. No divergence. Curves still descending at termination.
Runtime 5h on A100. Config: `configs/fairseq2/300m/ctc-finetune-hpc-e2.yaml`.
Checkpoint: `/work3/s204696/outputs/omniasr_e2`
Eval: `bsub < scripts/hpc/300m/11_eval_e2.sh` (combined test), subset configs also available.

```yaml
optimizer:
  config:
    lr: 3e-5                 # 3× increase from baseline

lr_scheduler:
  name: "tri_stage"
  config:
    stage_ratio: [0.1, 0.0, 0.9]
    start_lr_scale: 0.01
    final_lr_scale: 0.05

regime:
  num_steps: 30_000
```

---

### E3 — Upstream Meta LR (5e-5), 30k steps ✓ DONE (wobbly-pond-25, 2026-04-03)

**Result:** val/WER **35.8%**, UER 14.1% at step 30k. lr=5e-5 beats lr=3e-5 (E2: 38.6%) by 2.8pp.
Config: `configs/fairseq2/300m/ctc-finetune-hpc-e3.yaml`
Checkpoint: `/work3/s204696/outputs/omniasr_e3/ws_1.88015460/checkpoints/step_30000/model`
Eval: `bsub < scripts/hpc/300m/12_eval_e3.sh`

```yaml
optimizer:
  config:
    lr: 5e-5

regime:
  num_steps: 30_000
```

---

### E4 — Larger effective batch

**Hypothesis:** Doubling grad accumulation increases effective batch size, which helps CTC training
stability (more frames per gradient update → less noise).

```yaml
trainer:
  grad_accumulation:
    num_batches: 8            # double from 4; no extra VRAM cost
```

**Note:** This halves the number of optimizer steps per wall-clock hour. Compensate by setting
`num_steps: 40_000` to maintain the same number of data passes.

---

### E5 — Combined best settings ✓ DONE (rose-dream-26, 2026-04-03)

**Result:** val/WER **37.2%**, UER 14.6% at step 40k. **Worse than E3 (35.8%)** — encoder freeze at lr=3e-5 hurt rather than helped.
Config: `configs/fairseq2/300m/ctc-finetune-hpc-e5.yaml`
Checkpoint: `/work3/s204696/outputs/omniasr_e5/ws_1.f3ca97e3/checkpoints/step_40000/model`
Eval: `bsub < scripts/hpc/300m/15_eval_e5.sh`

```yaml
optimizer:
  config:
    lr: 3e-5

trainer:
  freeze_encoder_for_n_steps: 2000

regime:
  num_steps: 40_000
```

---

### E6 — Fix shuffle windows + lr=5e-5, 50k steps ✓ DONE (bumbling-dawn-28, 2026-04-04)

**Hypothesis:** shuffle_window=1 (used in all prior experiments) means examples are never shuffled
within a batch window — effectively in-order training. Fixing to 1000 should improve generalisation.

**Result:** val/WER **32.7%**, UER 12.9% at step 50k. Biggest improvement yet — 3pp better than E3.
Config: `configs/fairseq2/300m/ctc-finetune-hpc-e6.yaml`
Checkpoint: `/work3/s204696/outputs/omniasr_e6/ws_1.0bb2600b/checkpoints/step_50000/model`
Eval: `bsub < scripts/hpc/300m/16_eval_e6.sh`
Runtime: 7.72h on A100.

```yaml
optimizer:
  config:
    lr: 5e-5

dataset:
  asr_task_config:
    batch_shuffle_window: 1000
    example_shuffle_window: 1000

regime:
  num_steps: 50_000
```

---

### E7 — Resume E3 to 55k steps ✗ CRASHED (true-firefly-27, 2026-04-04)

**Hypothesis:** E3 was still converging at 30k. Resume with more steps.

**Result:** val/WER **32.6%**, UER 12.9% at step 53900 (crashed ~2k before target).
Matched E6 despite no shuffle fix — more steps compensate partially.
Config: `configs/fairseq2/300m/ctc-finetune-hpc-e7.yaml`
Output dir: `/work3/s204696/outputs/omniasr_e3/` (shared with E3, different workspace hash)
Best checkpoint: `ls /work3/s204696/outputs/omniasr_e3/` to find E7 workspace, then `step_53000/model`
Eval: update `ctc-eval-e7.yaml` with correct workspace hash, then `bsub < scripts/hpc/300m/17_eval_e7.sh`

```yaml
optimizer:
  config:
    lr: 5e-5

regime:
  num_steps: 55_000   # resumed from E3's step_30000
```

---

### E6-1B — 1B model with E6 settings ✓ DONE (doctor-voyager-51, 2026-04-08)

**Hypothesis:** Scaling from 300M to 1B should improve WER substantially with the same hyperparameters.

**Result:** val/WER **25.2%**, UER 9.97% at step 50k. 7.5pp gain over 300M. 38.2h on A100-40GB.
Config: `configs/fairseq2/1b/ctc-finetune-hpc-e6-1b.yaml`
Checkpoint: `/work3/s204696/outputs/omniasr_e6_1b/ws_1.f85211dd/checkpoints/step_50000/model`
Eval: pending (see Phase 8 below)

```yaml
model:
  name: "omniASR_CTC_1B_v2"

optimizer:
  config:
    lr: 5e-5

dataset:
  asr_task_config:
    max_num_elements: 1_920_000
    batch_shuffle_window: 1000
    example_shuffle_window: 1000

trainer:
  grad_accumulation:
    num_batches: 8

regime:
  num_steps: 50_000
```

---

## Phase 8: Model Archival + Comprehensive Evaluation

### 8A — Download & store trained model checkpoints

Download the finetuned checkpoints from HPC to local storage for archival and reproducibility.

| Model | Source (HPC) | Steps | Notes |
|-------|-------------|-------|-------|
| 300M E6 (finetuned) | `/work3/s204696/outputs/omniasr_e6/ws_1.0bb2600b/checkpoints/step_50000/model` | 50k | Best 300M run |
| 1B E6 (finetuned) | `/work3/s204696/outputs/omniasr_e6_1b/ws_1.f85211dd/checkpoints/step_50000/model` | 50k | Best 1B run |

**Action:** `scp` or `rsync` the model directories (just `model/` subdirectory, not optimizer/data_reader state) to local storage. Upload to W&B as artifacts for permanent archival.

### 8B — Comprehensive evaluation matrix

Evaluate all 4 models on the held-out test split (combined, read_aloud, conversation):

| # | Model | Type | Checkpoint | Status |
|---|-------|------|-----------|--------|
| 1 | omniASR_CTC_300M_v2 | **base (zero-shot)** | (no model.path — use pretrained) | pending |
| 2 | omniASR_CTC_1B_v2 | **base (zero-shot)** | (no model.path — use pretrained) | pending |
| 3 | omniASR_CTC_300M_v2 | **finetuned (E6, 50k)** | `/work3/.../omniasr_e6/.../step_50000/model` | pending |
| 4 | omniASR_CTC_1B_v2 | **finetuned (E6-1B, 50k)** | `/work3/.../omniasr_e6_1b/.../step_50000/model` | pending |

Each model evaluated on 3 splits: combined test, read_aloud/test, conversation/test.
That's **12 eval runs total** (4 models × 3 splits).

**Expected result format:**

| Model | Combined WER | Combined CER | Read-aloud WER | Read-aloud CER | Conversation WER | Conversation CER |
|-------|-------------|-------------|---------------|----------------|-----------------|-----------------|
| 300M base | — | — | — | — | — | — |
| 1B base | — | — | — | — | — | — |
| 300M finetuned | — | — | — | — | — | — |
| 1B finetuned | — | — | — | — | — | — |
| *whisper-large-v3 ZS* | *—* | *10.1%* | *—* | *—* | *—* | *27.5%* |
| *roest-wav2vec2-315m-v3* | *—* | *5.9%* | *—* | *—* | *—* | *13.7%* |

**Configs needed:** 12 eval YAML files (4 models × 3 splits) + corresponding BSUB scripts.

---

## Phase 9: LR Extension Phase (conditional on Phase 8 results)

**Decision gate:** If the finetuned models are still improving (val WER curves were descending at termination) or if there's a meaningful gap to the benchmark, extend training with a lower LR.

### 9A — 300M LR extension (E8)

Resume 300M E6 from step 50k with reduced LR for additional steps.

```yaml
model:
  name: "omniASR_CTC_300M_v2"

optimizer:
  config:
    lr: 1e-5             # reduced from 5e-5 → gentler refinement

lr_scheduler:
  name: "tri_stage"
  config:
    stage_ratio: [0.05, 0.0, 0.95]  # short warmup for resumed training
    start_lr_scale: 0.1              # start at 10% of peak (1e-6)
    final_lr_scale: 0.05

regime:
  num_steps: 70_000      # 20k additional steps from step 50k
```

### 9B — 1B LR extension (E8-1B)

Resume 1B E6 from step 50k with reduced LR.

```yaml
model:
  name: "omniASR_CTC_1B_v2"

optimizer:
  config:
    lr: 1e-5             # reduced from 5e-5

trainer:
  grad_accumulation:
    num_batches: 8       # same as E6-1B

regime:
  num_steps: 70_000      # 20k additional steps from step 50k
```

**Stopping criterion:** If val WER improves by <0.5pp over 10k extension steps, stop early — the model has converged.

---

## Phase 10: 3B Model Training

**Prerequisites:**
1. Phase 8 evals completed (need baseline numbers)
2. 3B VRAM probe run successfully on A100-80GB
3. `/work3` quota has headroom (~50 GB needed for 3B checkpoints)

### 10A — 3B VRAM probe

The 3B probe configs exist but were **never run** (no W&B runs with tag "3b"):
- `configs/fairseq2/3b/vram-probe-3b.yaml` — conservative: `max_num_elements=1.92M`, `grad_accum=8`
- `configs/fairseq2/3b/vram-probe-3b-tiny.yaml` — fallback: `max_num_elements=960k`, `grad_accum=16`
- Script: `scripts/hpc/3b/06d_vram_probe_3b.sh` (needs `select[gpu80gb]` — A100-80GB nodes)

Submit with: `bsub < scripts/hpc/3b/06d_vram_probe_3b.sh`
If OOM → fall back to: `bsub < scripts/hpc/3b/06e_vram_probe_3b_tiny.sh`

### 10B — 3B full training (E6-3B)

Use the batch shape from the successful probe run. Expected config:

```yaml
model:
  name: "omniASR_CTC_3B_v2"

optimizer:
  config:
    lr: 5e-5              # same as E6 — or lower if 3B is less stable

dataset:
  asr_task_config:
    max_num_elements: TBD  # from probe result
    batch_shuffle_window: 1000
    example_shuffle_window: 1000

trainer:
  grad_accumulation:
    num_batches: TBD       # from probe result
  mixed_precision:
    dtype: "torch.bfloat16"

regime:
  num_steps: 50_000
  # Queue: gpua100 with select[gpu80gb]
  # Expected runtime: ~60-80h (2.5-3 days walltime → need -W 80:00 or two 40h submissions with resume)
```

**Risk:** A100-80GB nodes are scarce on `gpua100`. Job may sit in PEND for days. Walltime max is 24h — will likely need multiple resume submissions.

**Stopping criterion:** If 3B WER ≤ 1B WER + 1%, the 3B overhead isn't justified. Report 1B as best.

---

## Decision Tree

```
warm-frost-17 (WER 47.5%) → E2 lr=3e-5 30k (WER 38.6%) ✓
    │
    ├─ E3 lr=5e-5 30k (WER 35.8%) ✓ — lr=5e-5 is better
    ├─ E5 lr=3e-5 freeze 40k (WER 37.2%) ✓ — freeze didn't help with lr=3e-5
    ├─ E6 lr=5e-5 shuffle1000 50k (WER 32.7%) ✓ — BEST: shuffle fix is key
    └─ E7 resume E3 →55k (WER 32.6% @ 53.9k, crashed) — matches E6 via more steps

Key finding: shuffle_window=1000 (E6) = biggest single improvement (3pp over E3).

1B scaling (E6-1B):
    E6-1B (doctor-voyager-51) 50k lr=5e-5 shuffle1000 (WER 25.2%) ✓
    — 7.5pp better than 300M E6. 38.2h on A100-40GB (96% VRAM).
    — Best WER 25.21% at step 49k, slight rise to 25.28% at 50k → near plateau.

Next: comprehensive eval (base + finetuned, 300M + 1B), then LR extension, then 3B.
```

---

## Evaluation Protocol

Each experiment run should be evaluated on **both subsets separately** to avoid the
read-aloud/conversation mix masking per-subset progress:

```bash
invoke train.omniasr-eval --split read_aloud/test
invoke train.omniasr-eval --split conversation/test
```

Report both CER (primary metric for CoRal-v3 comparisons) and WER.

---

## Benchmarks to Beat

| Model | Read-aloud CER | Conversation CER |
|-------|---------------|-----------------|
| whisper-large-v3 zero-shot | 10.1% | 27.5% |
| **roest-wav2vec2-315m-v3** | **5.9%** | **13.7%** |
| roest-wav2vec2-315m-v2 | 6.4% | 24.2% |

**Minimum bar:** beat whisper-large-v3 zero-shot (10.1% / 27.5%).
**Good result:** match roest-wav2vec2-315m-v3 (5.9% / 13.7%).
**Stretch:** beat roest-wav2vec2-315m-v3 — would be the best published CTC result at 300M scale.
**Moonshot (3B):** significantly beat roest-wav2vec2 with the 3B model — new SOTA for Danish CTC.

---

## GPU Budget Estimate

Actual timing from E2: **5h for 30k steps** on A100. Disk: **7.4 GB per 30k run** with pruning
(`keep_last_n_checkpoints: 2` + `keep_best_n_checkpoints: 1`).

| Experiment | Model | Steps | A100 hours | Disk | Status |
|------------|-------|-------|------------|------|--------|
| E0 (warm-frost-17) | 300M | 20k | ~3.3h | — | ✓ done |
| E2 (autumn-dawn) | 300M | 30k | 5.0h | 7.4 GB | ✓ done |
| E3 (wobbly-pond) | 300M | 30k | 4.65h | ~7 GB | ✓ done |
| E5 (rose-dream) | 300M | 40k | 6.46h | ~10 GB | ✓ done |
| E6 (bumbling-dawn) | 300M | 50k | 7.72h | ~12 GB | ✓ done |
| E7 (true-firefly) | 300M | 53.9k | ~10h (crashed) | ~13 GB | crashed |
| **E6-1B (doctor-voyager)** | **1B** | **50k** | **38.2h** | **~25 GB** | **✓ done** |
| Phase 8 evals (×12) | mixed | — | <1h each | negligible | pending |
| Phase 9 extensions (×2) | 300M+1B | +20k each | ~5h + ~15h | ~10 GB each | planned |
| Phase 10 3B probe | 3B | 500 | ~1h | ~5 GB | planned |
| Phase 10 3B full | 3B | 50k | ~60-80h (est.) | ~50 GB | planned |
