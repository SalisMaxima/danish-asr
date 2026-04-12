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
| **usual-totem (E6-3B)** | **30k** | **5e-5** | **tri_stage (10% warmup → cosine)** | **16** | **0** | **1000** | **24.8%** | **completed ✓** |

**E2 result (2026-04-02):** val/WER 38.6%, UER 15.2% at step 30k. No divergence — lr=3e-5 is safe.

**E3 result (2026-04-03):** val/WER 35.8%, UER 14.1% at step 30k. lr=5e-5 beats lr=3e-5 by 2.8pp.

**E5 result (2026-04-03):** val/WER 37.2%, UER 14.6% at step 40k. Freeze + more steps did NOT beat E3 — encoder freeze likely hurt with lr=3e-5.

**E6 result (2026-04-04):** val/WER **32.7%**, UER 12.9% at step 50k. **Key finding:** fixing shuffle_window (1→1000) dropped WER by 3pp vs E3 at same lr=5e-5. The no-shuffle setting in E2/E3/E5 caused overfitting — train/val loss gap was ~21 units; with proper shuffle it closes. Runtime: 7.7h.

**E7 result (2026-04-04):** val/WER **32.6%**, UER 12.9% at step 53900 (crashed). E7 resumed E3 (no shuffle fix) and matched E6 — suggesting more steps compensate partly for no shuffle, but the shuffle fix is cleaner. Best checkpoint at step ~53000 in `/work3/s204696/outputs/omniasr_e3/`.

**E6-1B result (2026-04-08):** val/WER **25.2%**, UER 9.97% at step 50k. **7.5pp improvement over 300M E6** (32.7% → 25.2%). Same hyperparameters as 300M E6 (lr=5e-5, shuffle1000, 50k steps), but with `omniASR_CTC_1B_v2` model, `max_num_elements=1.92M` (vs 2.56M for 300M), and `grad_accum=8` (vs 4). Runtime: 38.2h on A100-40GB. Peak reserved device memory: 96%. Best WER 25.21% at step 49k (plateau — only 0.07pp oscillation to final step). Config: `configs/fairseq2/1b/ctc-finetune-hpc-e6-1b.yaml`. Checkpoint: `/work3/s204696/outputs/omniasr_e6_1b/ws_1.f85211dd/checkpoints/step_50000/model`. W&B: doctor-voyager-51.

**E6-3B result (2026-04-12):** val/WER **24.782%**, UER **9.654%** at step **29k**; final step 30k was effectively unchanged at WER **24.789%**, UER **9.691%**. This is a **0.42pp dev WER improvement over 1B E6** (25.20% → 24.78%), but at substantially higher cost: **59.1h** on A100-80GB versus 38.2h for 1B. The 3B run therefore clears the "slightly better than 1B" bar, but **not yet** the predeclared ">1pp improvement to justify the overhead" bar. Batch shape came from the successful tiny 3B probe: `max_num_elements=960k`, `grad_accum=16`, shuffle windows 1000. Config: `configs/fairseq2/3b/ctc-finetune-hpc-e6-3b.yaml`. Output dir: `/work3/s204696/outputs/omniasr_e6_3b`. Latest checkpoint logged to W&B: `/work3/s204696/outputs/omniasr_e6_3b/ws_1.2172dba0/checkpoints/step_30000/model/pp_00/tp_00/sdp_00.pt`. W&B: usual-totem-74.

---

## Pre-flight Checklist (run before every experiment)

1. **Check scratch quota:** `getquota_work3.sh` — storagepool 6 must be under 350 GB hard limit
2. **Clean old outputs:** `rm -rf /work3/$USER/outputs/<old_run>/` (W&B cache no longer accumulates — checkpoint artifact uploads are disabled in `run_training.py`)
3. **Verify config has checkpoint pruning:**
   ```yaml
   regime:
     keep_last_n_checkpoints: 2
     keep_best_n_checkpoints: 1
   ```
   Without this, 30k steps × ~4 GB/checkpoint = 120 GB, which can hit the 350 GB quota and kills the job silently with exit code 120.

**Current quota status (2026-04-11):** quota upgraded to 350 GB (was 200 GB). Prior usage was ~87 GB.
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

**Completed (2026-04-08):** Downloaded flat `.pt` files via SFTP (login node: `transfer.gbar.dtu.dk`), renamed them descriptively, and uploaded to W&B as artifacts via `scripts/upload_checkpoints.py`.

```bash
# Download (SFTP — flat file, not directory)
sftp s204696@transfer.gbar.dtu.dk
  lmkdir models/300m_e6_50k
  lmkdir models/1b_e6_50k
  get .../omniasr_e6/.../step_50000/model/pp_00/tp_00/sdp_00.pt models/300m_e6_50k/omniASR_CTC_300M_v2_e6_step50k.pt
  get .../omniasr_e6_1b/.../step_50000/model/pp_00/tp_00/sdp_00.pt models/1b_e6_50k/omniASR_CTC_1B_v2_e6_step50k.pt

# Upload to W&B (from local machine, ~1 Gbps)
python scripts/upload_checkpoints.py
```

W&B artifacts: `omniASR-CTC-300M-v2-e6-50k:v0`, `omniASR-CTC-1B-v2-e6-50k:v0`

### 8B — Comprehensive evaluation matrix ✓ DONE (2026-04-09)

Evaluated all 4 models × 3 splits = 12 eval runs. Jobs: 28164239–28164241 (300M base, 300M E6, 1B base), 28164242 failed (missing `family`/`arch` in 1B configs — fixed in branch `fix/1b-eval-missing-family-arch`), rerun succeeded.

**Known issues discovered:**
- `run_eval.py` originally missed wrapped WER/CER lines: fairseq2 can split `Word Error Rate` / `Character Error Rate` labels from the following `(WER): X` / `(CER): X` lines. Phase 8B WER values were extracted manually from Python logs; parser handling was fixed afterward in this PR.
- Per-subset eval (read\_aloud / conversation) did not work: `dataset_summary_path` TSV controls training data mixing, not eval data filtering. All 3 split configs evaluated on the identical full combined test set. **Only combined WERs are valid.**

**Phase 8B Results — Combined Test Split (2026-04-09):**

| Model | Type | Test WER | Test UER | Val WER (dev) |
|-------|------|----------|----------|---------------|
| omniASR_CTC_300M_v2 | base (zero-shot) | **68.18%** | 28.67% | — |
| omniASR_CTC_300M_v2 | finetuned E6 50k | **30.73%** | 11.54% | 32.74% |
| omniASR_CTC_1B_v2 | base (zero-shot) | **55.39%** | 22.90% | — |
| omniASR_CTC_1B_v2 | finetuned E6 50k | **23.43%** | 8.78% | 25.20% |

Key findings:
- Finetuning gain: 300M 68.18% → 30.73% (−37.5pp), 1B 55.39% → 23.43% (−31.96pp)
- Scale gain: finetuned 300M 30.73% → finetuned 1B 23.43% (−7.3pp)
- Val→test generalisation is good (300M: −2.0pp, 1B: −1.8pp on test vs dev)
- Per-subset breakdown requires a different eval mechanism (subset TSVs don't filter eval data)

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

## Phase 10: 3B Model Scaling

**Prerequisites:**
1. Phase 8 evals completed (done)
2. 3B VRAM probe run successfully on A100-80GB (done)
3. `/work3` quota has headroom for the 3B workspace (done for the 30k run)

### 10A — 3B VRAM probe ✓ DONE

The conservative 3B batch shape was too ambitious, but the tiny 80GB-node probe shape was viable and became the training recipe:
- `max_num_elements=960k`
- `grad_accum=16`
- queue constraint: `select[gpu80gb]`

Relevant files:
- `configs/fairseq2/3b/vram-probe-3b.yaml`
- `configs/fairseq2/3b/vram-probe-3b-tiny.yaml`
- `scripts/hpc/3b/06d_vram_probe_3b.sh`
- `scripts/hpc/3b/06e_vram_probe_3b_tiny.sh`

### 10B — 3B full training (E6-3B) ✓ DONE (usual-totem-74, 2026-04-12)

**Result:** val/WER **24.782%** and UER **9.654%** at step **29k**; final step 30k ended at WER **24.789%** / UER **9.691%**. Runtime **59.1h** on A100-80GB. Compared with 1B E6 (25.20%), 3B gains only **0.42pp** on dev despite requiring an 80GB node and ~1.55× more wall-clock time. That makes this a real but modest gain, and it looks close to plateau by 30k.

Config used:

```yaml
model:
  name: "omniASR_CTC_3B_v2"

optimizer:
  config:
    lr: 5e-5

dataset:
  asr_task_config:
    max_num_elements: 960_000
    batch_shuffle_window: 1000
    example_shuffle_window: 1000

trainer:
  grad_accumulation:
    num_batches: 16
  mixed_precision:
    dtype: "torch.bfloat16"

regime:
  num_steps: 30_000
```

Files:
- Config: `configs/fairseq2/3b/ctc-finetune-hpc-e6-3b.yaml`
- Training script: `scripts/hpc/3b/14_train_e6_3b.sh`
- Output dir: `/work3/s204696/outputs/omniasr_e6_3b`
- Workspace with latest checkpoint: `/work3/s204696/outputs/omniasr_e6_3b/ws_1.2172dba0`

### 10C — Immediate next step: evaluate 3B on test, then decide on 50k resume

**Decision gate:** Do **not** automatically spend another ~40h on 3B just because 30k finished. First run the finetuned 3B eval on the combined test split and compare it directly against the 1B finetuned result (**23.43% test WER**).

What to do next:
- Add or adapt a 3B eval config/script mirroring the 1B E6 eval flow, then run the combined test eval for the finished 30k 3B checkpoint/workspace.
- If 3B test WER beats 1B by **>1pp**, resume to 50k using `configs/fairseq2/3b/ctc-finetune-hpc-e6-3b-50k.yaml` and `scripts/hpc/3b/15_train_e6_3b_50k_resume.sh`.
- If 3B test WER is within **~1pp** of 1B, treat 1B as the best cost/performance model and skip the 50k 3B continuation.
- If resumed to 50k, stop early unless dev WER improves by at least **0.5pp** over the first 10k extra steps.

**Reasoning:** the current 3B dev gain over 1B is only **0.42pp**, and the curve is already nearly flat from 29k → 30k. That is encouraging, but not enough by itself to justify the extra GPU cost without a test-set win.

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

3B scaling (E6-3B):
    E6-3B (usual-totem-74) 30k lr=5e-5 shuffle1000 (WER 24.782%) ✓
    — 0.42pp better than 1B E6 on dev, but 59.1h on A100-80GB.
    — Best WER at step 29k; step 30k is flat (24.789%) → likely near plateau.

Next: evaluate finetuned 3B on the combined test split, then only resume 3B to 50k if it beats 1B by a meaningful margin.
```

---

## Evaluation Protocol

**Current working protocol:** evaluate on the **combined test split** and report both CER and WER.

**Important:** the existing read-aloud / conversation split configs are currently misleading for fairseq2 evals. As discovered in Phase 8B, `dataset_summary_path` affects training-data mixing, not eval-set filtering, so the prior per-subset configs all ran on the same full test set.

Until eval filtering is fixed, do this:
- Run the combined test eval for each checkpoint/workspace.
- Compare models on combined test WER/CER first.
- Treat per-subset breakdown as a follow-up task requiring a different eval mechanism.

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
| Phase 8 evals (×12) | mixed | — | <1h each | negligible | ✓ done |
| Phase 9 extensions (×2) | 300M+1B | +20k each | ~5h + ~15h | ~10 GB each | planned |
| Phase 10 3B probe | 3B | small probe | ~1h | ~5 GB | ✓ done |
| **E6-3B (usual-totem)** | **3B** | **30k** | **59.1h** | **TBD / check quota logs** | **✓ done** |
| Phase 10C 3B resume | 3B | +20k | ~40h | incremental | conditional |
