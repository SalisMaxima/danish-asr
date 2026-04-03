# Short-Term Experiment Plan

Based on the first full training runs (effortless-wildflower-13, lively-surf-16, warm-frost-17)
observed at step ~10k with val/WER ~47–49% and curves still clearly descending.

**Target:** CER < 6% on CoRal-v3 read-aloud, CER < 14% on conversation
(matching `roest-wav2vec2-315m-v3` — the best published CTC result at this model scale).

---

## Current Baseline

| Run | Steps | LR | Scheduler | grad_accum | Encoder freeze | Final WER | Status |
|-----|-------|----|-----------|------------|----------------|-----------|--------|
| effortless-wildflower-13 | 5k | 1e-5 | none | 4 | 0 | ~49% | stopped early |
| warm-frost-17 | 20k | 1e-5 | tri_stage (10% warmup → cosine) | 4 | 0 | ~47.5% | completed |
| autumn-dawn (E2) | 30k | 3e-5 | tri_stage (10% warmup → cosine) | 4 | 0 | **38.6%** | completed ✓ |

**E2 result (2026-04-02):** val/WER 38.6%, UER 15.2% at step 30k. No divergence — lr=3e-5 is safe.
Curves still descending at termination (loss, UER, WER all declining). Runtime: 5h on A100.
Decision: proceed with E5 (E2 settings + encoder freeze + 40k steps).

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
Runtime 5h on A100. Config: `configs/fairseq2/ctc-finetune-hpc-e2.yaml`.
Checkpoint: `/work3/s204696/outputs/omniasr_e2`
Eval: `bsub < scripts/hpc/11_eval_e2.sh` (combined test), subset configs also available.

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

### E3 — Upstream Meta LR (5e-5), 30k steps

**Hypothesis:** 5e-5 (upstream default) converges faster than E2's 3e-5 with identical other settings.
Config: `configs/fairseq2/ctc-finetune-hpc-e3.yaml` (max_num_elements matches E2 for clean comparison).
Script: `scripts/hpc/05_train_e3.sh` (walltime 14h, conservative).

```yaml
optimizer:
  config:
    lr: 5e-5

regime:
  num_steps: 30_000
```

**Watch for:** Divergence in first 3k steps. Compare final WER directly against E2 (38.6%).
**Expected outcome:** Faster convergence; possibly lower WER floor. If it diverges, E2's 3e-5 is the ceiling.

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

### E5 — Combined best settings (final run) → QUEUED

E2 validated lr=3e-5 (no divergence, WER 38.6%, curves still descending).
E5 combines: lr=3e-5 + freeze_encoder 2k steps + 40k steps for more headroom.
Config: `configs/fairseq2/ctc-finetune-hpc-e5.yaml`
Script: `scripts/hpc/10_train_e5.sh` (walltime 8h — based on E2's 5h/30k rate)

```yaml
optimizer:
  config:
    lr: 3e-5

trainer:
  freeze_encoder_for_n_steps: 2000

regime:
  num_steps: 40_000
```

This is the run to report in the final evaluation.

---

## Decision Tree

```
warm-frost-17 finishes ✓ WER ~47.5% → ran E2 in parallel
    │
    └─ WER > 30% → ran E2 (lr=3e-5, 30k) ✓
                       │
                       └─ E2 WER 38.6%, no divergence, curves descending
                              │
                              ├─ Submit E5 (lr=3e-5 + freeze + 40k) ← NEXT
                              └─ Submit E3 (lr=5e-5, 30k) for LR upper bound
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

---

## GPU Budget Estimate

Actual timing from E2: **5h for 30k steps** on A100. Disk: **7.4 GB per 30k run** with pruning
(`keep_last_n_checkpoints: 2` + `keep_best_n_checkpoints: 1`).

| Experiment | Steps | Est. A100 hours | Est. disk | Status |
|------------|-------|-----------------|-----------|--------|
| E0 (warm-frost-17) | 20k | ~3.3h | — | ✓ done |
| E2 (autumn-dawn) | 30k | **5h actual** | **7.4 GB actual** | ✓ done |
| E3 (lr=5e-5) | 30k | ~5h | ~7.4 GB | pending |
| E5 (combined) | 40k | ~7h | ~10 GB | pending |
| E2 eval | — | <1h | negligible | pending |
| **Remaining** | | **~13h** | **~18 GB** | |

Scratch headroom: 112 GB free — safe to queue E3, E5, and eval simultaneously.
