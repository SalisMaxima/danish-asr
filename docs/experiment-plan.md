# Short-Term Experiment Plan

Based on the first full training runs (effortless-wildflower-13, lively-surf-16, warm-frost-17)
observed at step ~10k with val/WER ~47–49% and curves still clearly descending.

**Target:** CER < 6% on CoRal-v3 read-aloud, CER < 14% on conversation
(matching `roest-wav2vec2-315m-v3` — the best published CTC result at this model scale).

---

## Current Baseline

| Run | Steps | LR | Scheduler | grad_accum | Encoder freeze | WER @ 10k |
|-----|-------|----|-----------|------------|----------------|-----------|
| effortless-wildflower-13 | 5k | 1e-5 | none | 4 | 0 | ~49% (stopped) |
| warm-frost-17 | 20k | 1e-5 | tri_stage (10% warmup → cosine) | 4 | 0 | ~47.5% (running) |

**Problem:** Curves not converged. Model likely underfitting — upstream Meta default is `lr=5e-5`,
we are 5× more conservative with no evidence that warrants it.

---

## Pre-flight Checklist (run before every experiment)

1. **Check scratch quota:** `getquota_work3.sh` — storagepool 6 must be under 200 GB hard limit
2. **Clean old outputs:** `rm -rf /work3/$USER/outputs/<old_run>/` (W&B cache no longer accumulates — checkpoint artifact uploads are disabled in `run_training.py`)
3. **Verify config has checkpoint pruning:**
   ```yaml
   keep_last_n_checkpoints: 2
   keep_best_n_checkpoints: 1
   ```
   Without this, 30k steps × ~4 GB/checkpoint = 120 GB, which reliably hits the 200 GB quota and kills the job silently with exit code 120.

---

## Experiment Queue

Run in this order. Each changes **one variable** from the settled baseline (warm-frost-17).

### E0 — Wait for warm-frost-17 to finish (no action needed)

Let the 20k run complete. This gives the true baseline: tri_stage scheduler + 1e-5 + 20k steps.

**Decision gate:** If final WER > 35%, the LR is too low or steps too few — proceed with E1 and E2 in parallel.

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

### E2 — Higher learning rate

**Hypothesis:** 1e-5 is too conservative. The upstream default is 5e-5 and roest models use similar.

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

**Watch for:** Training loss spike or divergence in first 3k steps → revert to 1e-5.
**Expected outcome:** Faster convergence; potentially lower final WER.

---

### E3 — Encoder freezing during warmup

**Hypothesis:** Letting the CTC head stabilize before the encoder moves prevents early catastrophic
forgetting of the multilingual wav2vec2 representations.

```yaml
trainer:
  freeze_encoder_for_n_steps: 2000   # freeze encoder for first 2k steps

regime:
  num_steps: 30_000
```

All other parameters same as warm-frost-17. Combine with E2's LR if E2 shows improvement.

**Expected outcome:** Faster early convergence; smoother loss curve in steps 0–5k.

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

### E5 — Combined best settings (final run)

After E1–E4, combine the settings that showed improvement:

```yaml
optimizer:
  config:
    lr: <best from E2>

trainer:
  freeze_encoder_for_n_steps: <best from E3>
  grad_accumulation:
    num_batches: <best from E4>

regime:
  num_steps: 40_000
```

This is the run to report in the final evaluation.

---

## Decision Tree

```
warm-frost-17 finishes
    │
    ├─ WER < 30% → run E3 + E5 only (already in good shape)
    │
    └─ WER > 30% → run E1 and E2 in parallel
                       │
                       ├─ E2 diverges → stick with 1e-5, run E1 longer
                       └─ E2 improves → combine with E3 for E5
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

| Experiment | Steps | Est. A100 hours |
|------------|-------|-----------------|
| E0 (wait) | 20k (10k remaining) | ~4h remaining |
| E1 | 40k | ~16h |
| E2 | 30k | ~12h |
| E3 | 30k | ~12h |
| E5 (combined) | 40k | ~16h |
| **Total** | | **~60h** |

Fits within the ~30h remaining in the Phase 5 budget if E1 and E2 are run in parallel,
and E3/E4 are skipped if E0/E2 already hit the target.
