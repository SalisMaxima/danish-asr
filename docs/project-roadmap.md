# Project Roadmap

End-to-end plan for fine-tuning `omniASR_CTC_300M` on CoRal-v3 Danish speech.

## Motivation

Danish ASR models face challenges with dialect diversity. Research by Anders Sogaard (KU), presented at the CoRal 2026 conference, demonstrated that:

- ASR models perform significantly worse for elderly speakers with strong dialects (e.g., in elder care applications)
- Larger models can exhibit greater prediction bias between demographic groups
- Enforcing equal loss across demographic groups during training can actually improve overall performance (not just fairness)
- These findings mean fairness-aware evaluation is not just an ethical concern but can guide better model development

The CoRal team's own experiments showed that training on the full diverse dataset (all dialects) produces more robust models than training on dialect-specific subsets — with the exception of very distinct dialects like Sonderjysk, where focused training helps at equal data sizes. However, the full dataset still wins when all data is available.

Our project fine-tunes Meta's omniASR_CTC_300M on CoRal-v3 and evaluates performance across demographic groups to understand how well the model serves different Danish speaker populations.

### Why CTC (omniASR) Instead of Whisper?

The CoRal team trained both Whisper and Wav2Vec2 (CTC) models. Their largest Whisper achieved the best headline accuracy, but CTC models offer practical advantages:

- **No hallucination:** CTC predicts per audio frame and cannot generate text that wasn't in the audio
- **Smaller model size:** 325M vs 1.5B+ parameters
- **Faster inference:** Parallel output, no autoregressive decoding
- **Predictable behavior:** Important for deployment in sensitive contexts like healthcare

## Phase 1: Data Pipeline (DONE)

- [x] Updated `data.py` to load `CoRal-project/coral-v3`
- [x] Fixed split naming (`validation`, not `val`)
- [x] Created `tests/test_data.py` (11 tests passing)
- [ ] HuggingFace authentication + data download verification
- [ ] Data exploration notebook (audio lengths, text stats, dialect distribution)

## Phase 2: Environment & Baseline

**Goal:** Install omnilingual ASR, establish zero-shot baseline WER.

| Step | Task | Estimated Time | Status |
|---|---|---|---|
| 2.1 | `uv add omnilingual-asr` + verify import | 30 min | DONE |
| 2.2 | Download `omniASR_CTC_300M_v2` checkpoint | 10 min | DONE (via fairseq2 dry-run) |
| 2.3 | Run zero-shot inference on CoRal-v3 test set | 1-2 hours | Planned (Module E) |
| 2.4 | Compute baseline WER/CER (overall + per-dialect) | 30 min | Planned |

**Output:** Baseline WER numbers for the pre-trained model on Danish.

## Phase 3: Data Conversion (CoRal-v3 to Parquet) — DONE

**Goal:** Convert HuggingFace dataset to omnilingual ASR's required Parquet format.

See [data-preparation.md](data-preparation.md) for full details.

| Step | Task | Status |
|---|---|---|
| 3.1 | Write unified preprocessing module | DONE — `src/danish_asr/preprocessing.py` |
| 3.2 | Run local dry-run (small slice) | DONE — `data/parquet_dryrun/` |
| 3.3 | Verify preprocessed data locally | DONE — `invoke data.verify-preprocessed` passed |
| 3.4 | Upload universal Parquet to HPC | DONE — `/work3/s204696/data/preprocessed/` (~200GB) |
| 3.5 | HPC pipeline scripts | DONE — `scripts/hpc/` |

**Notes (omnilingual-asr 0.2.0):**
- No text normalization — `omniASR_tokenizer_written_v2` handles it natively
- `audio_bytes` schema: `pa.binary()` (raw bytes), NOT `pa.list_(pa.int8())`

### Conversion field mapping

| Parquet Field | CoRal Source | Transform |
|---|---|---|
| `text` | `sample["text"]` | Raw — no normalization (v2 tokenizer handles it) |
| `audio_bytes` | `sample["audio"]` | Resample 48→16kHz, FLAC encode, `pa.binary()` |
| `audio_size` | computed | `len(resampled_waveform)` |
| `corpus` | subset name | `"coral_v3_read_aloud"` or `"coral_v3_conversation"` |
| `split` | split name | `validation` → `dev` |
| `language` | constant | `"dan_Latn"` |

## Scratch Space Management (MANDATORY for every training run)

> **Gotcha:** `/work3` NVME pool (storagepool 6) has a **350 GB hard quota**. When hit, jobs exit
> silently with code 120 and leave no traceback. This has caused multiple lost training runs.

**Before every job submission:**
```bash
getquota_work3.sh          # storagepool 6 must be under 350 GB
rm -rf /work3/$USER/outputs/<old_run>/   # remove runs you no longer need
```

**Every fairseq2 training config must include:**
```yaml
regime:
  keep_last_n_checkpoints: 2    # ~4 GB × 2 = 8 GB on disk
  keep_best_n_checkpoints: 1    # +4 GB for best WER ckpt
```

**`run_training.py` W&B policy:** checkpoint artifact uploads are fully disabled — `wandb.log_artifact()` caches the full `.pt` file locally before uploading, which would add 4 GB per checkpoint to `wandb/cache/`. W&B only receives metrics, config, and logs. Checkpoints stay on HPC scratch.

**Budget with these settings:**
- Data: ~71 GB (fixed)
- Checkpoints: ~12 GB (3 × 4 GB max)
- W&B cache: ~0 GB (no artifact uploads in `run_training.py`)
- Caches/logs: ~2 GB
- **Total: ~85 GB — well under the 350 GB limit**

---

## Phase 4: Training Setup & Smoke Test

**Goal:** Verify training runs without errors before committing to a long run.

See [finetuning-recipe.md](finetuning-recipe.md) for config details.

| Step | Task | Where |
|---|---|---|
| 4.1 | Create `configs/omniasr/ctc-finetune-danish.yaml` | Local |
| 4.2 | Create HPC job scripts in `scripts/hpc/` | Local |
| 4.3 | Smoke test: 10 steps on local GPU (12GB) | Local |
| 4.4 | Smoke test: 10 steps on DTU A100 | HPC |
| 4.5 | Monitor VRAM usage, adjust `max_num_elements` | Both |

**Key config for 12GB local GPU:**
```yaml
max_audio_len: 480_000       # 30s
max_num_elements: 3_840_000
grad_accumulation: 8
lr: 1e-05
```

**Key config for A100 (40/80GB):**
```yaml
max_audio_len: 960_000       # 60s
max_num_elements: 3_840_000  # reduced from 7.68M — higher values cause memory fragmentation crashes
grad_accumulation: 4
lr: 1e-05
```

## Phase 5: Full Training

**Goal:** Fine-tune to convergence, select best checkpoint.

| Step | Task | Where |
|---|---|---|
| 5.1 | Submit 5,000-step training job | HPC (A100) |
| 5.2 | Monitor training loss + validation WER | W&B / logs |
| 5.3 | If needed: extend to 10K steps or adjust LR | HPC |

Note: The CoRal team found that data augmentation (audio distortion to simulate real-life conditions) improved model robustness anecdotally, though they did not measure the exact contribution. Consider enabling fairseq2's built-in augmentation if available.

### Phase 5B: Larger Model Investigation (if time and GPU budget permit)

The omniASR family includes larger CTC models that all support the same fairseq2 finetuning recipe. Given that the 7B LLM zero-shot achieves 5.0% CER on Danish (vs ~10% for whisper-large-v3 zero-shot on CoRal read-aloud), there is headroom to gain from scale.

| Model | Params | Training VRAM (est.) | Download |
|-------|--------|----------------------|----------|
| omniASR_CTC_300M_v2 | 325M | ~10–12 GB | 1.3 GiB |
| omniASR_CTC_1B_v2 | 975M | ~18–22 GB | 3.7 GiB |
| omniASR_CTC_3B_v2 | 3.1B | ~40–50 GB | 12.0 GiB |
| omniASR_CTC_7B_v2 | 6.5B | >80 GB (multi-GPU) | 25.0 GiB |

**Priority order:** Try `omniASR_CTC_1B_v2` first — fits on a single A100-40GB with grad accumulation, same recipe as 300M.

| Step | Task | Where |
|---|---|---|
| 5B.1 | Run 300M to convergence, record best CER | HPC |
| 5B.2 | Fine-tune 1B_v2 with same config, compare CER | HPC |
| 5B.3 | If 1B improves significantly, try 3B_v2 (needs A100-80GB) | HPC |
| 5B.4 | Compare: CER gain vs GPU cost across model sizes | Analysis |

**Stopping criterion:** If 1B CER ≤ 300M CER + 1% (no meaningful gain), skip 3B.

## Phase 6: Evaluation & Fairness Analysis

**Goal:** Comprehensive evaluation with demographic breakdown, inspired by the CoRal conference findings.

### 6A: Overall Performance

| Step | Task |
|---|---|
| 6A.1 | Evaluate best checkpoint on test set (overall WER/CER) |
| 6A.2 | Per-subset comparison: read_aloud vs conversation performance |
| 6A.3 | Compare: zero-shot baseline vs fine-tuned |

### 6B: Demographic Fairness Analysis

CoRal's rich metadata enables systematic fairness evaluation. Anders Sogaard's research showed that demographic performance gaps grow with model size and that equal-loss training can improve both fairness and overall performance.

| Step | Task |
|---|---|
| 6B.1 | Per-dialect WER/CER breakdown (10 dialect categories) |
| 6B.2 | Per-age-group WER/CER analysis (0-24, 25-49, 50+) |
| 6B.3 | Per-gender WER/CER analysis |
| 6B.4 | Cross-group analysis: identify which groups the model serves worst |
| 6B.5 | Compare demographic gaps: zero-shot vs fine-tuned model |

### 6C: Comparison with Existing Models

| Step | Task |
|---|---|
| 6C.1 | Compare against existing `roest-wav2vec2` models (trained on CoRal) |
| 6C.2 | Compare against CoRal team's published CER benchmarks (15-25% on conversation) |
| 6C.3 | Error analysis: common failure patterns |

### Key Questions for Evaluation

- Does fine-tuning on CoRal-v3 reduce the performance gap between standard Danish and dialect speakers?
- Which dialects benefit most from fine-tuning?
- Are there demographic groups (age, gender, dialect combinations) where the model still underperforms?
- How does conversation speech performance compare to read-aloud?

## Phase 7: Report & Wrap-up

| Step | Task |
|---|---|
| 7.1 | Write project report (DTU 5 ECTS format) |
| 7.2 | Prepare results tables and visualizations |
| 7.3 | Fairness analysis section with demographic breakdowns |
| 7.4 | Clean up repository, final documentation |

## Timeline (13 weeks)

| Weeks | Phase | Status |
|---|---|---|
| 1-3 | Phase 1: Data Pipeline | DONE |
| 4-5 | Phase 2-3: Environment + Data Conversion | DONE |
| 6-7 | Phase 4: Training Setup | Next — submit HPC Parquet job (Module D) |
| 8-9 | Phase 5: Full Training | Planned |
| 10-11 | Phase 6: Evaluation & Fairness | Planned |
| 12-13 | Phase 7: Report | Planned |

## Resource Budget

| Resource | Estimated Usage |
|---|---|
| Storage (HPC) | ~150 GB (raw + Parquet + checkpoints) |
| GPU hours (training) | 4-8 hours A100 |
| GPU hours (eval) | 2-3 hours A100 |
| GPU hours (experiments) | 10-20 hours A100 |
| Total GPU budget | ~30 hours A100 |
