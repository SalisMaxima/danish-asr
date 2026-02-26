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

| Step | Task | Estimated Time |
|---|---|---|
| 2.1 | `uv add omnilingual-asr` + verify import | 30 min |
| 2.2 | Download `omniASR_CTC_300M` checkpoint (~1.3 GiB) | 10 min |
| 2.3 | Run zero-shot inference on CoRal-v3 test set | 1-2 hours |
| 2.4 | Compute baseline WER/CER (overall + per-dialect) | 30 min |

**Output:** Baseline WER numbers for the pre-trained model on Danish.

## Phase 3: Data Conversion (CoRal-v3 to Parquet)

**Goal:** Convert HuggingFace dataset to omnilingual ASR's required Parquet format.

See [data-preparation.md](data-preparation.md) for full details.

| Step | Task | Estimated Time |
|---|---|---|
| 3.1 | Write `scripts/convert_coral_to_parquet.py` | 2-4 hours |
| 3.2 | Run conversion (474h audio, both subsets to Parquet) | 5-10 hours (CPU job on HPC) |
| 3.3 | Generate `language_distribution_0.tsv` stats | 15 min |
| 3.4 | Create fairseq2 asset card `coral_v3_danish.yaml` | 15 min |
| 3.5 | Verify with dataloader example script | 30 min |

**Output:** Parquet dataset at `data/parquet/version=0/corpus=coral_v3_*/` + asset card.

### Conversion field mapping

| Parquet Field | CoRal-v2 Source | Transform |
|---|---|---|
| `text` | `sample["text"]` | `text_normalize()` (lowercase, remove punctuation/numbers) |
| `audio_bytes` | `sample["audio"]` | Resample 48 to 16kHz, FLAC encode, `list<int8>` |
| `audio_size` | computed | `len(resampled_waveform)` |
| `corpus` | subset name | `"coral_v3_read_aloud"` or `"coral_v3_conversation"` |
| `split` | split name | `validation` to `dev` |
| `language` | constant | `"dan_Latn"` |

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
max_num_elements: 7_680_000
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

## Phase 6: Evaluation & Fairness Analysis

**Goal:** Comprehensive evaluation with demographic breakdown, inspired by the CoRal conference findings.

### 6A: Overall Performance

| Step | Task |
|---|---|
| 6A.1 | Evaluate best checkpoint on test set (overall WER/CER) |
| 6A.2 | Per-subset comparison: read_aloud vs conversational performance |
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
- How does conversational speech performance compare to read-aloud?

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
| 4-5 | Phase 2-3: Baseline + Data Conversion | Next |
| 6-7 | Phase 4: Training Setup | Planned |
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
