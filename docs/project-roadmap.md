# Project Roadmap

End-to-end plan for fine-tuning `omniASR_CTC_300M` on CoRal-v2 Danish speech.

## Phase 1: Data Pipeline (DONE)

- [x] Updated `data.py` to load `CoRal-project/coral-v2`
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
| 2.3 | Run zero-shot inference on CoRal-v2 test set | 1-2 hours |
| 2.4 | Compute baseline WER/CER (overall + per-dialect) | 30 min |

**Output:** Baseline WER numbers for the pre-trained model on Danish.

## Phase 3: Data Conversion (CoRal-v2 → Parquet)

**Goal:** Convert HuggingFace dataset to omnilingual ASR's required Parquet format.

See [data-preparation.md](data-preparation.md) for full details.

| Step | Task | Estimated Time |
|---|---|---|
| 3.1 | Write `scripts/convert_coral_to_parquet.py` | 2-4 hours |
| 3.2 | Run conversion (474h audio, both subsets → Parquet) | 5-10 hours (CPU job on HPC) |
| 3.3 | Generate `language_distribution_0.tsv` stats | 15 min |
| 3.4 | Create fairseq2 asset card `coral_v2_danish.yaml` | 15 min |
| 3.5 | Verify with dataloader example script | 30 min |

**Output:** Parquet dataset at `data/parquet/version=0/corpus=coral_v2_*/` + asset card.

### Conversion field mapping

| Parquet Field | CoRal-v2 Source | Transform |
|---|---|---|
| `text` | `sample["text"]` | `text_normalize()` (lowercase, remove punctuation/numbers) |
| `audio_bytes` | `sample["audio"]` | Resample 48→16kHz, FLAC encode, `list<int8>` |
| `audio_size` | computed | `len(resampled_waveform)` |
| `corpus` | subset name | `"coral_v2_read_aloud"` or `"coral_v2_conversational"` |
| `split` | split name | `validation` → `dev` |
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

**Estimated training time:** 2-4 hours on single A100 for 5K steps.

## Phase 6: Evaluation & Analysis

**Goal:** Comprehensive evaluation and per-dialect breakdown.

| Step | Task |
|---|---|
| 6.1 | Evaluate best checkpoint on test set (overall WER/CER) |
| 6.2 | Per-dialect WER breakdown using CoRal-v2 metadata |
| 6.3 | Per-age-group and per-gender WER analysis |
| 6.4 | Compare: zero-shot baseline vs fine-tuned vs existing `roest-wav2vec2` models |
| 6.5 | Per-subset comparison: read_aloud vs conversational performance |
| 6.6 | Error analysis: common failure patterns |

## Phase 7: Report & Wrap-up

| Step | Task |
|---|---|
| 7.1 | Write project report (DTU 5 ECTS format) |
| 7.2 | Prepare results tables and visualizations |
| 7.3 | Clean up repository, final documentation |

## Resource Budget

| Resource | Estimated Usage |
|---|---|
| Storage (HPC) | ~150 GB (raw + Parquet + checkpoints) |
| GPU hours (training) | 4-8 hours A100 |
| GPU hours (eval) | 2-3 hours A100 |
| GPU hours (experiments) | 10-20 hours A100 |
| Total GPU budget | ~30 hours A100 |
