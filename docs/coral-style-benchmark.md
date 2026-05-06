# CoRal-Style Direct Benchmark

This document describes the new Alexandra/CoRal-compatible benchmark path for
comparing the already fine-tuned `omniASR_CTC_300M_v2`, `omniASR_CTC_1B_v2`,
and `omniASR_CTC_3B_v2` checkpoints directly against the public Røst v3 results.

The short version: the existing project evaluation table is WER-first and uses
the fairseq2 evaluation path. Alexandra Institute's public Røst comparison is
CER-first, split-specific, and uses a different normalization and length filter.
The new benchmark harness exists so that comparisons against Røst are made on
the same kind of measurement, rather than mixing incompatible metrics.

## Why This Exists

The project already has strong WER results for the fine-tuned omniASR models,
but those numbers are not directly comparable to the Røst v3 model-card table.

There are three important differences:

1. **Metric:** the Røst table reports CER as the headline number. Your current
   central table reports WER.
2. **Filtering:** the CoRal/Røst public evaluation filters examples to roughly
   `0.5s < duration < 10.0s`. Your fairseq2 evaluation configs allow much longer
   utterances.
3. **Text normalization:** Alexandra's evaluation normalizes transcripts with a
   CoRal-specific transform, including Danish symbol substitutions, filler
   removal, a fixed character allowlist, and numeral conversion. Your fairseq2
   Parquet conversion uses omniASR normalization instead.

The new code keeps these concerns separate. Existing fairseq2 evaluation remains
unchanged. The CoRal-style benchmark path is an additional measurement layer for
direct public comparison.

## What Was Added

### Benchmark Module

`src/danish_asr/coral_benchmark.py` contains the reusable benchmark logic:

- `normalize_coral_benchmark_text()` implements Alexandra-compatible evaluation
  normalization:
  - NFKC Unicode normalization
  - lowercasing
  - Danish filler removal, such as `øhm`, `ehm`, and related hesitation sounds
  - symbol substitutions such as `% -> procent`, `+ -> plus`, `kg -> kilo`
  - `aa -> å`
  - Danish numeral conversion, e.g. `12 -> tolv`
  - character filtering with the allowlist `abcdefghijklmnopqrstuvwxyzæøå0123456789éü`
- `bounded_error_rate()` implements Alexandra-style aggregate edit error rate.
  Insertions are included in the denominator, so the official `cer_coral` and
  `wer_coral` stay bounded between `0%` and `100%`.
- `score_coral_style()` returns both CoRal-style and plain `jiwer` metrics:
  - `cer_coral`
  - `wer_coral`
  - `cer_jiwer`
  - `wer_jiwer`
- `load_coral_v3_test_subset()` loads raw `CoRal-project/coral-v3` test examples
  directly from Hugging Face for either `read_aloud` or `conversation`.
- `score_by_group()` provides demographic score breakdowns by:
  - age group
  - gender
  - dialect

### Benchmark CLI

`scripts/hpc/benchmark_coral_style.py` evaluates one OmniASR checkpoint on one
CoRal-v3 subset.

Required inputs:

- `--checkpoint-path`
- `--model-arch`, one of `300m_v2`, `1b_v2`, or `3b_v2`
- `--subset`, one of `read_aloud` or `conversation`
- `--output-dir`

Useful optional inputs:

- `--batch-size`
- `--dtype`
- `--max-samples`
- `--cache-dir`
- `--tokenizer-name`
- `--tokenizer-model-path`

The CLI uses greedy CTC decoding. That is intentional: the direct comparison rows
should describe the base fine-tuned checkpoint behavior. Beam search and KenLM
decoding should be reported separately.

### Matrix Runner

`scripts/hpc/benchmark_coral_style_matrix.sh` runs the fixed comparison matrix:

- `omniASR_CTC_300M_v2` E6 at 50k steps
- `omniASR_CTC_1B_v2` E6 at 50k steps
- `omniASR_CTC_3B_v2` E6 at 30k steps
- both `read_aloud` and `conversation`

Default checkpoint paths:

| Model | Checkpoint |
|---|---|
| `omniASR_CTC_300M_v2` E6 | `/work3/s204696/outputs/omniasr_e6/ws_1.0bb2600b/checkpoints/step_50000/model` |
| `omniASR_CTC_1B_v2` E6 | `/work3/s204696/outputs/omniasr_e6_1b/ws_1.f85211dd/checkpoints/step_50000/model` |
| `omniASR_CTC_3B_v2` E6 | `/work3/s204696/outputs/omniasr_e6_3b/ws_1.2172dba0/checkpoints/step_30000/model` |

### Documentation Updates

`docs/evaluation-results.md` now has a dedicated **CoRal-Style CER Benchmark**
section with published Røst anchors:

| Model | Read-aloud CER | Conversation CER |
|---|---:|---:|
| `CoRal-project/roest-v3-whisper-1.5b` | `4.5%` | `11.6%` |
| `CoRal-project/roest-v3-wav2vec2-315m` | `5.9%` | `13.7%` |
| `openai/whisper-large-v3` | `10.1%` | `27.5%` |

The omniASR rows are intentionally left as `pending` until the matrix runner has
been executed on DTU HPC.

## How To Run

Run the full benchmark matrix on HPC:

```bash
bash scripts/hpc/benchmark_coral_style_matrix.sh
```

Run a quick smoke test:

```bash
MAX_SAMPLES=5 bash scripts/hpc/benchmark_coral_style_matrix.sh
```

Run a single benchmark manually:

```bash
uv run python scripts/hpc/benchmark_coral_style.py \
  --checkpoint-path /work3/s204696/outputs/omniasr_e6/ws_1.0bb2600b/checkpoints/step_50000/model \
  --model-arch 300m_v2 \
  --subset read_aloud \
  --batch-size 2 \
  --dtype bfloat16 \
  --output-dir /work3/$USER/outputs/coral_style_benchmark/omniasr_ctc_300m_e6_50k/read_aloud
```

## Output Artifacts

Each benchmark run writes these files under the requested `--output-dir`:

| File | Purpose |
|---|---|
| `predictions.txt` | Normalized model predictions, one per example |
| `references.txt` | Normalized CoRal references, one per example |
| `records.jsonl` | Per-example raw and normalized prediction/reference records plus metadata |
| `scores.json` | Overall metrics and run metadata |
| `by_group.csv` | CER/WER breakdowns by age group, gender, dialect, and combinations |

The official comparison value is:

```text
scores.scores.cer_coral
```

All metric values are percentages.

## How To Interpret Results

Use `cer_coral` for the direct comparison to Røst. It is the closest local
implementation of the public CoRal/Røst scoring protocol.

Use `wer_coral` as a secondary sanity check. It is useful for continuity with
the rest of this project, but it is not the headline number in the Røst model
cards.

Use `cer_jiwer` and `wer_jiwer` only to understand how much the bounded CoRal
metric differs from plain `jiwer`. These should not be mixed into the public
comparison table unless explicitly labelled.

Keep `read_aloud` and `conversation` separate. Røst reports them separately, and
conversation is consistently harder.

## Important Differences From Existing Evaluation

The new benchmark does **not** replace `scripts/hpc/run_eval.py`.

Existing fairseq2 evaluation remains the right place to track internal model
progress and W&B continuity. The CoRal-style benchmark is for public apples-to-
apples comparison against Alexandra's Røst results.

Key differences:

| Area | Existing fairseq2 eval | CoRal-style benchmark |
|---|---|---|
| Main metric | WER | CER |
| Data source | local fairseq2 Parquet | raw HF `CoRal-project/coral-v3` |
| Split shape | combined and split-tagged configs | explicit `read_aloud` / `conversation` |
| Duration filter | longer utterances allowed | `0.5s < duration < 10.0s` |
| Text normalization | omniASR/fairseq2 path | Alexandra-compatible benchmark path |
| Decoder | fairseq2 eval recipe | greedy CTC through local inference pipeline |

## Lessons From Alexandra's Setup

The implementation also surfaces several concrete follow-up ideas for training
and evaluation.

### Evaluation Lessons

Keep a CER-first comparison table for Røst-style reporting. This avoids the
common mistake of comparing your WER numbers against their CER numbers.

Keep demographic breakdowns close to the benchmark output. The CoRal project is
not only about aggregate accuracy; the dataset is designed to expose performance
differences across dialect, gender, age, and non-native speech.

Keep the raw prediction records. When the CER gap is small, error inspection is
more useful than another aggregate table.

### Finetuning Lessons

Alexandra trains on both `read_aloud` and `conversation` and interleaves the
data. Your fairseq2 mixture should be audited so the effective sampling ratio is
clear. If one subset dominates, the model may look strong on the combined test
while lagging on the harder conversation split.

Alexandra's ASR training path uses audio augmentation: peak normalization, gain,
background noise, colored noise, and random filters. Your current fairseq2 CTC
configs mainly normalize audio. A sensible next ablation is a moderate
augmentation run focused on whether conversation CER improves.

Alexandra's public benchmark uses short utterances. A future training experiment
should test whether chunking or filtering closer to `1-10s` improves CoRal-style
CER without hurting the longer-utterance fairseq2 evaluation.

### Decoding Lessons

Greedy CTC should remain the official direct-comparison decoder for the first
omniASR table. KenLM and beam search are valuable, but they answer a different
question: how much downstream decoding can improve a fine-tuned checkpoint.

Report LM decoding separately so readers do not confuse model quality with
decoder assistance.

## Validation

The implementation is covered by `tests/test_coral_benchmark.py`.

The tests cover:

- Danish normalization edge cases
- bounded CER/WER with insertions in the denominator
- CoRal-style duration, rejected-example, and empty-text filtering
- normalized prediction/reference pairing
- demographic group scoring
- output artifact creation
- CLI smoke behavior without needing the gated dataset or HPC checkpoints

Validation commands run after implementation:

```bash
uv run pytest
uv run ruff check
uv run python -m scripts.hpc.benchmark_coral_style --help
```

All tests and ruff checks passed at implementation time.

## Known Caveat

`graphify update .` was run after the code changes, but the Graphify CLI refused
to overwrite the existing graph because the regenerated graph had fewer nodes
than the current `graphify-out/graph.json`. The benchmark implementation itself
is tested and linted, but the Graphify cache may need a manual refresh if the
project requires the knowledge graph to be exactly current.
