# OmniASR CoRal Methodology Alignment Plan

This document is a prioritized, resume-friendly list of changes to bring the
Meta Omnilingual ASR experiments closer to Alexandra Institute's CoRal/Røst
methodology, while still making sense for `omniASR_CTC_*_v2` models and the
current DTU HPC constraint: **one GPU with up to 80GB VRAM**.

The goal is not to copy Alexandra's exact architecture choices. Their strongest
published model is Whisper-large-v3, and their CTC reference is Wav2Vec2-XLS-R.
This project uses Meta's Omnilingual ASR CTC models. The goal is therefore to
match the **data regime, evaluation regime, normalization, sampling discipline,
and robustness choices** closely enough that comparisons are meaningful.

For the concrete execution order that follows from this methodology work, see
[CoRal Alignment Execution Plan](coral-alignment-execution-plan.md).

## Priority 0 — Lock The Direct Benchmark Before New Training

**Why this is first:** without a CoRal-style benchmark, new training experiments
can look better or worse for reasons that are only metric/filter artifacts.

### Change

Run the new CoRal-style benchmark harness for:

- `omniASR_CTC_300M_v2` E6, 50k steps
- `omniASR_CTC_1B_v2` E6, 50k steps
- `omniASR_CTC_3B_v2` E6, 30k steps
- both `read_aloud` and `conversation`

Use:

```bash
bash scripts/hpc/benchmark_coral_style_matrix.sh
```

For a smoke run:

```bash
MAX_SAMPLES=5 bash scripts/hpc/benchmark_coral_style_matrix.sh
```

### Acceptance Criteria

- `docs/evaluation-results.md` has real `cer_coral` values for 300M, 1B, and 3B.
- Each run has `scores.json`, `records.jsonl`, `predictions.txt`, `references.txt`,
  and `by_group.csv`.
- The result table clearly distinguishes:
  - published Røst model-card CER
  - locally computed OmniASR CoRal-style CER
  - existing fairseq2 WER

### HPC Notes

This is inference only. Run 300M and 1B first on A100-40GB if available, then
3B on A100-80GB. If 3B memory is tight, reduce `BATCH_SIZE=1`.

## Priority 1 — Match Alexandra's Evaluation Regime Exactly

**Why this matters:** Alexandra's public results use short CoRal-v3 test examples.
Your existing fairseq2 eval allows much longer utterances, so the current WER
table is not a direct Røst comparison.

### Change

Keep the CoRal-style benchmark path as the official public comparison path:

- raw `CoRal-project/coral-v3`
- separate `read_aloud` and `conversation`
- `0.5s < duration < 10.0s`
- rejected examples skipped
- empty references skipped
- Alexandra-compatible normalization
- official metric: `cer_coral`

Do not replace the fairseq2 eval path. Use both:

- fairseq2 eval for internal model tracking and W&B continuity
- CoRal-style benchmark for public comparison to Røst

### Acceptance Criteria

- Every future model result has:
  - combined fairseq2 WER
  - CoRal-style read-aloud CER
  - CoRal-style conversation CER
- Any claim against Røst uses CoRal-style CER only.

### HPC Notes

No training cost. This is a measurement discipline change.

## Priority 2 — Audit The Effective Read-Aloud / Conversation Sampling Ratio

**Why this matters:** Alexandra trains on both subsets and interleaves them. Their
default `dataset_probabilities: null` means each configured dataset is sampled
equally often, which may upweight the smaller subset. Your fairseq2 configs use
mixture Parquet with `beta_corpus: 0.5` and `beta_language: 0.5`, which is similar
in spirit but not obviously identical.

### Change

Add a lightweight audit script or documentation command that reports the effective
training mixture for `coral_v3_read_aloud` and `coral_v3_conversation` from
`data/parquet/version=0/language_distribution_0.tsv`.

The audit should report:

- examples per corpus
- hours per corpus
- expected sampling weight under the current fairseq2 mixture settings
- whether conversation is underweighted, proportional, or upweighted

### Recommended Target

For a CoRal-aligned experiment, prefer a balanced corpus schedule:

- either approximately `50/50` by corpus sampling,
- or at least a documented conversation-upweighted schedule if conversation CER
  is the primary weakness.

### Acceptance Criteria

- `docs/evaluation-results.md` or this document records the actual effective
  read/conversation training ratio for E6.
- A future config explicitly states whether it uses proportional or balanced
  subset sampling.

### HPC Notes

No meaningful GPU cost. This can be done before more training.

## Priority 3 — Add A Short-Utterance Training Variant

**Why this matters:** Alexandra's ASR finetuning filters training examples to
`1.0s-10.0s`, while your current fairseq2 CTC configs use roughly `2s-60s`.
This is likely one of the biggest methodology differences.

### Change

Create a new E8-style OmniASR CTC experiment that keeps the successful E6 recipe
but changes the length regime:

- model: start with `omniASR_CTC_1B_v2`
- train split: CoRal-v3 train
- valid split: CoRal-v3 dev
- min audio length: approximately `1.0s`
- max audio length: approximately `10.0s`
- keep shuffle windows at `1000`
- keep LR at `5e-5` initially
- keep tri-stage scheduler initially
- keep greedy CTC decoding for direct comparison

### Recommended First Config

Start with 1B, not 3B:

- The 1B model is much stronger than 300M.
- The 3B model only beat 1B modestly in the existing results.
- A100-80GB can run 1B comfortably, leaving headroom for longer runs or more
  stable batch shapes.

Suggested first run:

```yaml
model:
  name: "omniASR_CTC_1B_v2"

dataset:
  asr_task_config:
    min_audio_len: 16_000       # 1.0s at 16kHz
    max_audio_len: 160_000      # 10.0s at 16kHz
    batch_shuffle_window: 1000
    example_shuffle_window: 1000
    normalize_audio: true

optimizer:
  config:
    lr: 5e-5

trainer:
  mixed_precision:
    dtype: "torch.bfloat16"
  grad_accumulation:
    num_batches: 8

regime:
  num_steps: 50_000
  validate_after_n_steps: 500
  validate_every_n_steps: 1000
  checkpoint_every_n_steps: 1000
  keep_last_n_checkpoints: 2
  keep_best_n_checkpoints: 1
```

### Evaluation

Evaluate with both:

- existing fairseq2 combined WER
- CoRal-style read-aloud/conversation CER

### Acceptance Criteria

- Conversation `cer_coral` improves versus the current 1B E6 benchmark.
- Read-aloud `cer_coral` does not regress enough to erase the conversation gain.
- If combined WER worsens but CoRal-style CER improves, document the tradeoff
  explicitly instead of treating it as failure.

### HPC Notes

On one A100-80GB:

- 1B is the recommended first target.
- 3B short-utterance training can follow if 1B improves materially.
- Shorter utterances may reduce per-batch memory pressure, but fairseq2 batch
  packing behavior should still be checked with a smoke run.

## Priority 4 — Add A Moderate Audio Augmentation Ablation

**Why this matters:** Alexandra's training pipeline uses audio augmentation,
including peak normalization, random gain, ESC-50 background noise, colored noise,
and random filters. Your current fairseq2 configs mainly show `normalize_audio:
true`. This may matter most for conversation robustness.

### Change

Design one augmentation ablation after the short-utterance baseline is measured.

Target behavior:

- keep peak/audio normalization
- add modest random gain
- add low-probability background noise
- add low-probability colored noise
- add low-probability band/high/low-pass filtering

### Implementation Options

Preferred path:

- implement augmentation in the data/preprocessing or fairseq2 data pipeline only
  if it can be done without material CPU bottlenecks on HPC.

Fallback path:

- create an augmented training Parquet variant offline, then train normally from
  that dataset.

### Recommended First Experiment

Run on `omniASR_CTC_1B_v2` with the short-utterance config from Priority 3.

Keep everything else fixed:

- same model
- same length filter
- same LR
- same scheduler
- same steps
- same sampling ratio

Only add augmentation.

### Acceptance Criteria

- Conversation `cer_coral` improves.
- Demographic group gaps do not widen substantially.
- Training throughput remains acceptable on one A100-80GB node.

### HPC Notes

Be careful with CPU-side augmentation. If augmentation happens on the fly and
starves the GPU, the wall-clock cost can become worse than a larger model. Use
a smoke run to check throughput before committing to 50k steps.

## Priority 5 — Align Text Normalization For Benchmarking, Not Necessarily Training

**Why this matters:** Alexandra's evaluation normalization differs from the
omniASR normalization used in the current Parquet conversion path. That affects
reported CER/WER.

### Change

Keep benchmark normalization separate and stable. Do not rewrite all existing
training data until there is evidence that training-time normalization is the
bottleneck.

For future experiments, add metadata to configs/docs stating:

- training text normalization used
- evaluation text normalization used
- whether numerals were removed, kept, or converted
- whether punctuation was removed or converted

### Optional Future Experiment

Train a 300M or 1B model on a CoRal-compatible normalized text variant:

- keep digits/number semantics closer to Alexandra's setup
- avoid `remove_numbers=True` if it discards useful spoken content
- compare against the current E6 baseline using the CoRal-style benchmark

### Acceptance Criteria

- No future benchmark result is reported without documenting the normalization
  path.
- If a new text-normalized training corpus is created, it is versioned separately
  from the existing omniASR Parquet corpus.

### HPC Notes

Training-data regeneration is mostly CPU/storage work. It can be done before GPU
time is reserved.

## Priority 6 — Revisit Step Budget After Short-Utterance Results

**Why this matters:** Alexandra's Wav2Vec2 defaults can run to `100k` steps, while
their Whisper v3 command uses `30k` steps. Your 1B E6 run plateaued around 50k on
the current regime, and 3B gave only a modest gain at 30k.

### Change

Do not blindly increase steps yet. First run the short-utterance 1B experiment.
Then inspect:

- dev WER curve
- CoRal-style read-aloud CER
- CoRal-style conversation CER
- train/validation gap
- throughput

### Decision Rule

If short-utterance 1B is still improving at 50k:

- extend to 75k before scaling to 3B

If short-utterance 1B plateaus before 50k:

- prioritize augmentation or sampling changes over more steps

If 1B improves substantially:

- run a 3B short-utterance version only after the 1B recipe is stable

### HPC Notes

One A100-80GB is enough for 3B, but the previous 3B gain over 1B was small. Spend
80GB GPU time on recipe improvements first, then scale.

## Priority 7 — Separate LM / Beam Decoding From Model Training Comparisons

**Why this matters:** Alexandra's Wav2Vec2 pipeline can include a KenLM decoder.
Your current direct comparison should stay greedy CTC first, otherwise model
quality and decoder assistance get mixed together.

### Change

After the greedy CoRal-style table is populated, run a second decoding table:

- greedy CTC
- beam without LM
- beam with Danish KenLM

Use the same fixed checkpoints and same CoRal-style benchmark output format.

### Acceptance Criteria

- Main comparison table remains greedy CTC.
- LM-assisted results are documented in a separate table.
- KenLM corpus excludes CoRal-v3 validation/test text.

### HPC Notes

This is inference-heavy and may be CPU-memory sensitive depending on beam width
and LM size. Start with 1B and `conversation`, since conversation is the harder
split and likely benefits most.

## Priority 8 — Add Fairness-Oriented Reporting To Every CoRal-Style Result

**Why this matters:** CoRal is explicitly designed around dialect, age, gender,
and accent coverage. Aggregate CER alone can hide regressions for the groups the
dataset was built to evaluate.

### Change

Use `by_group.csv` from the benchmark harness as a first-class artifact.

For each model, summarize:

- best and worst dialect group CER
- age-group spread
- gender spread
- non-native speaker CER where enough examples exist

### Acceptance Criteria

- `docs/evaluation-results.md` includes at least a short fairness note for the
  best model after the benchmark matrix is run.
- Any method change that improves aggregate CER but worsens a group substantially
  is flagged explicitly.

### HPC Notes

No extra GPU cost if `by_group.csv` is produced during the benchmark run.

## Recommended Execution Order

1. Run CoRal-style benchmark matrix for existing 300M/1B/3B checkpoints.
2. Fill in `docs/evaluation-results.md` with real `cer_coral` values.
3. Audit read-aloud/conversation sampling ratio.
4. Create and smoke-test a 1B short-utterance config.
5. Run 1B short-utterance training to 50k.
6. Evaluate with both fairseq2 WER and CoRal-style CER.
7. Add moderate augmentation only after the short-utterance baseline is measured.
8. Decide whether to extend 1B, add augmentation, or scale the improved recipe to 3B.
9. Run LM/beam decoding as a separate result table.
10. Add fairness summaries from `by_group.csv`.

## Current Best Guess

The highest-value next training experiment is **not** another larger model run.
It is:

```text
omniASR_CTC_1B_v2 + E6 recipe + CoRal-style short utterance regime
```

Reasoning:

- 1B already captured most of the gain over 300M.
- 3B improved only modestly over 1B in the current setup.
- Alexandra's public benchmark is short-utterance and CER-first.
- The current methodology mismatch is large enough that fixing it may matter
  more than scaling the model.

If that experiment improves conversation CER, then the next step is an
augmentation ablation on the same 1B setup. Only after those two are understood
does it make sense to spend A100-80GB time on a 3B rerun.
