# CoRal Alignment Execution Plan

This document turns the methodology alignment work into a concrete step-by-step
project plan.

It is designed to answer three questions in the right order:

1. How strong are the current OmniASR CTC checkpoints under a CoRal/Røst-style benchmark?
2. How much of the gap to the LLM track is architectural, and how much is caused by decoding or benchmark mismatch?
3. What is the highest-value next training experiment once the benchmark is trustworthy?

The plan below assumes the benchmark harness from this PR exists and that the
current best checkpoints are:

- `omniASR_CTC_300M_v2` E6, 50k
- `omniASR_CTC_1B_v2` E6, 50k
- `omniASR_CTC_3B_v2` E6, 30k
- `omniASR_LLM_300M_v2` best validation run: `clear-butterfly-11`
- `omniASR_LLM_1B_v2` best validation run: `crisp-eon-12`

## Phase 0 - Lock The Benchmark Before New Conclusions

### Step 0.1 - Run the Alexandra-aligned CoRal-style CTC matrix

Run the new Alexandra-aligned benchmark harness for the existing best CTC
checkpoints:

```bash
KENLM_BINARY=/work3/$USER/artifacts/lm/danish_lm_v1_3gram.bin \
bash scripts/hpc/benchmark_coral_style_alexandra_matrix.sh
```

Outputs required for each model, subset, and decoder row:

- `scores.json`
- `records.jsonl`
- `predictions.txt`
- `references.txt`
- `by_group.csv`

### Step 0.2 - Fill the CoRal-style CER table

Update `docs/evaluation-results.md` with real values for:

- `omniASR_CTC_300M_v2` E6 `CTC no_lm`
- `omniASR_CTC_300M_v2` E6 `CTC LM-enabled`
- `omniASR_CTC_1B_v2` E6 `CTC no_lm`
- `omniASR_CTC_1B_v2` E6 `CTC LM-enabled`
- `omniASR_CTC_3B_v2` E6 `CTC no_lm`
- `omniASR_CTC_3B_v2` E6 `CTC LM-enabled`

Required metrics:

- `cer_coral`
- `wer_coral`
- split: `read_aloud`
- split: `conversation`

### Step 0.3 - Freeze the reporting rule

From this point on:

- public comparison to Røst uses CoRal-style CER only
- internal training tracking can still use fairseq2 WER
- the two tables should stay separate
- the headline CTC rows use Alexandra-style labels:
  `CTC no_lm` and `CTC LM-enabled`

### Decision gate

Do not make new claims about beating Røst or about model-family superiority
until the CTC checkpoints have real CoRal-style CER numbers.

## Phase 1 - Separate Decoder Effects From Model Effects

### Step 1.1 - Keep Alexandra-style labels in the main comparison

The direct CTC vs Røst table should remain:

- `CTC no_lm`
- `CTC LM-enabled`
- no extra decoder-only rows in the main public-comparison table

This keeps the public table closer to Alexandra's evaluation logic while still
preserving a plain-CTC baseline.

### Step 1.2 - Add a separate CTC decoding comparison

After the greedy benchmark table is complete, run a second comparison on the
same CoRal-style subsets for the same CTC checkpoints:

- greedy CTC
- beam without LM
- beam with Danish KenLM

This should be reported in a separate table, not merged into the main benchmark.

For the first pass, use only `omniASR_CTC_3B_v2` E6-3B so decoder effects are
isolated from model-size effects.

### Step 1.3 - Use the same filtered examples

The decoding comparison should use the same CoRal-style regime:

- raw `CoRal-project/coral-v3`
- `read_aloud` and `conversation` separately
- `0.5s < duration < 10.0s`
- same normalization as the benchmark harness

### Decision gate

Only after this step can we say how much of the current CTC vs LLM gap is due
to plain greedy decoding versus stronger language-aware decoding.

## Phase 2 - Make The LLM Comparison Apples-to-Apples

### Step 2.1 - Benchmark the best LLM runs with the same CoRal-style regime

Add a CoRal-style benchmark path for the best LLM checkpoints/runs so they are
measured on the same evaluation regime as the CTC checkpoints.

Target runs:

- `omniASR_LLM_300M_v2` best run: `clear-butterfly-11`
- `omniASR_LLM_1B_v2` best run: `crisp-eon-12`

### Step 2.2 - Report CTC and LLM in matched conditions

Once LLM results are available under the same benchmark, compare:

- CTC `no_lm` CoRal-style CER
- CTC `LM-enabled` CoRal-style CER
- LLM CoRal-style CER

This is the first point where architecture claims become trustworthy.

### Step 2.3 - Keep validation and evaluation separate

Do not mix:

- current LLM validation WER from W&B
- CTC fairseq2 test WER
- CoRal-style CER

The LLM validation table is useful for training progress, but not for direct
public comparison.

### Decision gate

Only after this phase can we confidently answer whether the LLM track is better
because of:

- integrated autoregressive language modeling
- easier benchmark conditions
- decoder assistance differences
- or a real model-family advantage under matched evaluation

## Phase 3 - Audit Training-Regime Mismatch

### Step 3.1 - Audit read-aloud versus conversation sampling

Document the effective training mixture used by the current fairseq2 pipeline:

- examples per subset
- hours per subset
- expected sampling weight
- whether conversation is underweighted or balanced

### Step 3.2 - Record the result in docs

Add the actual sampling interpretation to either:

- `docs/evaluation-results.md`
- or `docs/omniasr-coral-methodology-alignment.md`

### Decision gate

If conversation is underweighted, fix that before spending more 80GB GPU time on
larger models.

## Phase 4 - Run The Highest-Value New Training Experiment

### Step 4.1 - Start with 1B, not 3B

The next training experiment should be:

```text
omniASR_CTC_1B_v2 + E6 recipe + short-utterance CoRal-style training regime
```

Why:

- 1B already gave most of the gain over 300M
- 3B only improved modestly over 1B in the current setup
- CoRal/Røst benchmarking is short-utterance and CER-first
- method alignment is likely more valuable than more scale right now

### Step 4.2 - Create a short-utterance 1B config

Target training regime:

- min audio length about `1.0s`
- max audio length about `10.0s`
- keep LR at `5e-5`
- keep shuffle windows at `1000`
- keep tri-stage scheduler
- keep greedy CTC for the first direct comparison

### Step 4.3 - Smoke test, then run full training

Suggested order:

1. smoke test the new 1B config
2. confirm throughput and memory
3. run to 50k steps
4. evaluate with both fairseq2 WER and CoRal-style CER

### Decision gate

If short-utterance 1B improves CoRal-style conversation CER without unacceptable
regression elsewhere, treat that as the new baseline recipe.

## Phase 5 - Add One Focused Augmentation Ablation

### Step 5.1 - Only change augmentation

After the short-utterance 1B baseline is measured, run one augmentation ablation
with everything else fixed.

Preferred additions:

- modest gain variation
- modest background noise
- modest colored noise
- modest random filtering

### Step 5.2 - Evaluate the same way

Measure the augmentation run with:

- fairseq2 combined WER
- CoRal-style read-aloud CER
- CoRal-style conversation CER
- `by_group.csv` fairness summaries

### Decision gate

Keep augmentation only if it improves the benchmark that matters, especially
conversation CER, without creating clearly worse fairness gaps.

## Phase 6 - Decide Whether To Scale Or Decode More

After Phases 0 through 5, choose the next investment based on evidence.

### Option A - CTC + LM closes most of the gap

If CTC + LM gets close to LLM on the matched CoRal-style benchmark:

- keep improving the CTC track
- prioritize better decoding and aligned training
- delay new LLM training

### Option B - LLM still clearly wins under matched evaluation

If LLM remains clearly better even after matched benchmarking and CTC + LM:

- prioritize stabilizing the LLM 1B training track
- add held-out LLM evaluation under the same CoRal-style benchmark
- decide whether LLM becomes the main public-comparison track

### Option C - Short-utterance CTC closes the benchmark gap

If short-utterance CTC improves substantially on CoRal-style CER:

- treat methodology alignment as the main win
- only then consider rerunning 3B with the improved recipe

## Phase 7 - Add Fairness Reporting To The Final Comparison

For whichever track becomes primary, summarize:

- best and worst dialect CER
- age-group spread
- gender spread
- non-native speaker performance where sample counts allow it

Use the benchmark harness artifacts:

- `by_group.csv`
- `records.jsonl`

This should be part of the final interpretation, not a later optional extra.

## Recommended Immediate Order

1. Run the CoRal-style benchmark matrix for current best CTC checkpoints.
2. Fill the CoRal-style CER table in `docs/evaluation-results.md`.
3. Run a separate CTC decoding comparison: greedy, beam, beam+KenLM.
4. Add a matched CoRal-style benchmark path for the best LLM runs.
5. Compare CTC greedy, CTC+LM, and LLM under the same benchmark regime.
6. Audit read-aloud/conversation training mixture.
7. Train a short-utterance `omniASR_CTC_1B_v2` experiment.
8. Evaluate the short-utterance run with both fairseq2 WER and CoRal-style CER.
9. Run one augmentation ablation on top of the short-utterance 1B recipe.
10. Decide whether the next big investment should be decoder work, LLM stabilization, or a 3B rerun.

## Claim Discipline

Until the steps above are complete, the safe claims are:

- current fairseq2 WER results show strong internal progress for OmniASR CTC
- CoRal-style benchmarking is the right path for public comparison to Røst
- current LLM validation runs are promising, but not yet directly comparable to
  the CTC benchmark results
- the contribution of decoding, evaluation regime, and model architecture has
  not yet been fully separated
