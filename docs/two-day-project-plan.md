# Two-Day Project Plan

## Current Status

HPC status:

| Job ID | Purpose | Notes |
|---|---|---|
| `28604682` | Full beam matrix | Completed 2026-06-06 |
| `28604701` | 1B short-utterance CTC training | E8, `0.5s-10s` training window, queued |

Known so far:

- Greedy CTC full evaluation completed successfully.
- 3B CTC is best, but only modestly ahead of 1B.
- Conversation remains harder than read-aloud.
- Beam/KenLM previously failed because logits needed `.detach()` before `.numpy()`.
- The detach fix is merged, pulled on HPC, and confirmed present.
- The full beam matrix now runs to completion.
- Beam without LM gives small but consistent gains over greedy.
- The current Alexandra-proxy KenLM helps `3b_e6_30k/read_aloud`, but hurts
  conversation and combined results for the current fixed `alpha=0.5`,
  `beta=1.5` setup.
- Short-utterance 1B training config is merged and queued.

## Day 1: Lock Decoder Comparison

The full beam matrix answered:

- Does beam search help over greedy CTC?
  - Yes, modestly and consistently.
- Does KenLM help over beam without LM?
  - Not with the current proxy LM/settings overall.
- Is the benefit larger on conversation than read-aloud?
  - No. KenLM hurts conversation, but helps 3B read-aloud.
- Which model benefits most: 300M, 1B, or 3B?
  - 3B remains the best CTC model, with `3b_e6_30k + beam_no_lm` the best
    combined long-utterance CTC result.

Key combined results:

| Model | Greedy WER | Beam WER | Beam+KenLM WER | Read |
|---|---:|---:|---:|---|
| `300m_e6_50k` | 31.75% | **31.30%** | 33.60% | beam helps; proxy LM hurts |
| `1b_e6_50k` | 24.42% | **24.03%** | 26.32% | beam helps; proxy LM hurts |
| `3b_e6_30k` | 24.08% | **23.82%** | 24.30% | best current CTC result is beam without LM |

Initial KenLM interpretation:

- The LM is active and correctly wired, but the ScandiWiki/ScandiReddit proxy
  LM biases outputs toward written Danish.
- Manual examples show it removing spoken markers, converting forms such as
  `ik` toward `ikke`, and sometimes distorting spoken number phrases.
- This is a domain/tuning issue, not evidence that beam search is broken.

## Day 1-2: Let Short-Utterance 1B Run

Job `28604701` tests the key methodology hypothesis:

> Does training on Alexandra-style short utterances, `0.5s-10s`, improve CoRal-style performance?

Even if it does not finish before the presentation, it is valuable as a queued/running next experiment with a clear rationale.

Monitor with:

```bash
bpeek 28604701
```

## Presentation Story

Core narrative:

> Model scaling worked, but methodology alignment became the real scientific issue.
> 3B CTC is feasible and strongest under greedy long-utterance evaluation, but the gain over 1B is modest.
> A fair comparison to Alexandra/Rost requires matching decoder, utterance length, metric, and split reporting.

## Results To Show

Already available:

- Greedy CTC full results.
- CTC model scaling: 300M -> 1B -> 3B.
- Read-aloud vs conversation split behavior.
- Evidence that conversation is consistently harder.

In progress:

- 1B short-utterance CTC training.

Caveat:

- LLM V2 results look promising, but they are not yet directly comparable to Alexandra CTC/Rost methodology.

## Slide Outline

1. Project goal and CoRal v3 setup
2. Model families tested: CTC V2 and LLM V2
3. CTC scaling: 300M -> 1B -> 3B
4. Current greedy CTC results
5. Why Alexandra comparison is not apples-to-apples yet
6. Beam + KenLM debugging and initial decoder findings
7. Short-utterance training hypothesis
8. What the next 48 hours decide
9. Reflections and next steps

## Immediate Commands

Check queue:

```bash
bstat
```

Check pending reasons:

```bash
bjobs -p 28604701
```

Inspect running jobs:

```bash
bpeek 28604701
```

Summarize decoder scores:

```bash
uv run python - <<'PY'
import json
import os
from pathlib import Path

root = Path("/work3") / os.environ["USER"] / "outputs/ctc_kenlm_my_method"
for p in sorted(root.rglob("scores.json")):
    model, split, decoder, _ = p.relative_to(root).parts
    s = json.loads(p.read_text())
    print(f"{model:12s} {split:12s} {decoder:22s} n={s.get('num_examples')} WER={s.get('wer'):.2f}% CER={s.get('cer'):.2f}%")
PY
```
