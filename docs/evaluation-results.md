# Evaluation Results

Central table of W&B evaluation results for the current omniASR Danish finetuning runs.

**Status note:** the combined-test results are the most trustworthy comparison point.
The current `read_aloud` / `conversation` eval configs are still marked as potentially
misleading in this repo's experiment notes, so the subset-tagged rows below should be
treated as provisional until eval filtering is fully verified.

## Combined Test Results

| Model | Training | Steps | Test WER | W&B run | Notes |
|---|---|---:|---:|---|---|
| `omniASR_CTC_300M_v2` | base (zero-shot) | — | **68.18%** | `copper-tree-75` | pretrained model, no finetuning |
| `omniASR_CTC_300M_v2` | finetuned E6 | 50k | **30.73%** | `balmy-vortex-87` | lr=`5e-5`, shuffle=`1000` |
| `omniASR_CTC_1B_v2` | base (zero-shot) | — | **55.39%** | `true-sound-81` | pretrained model, no finetuning |
| `omniASR_CTC_1B_v2` | finetuned E6-1B | 50k | **23.43%** | `deep-fire-90` | lr=`5e-5`, shuffle=`1000` |
| `omniASR_CTC_3B_v2` | base (zero-shot) | — | **52.87%** | `v7yi0pk2` | pretrained model, no finetuning; rerun on DTU HPC on 2026-04-22 |
| `omniASR_CTC_3B_v2` | finetuned E6-3B | 30k | **23.06%** | `lunar-rain-93` | lr=`5e-5`, shuffle=`1000`; beats `1B E6-1B 50k` by `0.38pp` |

## CoRal-Style CER Benchmark

This table is the direct comparison target against Alexandra Institute's public
Røst v3 model cards. The official metric for this table is `cer_coral`, computed
with the CoRal-style benchmark harness in `scripts/hpc/benchmark_coral_style.py`:
raw `CoRal-project/coral-v3` test audio, `read_aloud` and `conversation`
evaluated separately, `0.5s < duration < 10.0s`, Alexandra-compatible text
normalisation, and bounded aggregate CER/WER.

Published anchors are copied from the Røst v3 model cards. The omniASR rows are
intentionally marked `pending` until the local harness is run on DTU HPC; do not
compare the WER-only table above to these CER numbers directly.

| Model | Source | Params | Decoder | Read-aloud CER | Conversation CER | Notes |
|---|---|---:|---|---:|---:|---|
| `CoRal-project/roest-v3-whisper-1.5b` | published model card | 1.54B | Whisper seq2seq | **4.5%** | **11.6%** | Røst v3 Whisper, trained on CoRal-v3 read-aloud + conversation |
| `CoRal-project/roest-v3-wav2vec2-315m` | published model card | 315M | CTC | **5.9%** | **13.7%** | Best published Røst v3 CTC-sized reference |
| `openai/whisper-large-v3` | published Røst model card rerun | 1.54B | Whisper seq2seq | **10.1%** | **27.5%** | Zero-shot baseline in the Røst v3 table |
| `omniASR_CTC_300M_v2` E6 | local CoRal-style harness | 325M | greedy CTC | pending | pending | `/work3/s204696/outputs/omniasr_e6/ws_1.0bb2600b/checkpoints/step_50000/model` |
| `omniASR_CTC_1B_v2` E6 | local CoRal-style harness | 1B | greedy CTC | pending | pending | `/work3/s204696/outputs/omniasr_e6_1b/ws_1.f85211dd/checkpoints/step_50000/model` |
| `omniASR_CTC_3B_v2` E6 | local CoRal-style harness | 3B | greedy CTC | pending | pending | `/work3/s204696/outputs/omniasr_e6_3b/ws_1.2172dba0/checkpoints/step_30000/model` |

Run the full matrix with:

```bash
bash scripts/hpc/benchmark_coral_style_matrix.sh
```

For a quick smoke run:

```bash
MAX_SAMPLES=5 bash scripts/hpc/benchmark_coral_style_matrix.sh
```

Each run writes `predictions.txt`, `references.txt`, `records.jsonl`,
`scores.json`, and `by_group.csv` under
`/work3/$USER/outputs/coral_style_benchmark/<model>/<subset>/`.

### Lessons from Alexandra's Setup

- Keep this CER-first CoRal-style table separate from the existing WER-first
  fairseq2 eval table; the evaluation filters and text normalisation differ.
- Use the `0.5-10s` benchmark before making claims against Røst, because the
  existing fairseq2 configs evaluate much longer utterances.
- Audit the effective read-aloud/conversation sampling ratio in the fairseq2
  mixture; Alexandra trains on both subsets and interleaves them explicitly.
- Add a future augmentation ablation inspired by Alexandra's pipeline: peak
  normalisation, gain, background noise, coloured noise, and random filters.
- Treat KenLM/beam decoding as a separate improvement experiment. The direct
  comparison rows above should stay greedy CTC unless explicitly labelled.

## Split-Tagged Results

These runs are useful for directional analysis, but they should not yet be treated as
final benchmark evidence until split-aware eval filtering is confirmed.

| Model | Training | Eval tag | Test WER | W&B run |
|---|---|---|---:|---|
| `omniASR_CTC_300M_v2` | base (zero-shot) | `read_aloud` | **66.99%** | `peachy-breeze-76` |
| `omniASR_CTC_300M_v2` | base (zero-shot) | `conversation` | **69.67%** | `resilient-shadow-77` |
| `omniASR_CTC_300M_v2` | finetuned E6 | `read_aloud` | **28.63%** | `glamorous-shape-88` |
| `omniASR_CTC_300M_v2` | finetuned E6 | `conversation` | **33.37%** | `dainty-feather-89` |
| `omniASR_CTC_1B_v2` | base (zero-shot) | `read_aloud` | **54.57%** | `worldly-surf-82` |
| `omniASR_CTC_1B_v2` | base (zero-shot) | `conversation` | **56.42%** | `charmed-rain-83` |
| `omniASR_CTC_1B_v2` | finetuned E6-1B | `read_aloud` | **20.98%** | `upbeat-tree-91` |
| `omniASR_CTC_1B_v2` | finetuned E6-1B | `conversation` | **26.49%** | `silvery-durian-92` |
| `omniASR_CTC_3B_v2` | base (zero-shot) | `read_aloud` | **52.22%** | `fo35rc6b` |
| `omniASR_CTC_3B_v2` | base (zero-shot) | `conversation` | **53.68%** | `0uuxyj7k` |
| `omniASR_CTC_3B_v2` | finetuned E6-3B | `read_aloud` | **20.16%** | `lilac-pine-98` |
| `omniASR_CTC_3B_v2` | finetuned E6-3B | `conversation` | **26.69%** | `toasty-terrain-99` |

## Takeaways

- Finetuning is the main source of gain: `300M` improves from `68.18%` to `30.73%`, and `1B` improves from `55.39%` to `23.43%`.
- Scaling from `300M` to `1B` remains very valuable after finetuning: `30.73%` to `23.43%` on the combined test split.
- Conversation remains harder than read-aloud in the current split-tagged runs.
- `3B` follows the same pattern: base `52.87%` -> finetuned `23.06%`, with `read_aloud` easier (`20.16%`) than `conversation` (`26.69%`).
- `3B E6-3B 30k` is the current best combined-test result at `23.06%`, but it improves on `1B E6-1B 50k` by only `0.38pp` (`23.43%` -> `23.06%`).
- That means scaling from `1B` to `3B` is still helping, but only modestly relative to the extra compute already spent to reach the 3B checkpoint.

## LM Decoding Results

Iteration 1 keeps LM construction deliberately narrow: the KenLM corpus is built from
`CoRal v3` `train` transcripts only, using the same local fairseq2 parquet source as
the omniASR training/eval pipeline. That avoids validation/test leakage while keeping
the first decoding experiment easy to reproduce.

| Model | Training | Split | Decoder | LM | Beam | Alpha | Beta | WER | Notes |
|---|---|---|---|---|---:|---:|---:|---:|---|
| `omniASR_CTC_3B_v2` | finetuned E6-3B | `combined` | `greedy` | `none` | — | — | — | — | to be filled from `scripts/decode_ctc_with_lm.py` |
| `omniASR_CTC_3B_v2` | finetuned E6-3B | `combined` | `beam` | `none` | `64` | `0.0` | `0.0` | — | first non-LM beam-search comparison |
| `omniASR_CTC_3B_v2` | finetuned E6-3B | `combined` | `beam` | `danish_lm_v1_3gram` | `64` | `0.3` | `0.0` | — | iteration-1 tuning grid |
| `omniASR_CTC_3B_v2` | finetuned E6-3B | `combined` | `beam` | `danish_lm_v1_3gram` | `64` | `0.3` | `0.5` | — | iteration-1 tuning grid |
| `omniASR_CTC_3B_v2` | finetuned E6-3B | `combined` | `beam` | `danish_lm_v1_3gram` | `64` | `0.3` | `1.0` | — | iteration-1 tuning grid |
| `omniASR_CTC_3B_v2` | finetuned E6-3B | `combined` | `beam` | `danish_lm_v1_3gram` | `64` | `0.6` | `0.0` | — | iteration-1 tuning grid |
| `omniASR_CTC_3B_v2` | finetuned E6-3B | `combined` | `beam` | `danish_lm_v1_3gram` | `64` | `0.6` | `0.5` | — | iteration-1 tuning grid |
| `omniASR_CTC_3B_v2` | finetuned E6-3B | `combined` | `beam` | `danish_lm_v1_3gram` | `64` | `0.6` | `1.0` | — | iteration-1 tuning grid |
| `omniASR_CTC_3B_v2` | finetuned E6-3B | `combined` | `beam` | `danish_lm_v1_3gram` | `64` | `0.9` | `0.0` | — | iteration-1 tuning grid |
| `omniASR_CTC_3B_v2` | finetuned E6-3B | `combined` | `beam` | `danish_lm_v1_3gram` | `64` | `0.9` | `0.5` | — | iteration-1 tuning grid |
| `omniASR_CTC_3B_v2` | finetuned E6-3B | `combined` | `beam` | `danish_lm_v1_3gram` | `64` | `0.9` | `1.0` | — | iteration-1 tuning grid |

## Related Docs

- [Experiment Plan](experiment-plan.md)
- [Training Infrastructure](training-infrastructure.md)
- [Project Roadmap](project-roadmap.md)
