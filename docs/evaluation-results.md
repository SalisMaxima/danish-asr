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

This table is the main comparison table against Alexandra Institute's public
Røst v3 model cards. The main metric here is `cer_coral`, computed with the
CoRal-style benchmark harness in `scripts/hpc/benchmark_coral_style.py`.

In this benchmark, the raw `CoRal-project/coral-v3` test audio is used,
`read_aloud` and `conversation` are evaluated separately, only utterances
between `0.5s` and `10.0s` are kept, and the text normalisation follows the
same overall style as Alexandra's setup.

The published reference rows are copied from the Røst v3 model cards. The
omniASR rows are still marked `pending` until the local harness is run on DTU
HPC. So the WER-only table above should not be compared directly to these
CoRal-style CER numbers.

| Model | Source | Params | Decoder | Read-aloud CER | Read-aloud WER | Conversation CER | Conversation WER | Notes |
|---|---|---:|---|---:|---:|---:|---:|---|
| `CoRal-project/roest-v3-whisper-1.5b` | published model card | 1.54B | Whisper seq2seq | **4.5%** | — | **11.6%** | — | Røst v3 Whisper, trained on CoRal-v3 read-aloud + conversation |
| `CoRal-project/roest-v3-wav2vec2-315m` | published model card | 315M | CTC | **5.9%** | — | **13.7%** | — | Best published Røst v3 CTC-sized reference |
| `openai/whisper-large-v3` | published Røst model card rerun | 1.54B | Whisper seq2seq | **10.1%** | — | **27.5%** | — | Zero-shot baseline in the Røst v3 table |
| `omniASR_CTC_300M_v2` E6 | local CoRal-style harness | 325M | `CTC no_lm` | pending | pending | pending | pending | greedy CTC proxy for Alexandra `no_lm`; checkpoint `/work3/s204696/outputs/omniasr_e6/ws_1.0bb2600b/checkpoints/step_50000/model` |
| `omniASR_CTC_300M_v2` E6 | local CoRal-style harness | 325M | `CTC LM-enabled` | pending | pending | pending | pending | beam + KenLM proxy for Alexandra default decode |
| `omniASR_CTC_1B_v2` E6 | local CoRal-style harness | 1B | `CTC no_lm` | pending | pending | pending | pending | greedy CTC proxy for Alexandra `no_lm`; checkpoint `/work3/s204696/outputs/omniasr_e6_1b/ws_1.f85211dd/checkpoints/step_50000/model` |
| `omniASR_CTC_1B_v2` E6 | local CoRal-style harness | 1B | `CTC LM-enabled` | pending | pending | pending | pending | beam + KenLM proxy for Alexandra default decode |
| `omniASR_CTC_3B_v2` E6 | local CoRal-style harness | 3B | `CTC no_lm` | pending | pending | pending | pending | greedy CTC proxy for Alexandra `no_lm`; checkpoint `/work3/s204696/outputs/omniasr_e6_3b/ws_1.2172dba0/checkpoints/step_30000/model` |
| `omniASR_CTC_3B_v2` E6 | local CoRal-style harness | 3B | `CTC LM-enabled` | pending | pending | pending | pending | beam + KenLM proxy for Alexandra default decode |
| `omniASR_LLM_300M_v2` | planned follow-on | 300M | `autoregressive LLM` | pending | pending | pending | pending | benchmark `clear-butterfly-11` under the same CoRal-style regime later |
| `omniASR_LLM_1B_v2` | planned follow-on | 1B | `autoregressive LLM` | pending | pending | pending | pending | benchmark `crisp-eon-12` under the same CoRal-style regime later |

Run the Alexandra-aligned matrix with:

```bash
KENLM_BINARY=/work3/$USER/artifacts/lm/danish_lm_v1_3gram.bin \
bash scripts/hpc/benchmark_coral_style_alexandra_matrix.sh
```

For a quick smoke run:

```bash
KENLM_BINARY=/work3/$USER/artifacts/lm/danish_lm_v1_3gram.bin \
MAX_SAMPLES=5 bash scripts/hpc/benchmark_coral_style_alexandra_matrix.sh
```

Each run writes `predictions.txt`, `references.txt`, `records.jsonl`,
`scores.json`, and `by_group.csv` under
`/work3/$USER/outputs/coral_style_benchmark_alexandra/<model>/<subset>/<decoder>/`.

### Lessons from Alexandra's Setup

- Keep this CER-first CoRal-style table separate from the existing WER-first
  fairseq2 eval table; the evaluation filters and text normalisation differ.
- Use the `0.5-10s` benchmark before making claims against Røst, because the
  existing fairseq2 configs evaluate much longer utterances.
- Audit the effective read-aloud/conversation sampling ratio in the fairseq2
  mixture; Alexandra trains on both subsets and interleaves them explicitly.
- Add a future augmentation ablation inspired by Alexandra's pipeline: peak
  normalisation, gain, background noise, coloured noise, and random filters.
- Keep the direct comparison rows above in Alexandra-style labels:
  `CTC no_lm` and `CTC LM-enabled`.
- Keep the internal decoder labels `greedy`, `beam`, and `beam + KenLM` in a
  separate analysis table below, not in the main public-comparison table.

### Training Mixture Audit

The fairseq2 recipes do train on both CoRal subsets, but the raw dataset itself
is clearly not balanced.

- The current CTC and LLM configs all use the same `coral_v3_danish` mixture
  dataset with `storage_mode: "MIXTURE_PARQUET"` and `beta_corpus: 0.5`.
- The generated `language_distribution_0.tsv` shows:
  - `coral_v3_read_aloud`: `301,302` samples, `524.94` hours
  - `coral_v3_conversation`: `149,134` samples, `146.25` hours
- So the raw training data has about `2x` more `read_aloud` samples and about
  `3.6x` more `read_aloud` audio hours than `conversation`.
- This makes it very believable that `conversation` is the harder subset in our
  results.
- This also most likely matches what Alexandra Instituttet saw themselves,
  because their CoRal v3 finetuned models were trained on the same underlying
  CoRal v3 data mixture.

The main caution is that these numbers describe the raw corpus, not the exact
per-batch sampling seen during training. Since the recipes use `beta_corpus:
0.5`, fairseq2 may rebalance the two corpora to some extent. So the safest
conclusion is: both subsets are included, but the source data is strongly
skewed toward `read_aloud`, and that likely helps explain why `conversation`
remains harder.

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

## CTC Decoding Comparison

This section is a secondary analysis table. The main CoRal-style table above is
the Alexandra-aligned public comparison. Here we keep the more technical
decoder terms so it is easier to see how much gain comes from search alone and
how much comes from KenLM.

Iteration 1 keeps LM construction deliberately narrow: the KenLM corpus is
built from `CoRal v3` `train` transcripts only, using the same local fairseq2
parquet source as the omniASR training/eval pipeline. That avoids
validation/test leakage while keeping the first decoding experiment easy to
reproduce.

| Model | Training | Split | Decoder | LM | Beam | Alpha | Beta | CER | WER | Notes |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| `omniASR_CTC_3B_v2` | finetuned E6-3B | `read_aloud` | `greedy` | `none` | — | — | — | pending | pending | direct checkpoint baseline |
| `omniASR_CTC_3B_v2` | finetuned E6-3B | `read_aloud` | `beam` | `none` | `64` | `0.0` | `0.0` | pending | pending | search-only comparison |
| `omniASR_CTC_3B_v2` | finetuned E6-3B | `read_aloud` | `beam + KenLM` | `danish_lm_v1_3gram` | `64` | pending | pending | pending | pending | best tuned LM row only |
| `omniASR_CTC_3B_v2` | finetuned E6-3B | `conversation` | `greedy` | `none` | — | — | — | pending | pending | direct checkpoint baseline |
| `omniASR_CTC_3B_v2` | finetuned E6-3B | `conversation` | `beam` | `none` | `64` | `0.0` | `0.0` | pending | pending | search-only comparison |
| `omniASR_CTC_3B_v2` | finetuned E6-3B | `conversation` | `beam + KenLM` | `danish_lm_v1_3gram` | `64` | pending | pending | pending | pending | best tuned LM row only |

Run the decoder-analysis benchmark with:

```bash
KENLM_BINARY=/work3/$USER/artifacts/lm/danish_lm_v1_3gram.bin \
bash scripts/hpc/benchmark_coral_style_decoder_analysis.sh
```

## Related Docs

- [Experiment Plan](experiment-plan.md)
- [Training Infrastructure](training-infrastructure.md)
- [Project Roadmap](project-roadmap.md)
