# Evaluation Results

Central table of W&B evaluation results for the current omniASR Danish finetuning runs.

**Status note:** the combined-test results are the most trustworthy comparison point.
The current `read_aloud` / `conversation` eval configs are still marked as potentially
misleading in this repo's experiment notes, so the subset-tagged rows below should be
treated as provisional until eval filtering is fully verified.

**LLM base eval note:** the pre-fix May 9, 2026 `omniASR_LLM_300M_v2`
split-tagged base runs (`peachy-dragon-39`, `jolly-planet-40`) processed the
same 14,442 examples as the combined run (`curious-snow-38`), so only the
combined score is recorded below. Do not treat those split-tagged W&B scores as
subset results. The May 10, 2026 `omniASR_LLM_1B_v2` combined run
(`olive-wildflower-41`) finished before subset-root filtering was deployed; its
combined score is valid, but any pre-fix split-tagged follow-ons are not.

## Combined Test Results

These rows use the existing fairseq2 eval recipe over the combined CoRal-v3
test split. They are useful internal WER comparisons, but they are not yet the
Alexandra-aligned CoRal-style CER benchmark below.

| Model | Training | Steps | Train script | Train config | Eval script | Eval config | Test WER | W&B run | Notes |
|---|---|---:|---|---|---|---|---:|---|---|
| `omniASR_CTC_300M_v2` | base (zero-shot) | — | — | — | `scripts/hpc/300m/20_eval_base.sh` | `configs/fairseq2/300m/ctc-eval-base.yaml` | **68.18%** | `copper-tree-75` | pretrained model, no finetuning |
| `omniASR_LLM_300M_v2` | base (zero-shot) | — | — | — | `scripts/hpc/llm_300m/19_eval_base.sh` | `configs/fairseq2/llm_300m/llm-eval-base.yaml` | **53.67%** | `curious-snow-38` | pretrained model, no finetuning; W&B run `c3wrqniz` |
| `omniASR_CTC_300M_v2` | finetuned E6 | 50k | `scripts/hpc/300m/14_train_e6.sh` | `configs/fairseq2/300m/ctc-finetune-hpc-e6.yaml` | `scripts/hpc/300m/21_eval_e6_full.sh` | `configs/fairseq2/300m/ctc-eval-e6.yaml` | **30.73%** | `balmy-vortex-87` | lr=`5e-5`, shuffle=`1000` |
| `omniASR_LLM_300M_v2` | finetuned E1 | 20k | `scripts/hpc/llm_300m/14_train_e1.sh` | `configs/fairseq2/llm_300m/llm-finetune-hpc-e1.yaml` | `scripts/hpc/llm_300m/20_eval_e1.sh` | `configs/fairseq2/llm_300m/llm-eval-e1.yaml` | **20.98%** | `crimson-paper-20` | autoregressive LLM decoder |
| `omniASR_CTC_1B_v2` | base (zero-shot) | — | — | — | `scripts/hpc/1b/20_eval_base_1b.sh` | `configs/fairseq2/1b/ctc-eval-base.yaml` | **55.39%** | `true-sound-81` | pretrained model, no finetuning |
| `omniASR_LLM_1B_v2` | base (zero-shot) | — | — | — | `scripts/hpc/llm_1b/19_eval_base_1b.sh` | `configs/fairseq2/llm_1b/llm-eval-base-1b.yaml` | **48.79%** | `olive-wildflower-41` | pretrained model, no finetuning; W&B run `0vp66nud` |
| `omniASR_CTC_1B_v2` | finetuned E6-1B | 50k | `scripts/hpc/1b/14_train_e6_1b.sh` | `configs/fairseq2/1b/ctc-finetune-hpc-e6-1b.yaml` | `scripts/hpc/1b/21_eval_e6_1b.sh` | `configs/fairseq2/1b/ctc-eval-e6-1b.yaml` | **23.43%** | `deep-fire-90` | lr=`5e-5`, shuffle=`1000` |
| `omniASR_LLM_1B_v2` | finetuned E1 | 15k | `scripts/hpc/llm_1b/14_train_e1_1b.sh` | `configs/fairseq2/llm_1b/llm-finetune-hpc-e1-1b-15k.yaml` | `scripts/hpc/llm_1b/20_eval_e1_1b.sh` | `configs/fairseq2/llm_1b/llm-eval-e1-1b-15k.yaml` | **17.83%** | `happy-universe-21` | autoregressive LLM decoder |
| `omniASR_CTC_3B_v2` | base (zero-shot) | — | — | — | `scripts/hpc/3b/19_eval_base_3b.sh` | `configs/fairseq2/3b/ctc-eval-base-3b.yaml` | **52.87%** | `v7yi0pk2` | pretrained model, no finetuning; rerun on DTU HPC on 2026-04-22 |
| `omniASR_CTC_3B_v2` | finetuned E6-3B | 30k | `scripts/hpc/3b/14_train_e6_3b.sh` | `configs/fairseq2/3b/ctc-finetune-hpc-e6-3b.yaml` | `scripts/hpc/3b/20_eval_e6_3b.sh` | `configs/fairseq2/3b/ctc-eval-e6-3b.yaml` | **23.06%** | `lunar-rain-93` | lr=`5e-5`, shuffle=`1000`; beats `1B E6-1B 50k` by `0.38pp` |

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
bash scripts/hpc/submit_coral_ctc_kenlm_eval.sh full
```

For a quick smoke run:

```bash
bash scripts/hpc/submit_coral_ctc_kenlm_eval.sh smoke
```

The submit helper expects the KenLM artifact at
`/work3/$USER/artifacts/lm/danish_lm_alexandra_proxy_3gram.bin`. If it is
missing, build it first with:

```bash
bsub < scripts/hpc/build_lm_corpus.sh
bsub < scripts/hpc/build_kenlm.sh
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
final benchmark evidence until split-aware eval filtering is confirmed. The eval
configs here are split-filtered variants of the same fairseq2 eval recipes above.

| Model | Training | Eval tag | Train script | Train config | Eval script | Eval config | Test WER | W&B run |
|---|---|---|---|---|---|---|---:|---|
| `omniASR_CTC_300M_v2` | base (zero-shot) | `read_aloud` | — | — | `scripts/hpc/300m/20_eval_base.sh` | `configs/fairseq2/300m/ctc-eval-base-read-aloud.yaml` | **66.99%** | `peachy-breeze-76` |
| `omniASR_CTC_300M_v2` | base (zero-shot) | `conversation` | — | — | `scripts/hpc/300m/20_eval_base.sh` | `configs/fairseq2/300m/ctc-eval-base-conversation.yaml` | **69.67%** | `resilient-shadow-77` |
| `omniASR_LLM_300M_v2` | base (zero-shot) | `read_aloud` | — | — | `scripts/hpc/llm_300m/19_eval_base.sh` | `configs/fairseq2/llm_300m/llm-eval-base-read-aloud.yaml` | pending | pending |
| `omniASR_LLM_300M_v2` | base (zero-shot) | `conversation` | — | — | `scripts/hpc/llm_300m/19_eval_base.sh` | `configs/fairseq2/llm_300m/llm-eval-base-conversation.yaml` | pending | pending |
| `omniASR_CTC_300M_v2` | finetuned E6 | `read_aloud` | `scripts/hpc/300m/14_train_e6.sh` | `configs/fairseq2/300m/ctc-finetune-hpc-e6.yaml` | `scripts/hpc/300m/21_eval_e6_full.sh` | `configs/fairseq2/300m/ctc-eval-e6-read-aloud.yaml` | **28.63%** | `glamorous-shape-88` |
| `omniASR_CTC_300M_v2` | finetuned E6 | `conversation` | `scripts/hpc/300m/14_train_e6.sh` | `configs/fairseq2/300m/ctc-finetune-hpc-e6.yaml` | `scripts/hpc/300m/21_eval_e6_full.sh` | `configs/fairseq2/300m/ctc-eval-e6-conversation.yaml` | **33.37%** | `dainty-feather-89` |
| `omniASR_LLM_300M_v2` | finetuned E1 | `read_aloud` | `scripts/hpc/llm_300m/14_train_e1.sh` | `configs/fairseq2/llm_300m/llm-finetune-hpc-e1.yaml` | `scripts/hpc/llm_300m/20_eval_e1.sh` | `configs/fairseq2/llm_300m/llm-eval-e1-read-aloud.yaml` | **16.93%** | `vocal-surf-23` |
| `omniASR_LLM_300M_v2` | finetuned E1 | `conversation` | `scripts/hpc/llm_300m/14_train_e1.sh` | `configs/fairseq2/llm_300m/llm-finetune-hpc-e1.yaml` | `scripts/hpc/llm_300m/20_eval_e1.sh` | `configs/fairseq2/llm_300m/llm-eval-e1-conversation.yaml` | **26.00%** | `good-energy-25` |
| `omniASR_CTC_1B_v2` | base (zero-shot) | `read_aloud` | — | — | `scripts/hpc/1b/20_eval_base_1b.sh` | `configs/fairseq2/1b/ctc-eval-base-read-aloud.yaml` | **54.57%** | `worldly-surf-82` |
| `omniASR_CTC_1B_v2` | base (zero-shot) | `conversation` | — | — | `scripts/hpc/1b/20_eval_base_1b.sh` | `configs/fairseq2/1b/ctc-eval-base-conversation.yaml` | **56.42%** | `charmed-rain-83` |
| `omniASR_LLM_1B_v2` | base (zero-shot) | `read_aloud` | — | — | `scripts/hpc/llm_1b/19_eval_base_1b.sh` | `configs/fairseq2/llm_1b/llm-eval-base-1b-read-aloud.yaml` | pending | pending |
| `omniASR_LLM_1B_v2` | base (zero-shot) | `conversation` | — | — | `scripts/hpc/llm_1b/19_eval_base_1b.sh` | `configs/fairseq2/llm_1b/llm-eval-base-1b-conversation.yaml` | pending | pending |
| `omniASR_CTC_1B_v2` | finetuned E6-1B | `read_aloud` | `scripts/hpc/1b/14_train_e6_1b.sh` | `configs/fairseq2/1b/ctc-finetune-hpc-e6-1b.yaml` | `scripts/hpc/1b/21_eval_e6_1b.sh` | `configs/fairseq2/1b/ctc-eval-e6-1b-read-aloud.yaml` | **20.98%** | `upbeat-tree-91` |
| `omniASR_CTC_1B_v2` | finetuned E6-1B | `conversation` | `scripts/hpc/1b/14_train_e6_1b.sh` | `configs/fairseq2/1b/ctc-finetune-hpc-e6-1b.yaml` | `scripts/hpc/1b/21_eval_e6_1b.sh` | `configs/fairseq2/1b/ctc-eval-e6-1b-conversation.yaml` | **26.49%** | `silvery-durian-92` |
| `omniASR_LLM_1B_v2` | finetuned E1 | `read_aloud` | `scripts/hpc/llm_1b/14_train_e1_1b.sh` | `configs/fairseq2/llm_1b/llm-finetune-hpc-e1-1b-15k.yaml` | `scripts/hpc/llm_1b/20_eval_e1_1b.sh` | `configs/fairseq2/llm_1b/llm-eval-e1-1b-15k-read-aloud.yaml` | **13.68%** | `grateful-music-22` |
| `omniASR_LLM_1B_v2` | finetuned E1 | `conversation` | `scripts/hpc/llm_1b/14_train_e1_1b.sh` | `configs/fairseq2/llm_1b/llm-finetune-hpc-e1-1b-15k.yaml` | `scripts/hpc/llm_1b/20_eval_e1_1b.sh` | `configs/fairseq2/llm_1b/llm-eval-e1-1b-15k-conversation.yaml` | **23.01%** | `eager-elevator-24` |
| `omniASR_CTC_3B_v2` | base (zero-shot) | `read_aloud` | — | — | `scripts/hpc/3b/19_eval_base_3b.sh` | `configs/fairseq2/3b/ctc-eval-base-3b-read-aloud.yaml` | **52.22%** | `fo35rc6b` |
| `omniASR_CTC_3B_v2` | base (zero-shot) | `conversation` | — | — | `scripts/hpc/3b/19_eval_base_3b.sh` | `configs/fairseq2/3b/ctc-eval-base-3b-conversation.yaml` | **53.68%** | `0uuxyj7k` |
| `omniASR_CTC_3B_v2` | finetuned E6-3B | `read_aloud` | `scripts/hpc/3b/14_train_e6_3b.sh` | `configs/fairseq2/3b/ctc-finetune-hpc-e6-3b.yaml` | `scripts/hpc/3b/20_eval_e6_3b.sh` | `configs/fairseq2/3b/ctc-eval-e6-3b-read-aloud.yaml` | **20.16%** | `lilac-pine-98` |
| `omniASR_CTC_3B_v2` | finetuned E6-3B | `conversation` | `scripts/hpc/3b/14_train_e6_3b.sh` | `configs/fairseq2/3b/ctc-finetune-hpc-e6-3b.yaml` | `scripts/hpc/3b/20_eval_e6_3b.sh` | `configs/fairseq2/3b/ctc-eval-e6-3b-conversation.yaml` | **26.69%** | `toasty-terrain-99` |

## Takeaways

- Finetuning is the main source of gain: `300M` improves from `68.18%` to `30.73%`, and `1B` improves from `55.39%` to `23.43%`.
- Scaling from `300M` to `1B` remains very valuable after finetuning: `30.73%` to `23.43%` on the combined test split.
- Conversation remains harder than read-aloud in the current split-tagged runs.
- `3B` follows the same pattern: base `52.87%` -> finetuned `23.06%`, with `read_aloud` easier (`20.16%`) than `conversation` (`26.69%`).
- The LLM V2 track is now strongest on the old fairseq2 eval path: `300M E1 20k` reaches `20.98%`, and `1B E1 15k` reaches `17.83%`.
- `LLM 1B E1 15k` is the current best combined-test result, improving on `CTC 3B E6-3B 30k` by `5.23pp` (`23.06%` -> `17.83%`).

## CTC Decoding Comparison

This section is a secondary analysis table. The main CoRal-style table above is
the Alexandra-aligned public comparison. Here we keep the more technical
decoder terms so it is easier to see how much gain comes from search alone and
how much comes from KenLM.

The 2026-06-06 run used the existing fairseq2/parquet test splits, not the
short-utterance CoRal-style benchmark. It is therefore an internal decoder
analysis for the current long-utterance-trained CTC checkpoints.

Setup:

- `beam_width=64`
- `beam_lm` uses `alpha=0.5`, `beta=1.5`
- KenLM artifact:
  `/work3/$USER/artifacts/lm/danish_lm_alexandra_proxy_3gram.bin`
- LM corpus: Danish ScandiWiki plus Danish ScandiReddit, with CoRal-v3 test
  transcripts excluded
- output root: `/work3/$USER/outputs/ctc_kenlm_my_method`

| Model | Split | Greedy WER | Greedy CER | Beam WER | Beam CER | Beam+KenLM WER | Beam+KenLM CER | Initial read |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `300m_e6_50k` | combined | 31.75% | 12.24% | **31.30%** | **12.10%** | 33.60% | 12.96% | beam helps slightly; proxy LM hurts |
| `300m_e6_50k` | read_aloud | 28.56% | 9.12% | **28.32%** | **9.03%** | 30.16% | 9.18% | beam helps slightly; proxy LM hurts |
| `300m_e6_50k` | conversation | 35.19% | 16.73% | **34.53%** | **16.54%** | 37.31% | 18.41% | beam helps; proxy LM hurts strongly |
| `1b_e6_50k` | combined | 24.42% | 9.39% | **24.03%** | **9.32%** | 26.32% | 10.12% | beam helps slightly; proxy LM hurts |
| `1b_e6_50k` | read_aloud | 20.99% | 6.66% | **20.75%** | **6.61%** | 21.73% | 6.68% | beam helps slightly; proxy LM hurts |
| `1b_e6_50k` | conversation | 28.14% | 13.34% | **27.59%** | **13.24%** | 31.30% | 15.09% | beam helps; proxy LM hurts strongly |
| `3b_e6_30k` | combined | 24.08% | 9.15% | **23.82%** | **9.09%** | 24.30% | 9.41% | best combined result is beam without LM |
| `3b_e6_30k` | read_aloud | 20.14% | 6.36% | 20.05% | 6.33% | **19.29%** | **6.11%** | proxy LM helps scripted/read-aloud speech |
| `3b_e6_30k` | conversation | 28.35% | 13.17% | **27.90%** | **13.08%** | 29.73% | 14.17% | beam helps; proxy LM hurts conversation |

Initial interpretation:

- Beam search without LM gives small but consistent improvements over greedy
  decoding.
- The current Alexandra-proxy KenLM is mixed: it helps `3b_e6_30k` on
  `read_aloud`, but hurts conversation and therefore hurts most combined rows.
- Manual inspection suggests the proxy LM biases outputs toward written Danish:
  it removes spoken markers such as `oeh`, maps spoken forms such as `ik` toward
  `ikke`, and can turn spoken number phrases into more written-looking forms.
- This is evidence of LM/domain or alpha-beta mismatch, not evidence that beam
  search is broken. The LM was loaded with the expected path, `beam_width=64`,
  `alpha=0.5`, and `beta=1.5`.
- For the presentation, report `beam_no_lm` as the best current long-utterance
  CTC decoder result, and report `beam_lm` separately as an untuned proxy-LM
  experiment.

Recommended next decoder checks:

1. Tune `alpha`/`beta` on dev only, with lower LM weights such as
   `alpha in {0.05, 0.1, 0.2, 0.3}` and `beta in {0.0, 0.5, 1.0}`.
2. Add a unigram list to the pyctcdecode setup; the current run warns that
   unigrams are not available from the binary LM.
3. Build a more speech-like LM using CoRal train transcripts or other Danish
   spoken/conversation text, while still excluding dev/test transcripts.
4. Keep LM-assisted results split-specific: read-aloud may benefit from the
   written-text prior, while conversation currently does not.

## Related Docs

- [Experiment Plan](experiment-plan.md)
- [Training Infrastructure](training-infrastructure.md)
- [Project Roadmap](project-roadmap.md)
