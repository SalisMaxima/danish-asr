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
| `omniASR_CTC_3B_v2` | base (zero-shot) | — | pending | — | base 3B eval script added; not yet run |
| `omniASR_CTC_3B_v2` | finetuned E6-3B | 30k | **23.06%** | `lunar-rain-93` | lr=`5e-5`, shuffle=`1000`; beats `1B E6-1B 50k` by `0.38pp` |

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
| `omniASR_CTC_3B_v2` | base (zero-shot) | `read_aloud` | pending | — |
| `omniASR_CTC_3B_v2` | base (zero-shot) | `conversation` | pending | — |
| `omniASR_CTC_3B_v2` | finetuned E6-3B | `read_aloud` | pending | — |
| `omniASR_CTC_3B_v2` | finetuned E6-3B | `conversation` | pending | — |

## Takeaways

- Finetuning is the main source of gain: `300M` improves from `68.18%` to `30.73%`, and `1B` improves from `55.39%` to `23.43%`.
- Scaling from `300M` to `1B` remains very valuable after finetuning: `30.73%` to `23.43%` on the combined test split.
- Conversation remains harder than read-aloud in the current split-tagged runs.
- `3B E6-3B 30k` is the current best combined-test result at `23.06%`, but it improves on `1B E6-1B 50k` by only `0.38pp` (`23.43%` -> `23.06%`).
- That means scaling from `1B` to `3B` is still helping, but only modestly relative to the extra compute already spent to reach the 3B checkpoint.
- `3B` zero-shot and split-tagged evaluations are now wired in the repo, but those runs still need to be submitted before the pending rows above can be filled.

## Related Docs

- [Experiment Plan](experiment-plan.md)
- [Training Infrastructure](training-infrastructure.md)
- [Project Roadmap](project-roadmap.md)
