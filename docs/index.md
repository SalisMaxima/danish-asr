# Danish ASR

Fine-tuning Meta's Omnilingual ASR (`omniASR_CTC_300M_v2`) for Danish using CoRal-v3.

## Documentation

| Document | Description |
|---|---|
| [Project Roadmap](project-roadmap.md) | End-to-end plan with phases, timelines, and resource budget |
| [Evaluation Results](evaluation-results.md) | Central table of current 300M, 1B, and 3B test results |
| [Omnilingual ASR Overview](omnilingual-asr-overview.md) | Model architecture, variants, installation, and inference |
| [CoRal Dataset](coral-dataset.md) | Dataset splits, fields, demographics, and loading |
| [Data Preparation](data-preparation.md) | Converting CoRal-v3 to the required Parquet format |
| [Finetuning Recipe](finetuning-recipe.md) | CTC finetuning config, hyperparameters, and training commands |
| [Training Infrastructure](training-infrastructure.md) | Hydra configs, hardware profiles, VRAM budgets, and usage commands for all 3 models |
| [DTU HPC Setup](dtu-hpc-setup.md) | GPU queues, LSF job scripts, environment setup, storage layout |
| [Experiment Plan](experiment-plan.md) | Short-term hyperparameter experiments (LR, steps, encoder freeze, batch size) |

## Quick Reference

- **Model:** `omniASR_CTC_300M_v2` (325M params, ~2 GiB VRAM inference, ~10-12 GiB training)
- **Dataset:** [CoRal-project/coral-v3](https://huggingface.co/datasets/CoRal-project/coral-v3) (~710h Danish speech)
- **Language code:** `dan_Latn`
- **Framework:** fairseq2 (not HuggingFace Transformers)
- **Training:** CTC finetuning via omnilingual ASR recipe
- **HPC:** DTU LSF cluster, `gpua100` queue recommended
