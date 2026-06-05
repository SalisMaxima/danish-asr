# HPC Eval Validation Runbook

This repo now keeps the implementation local and leaves HPC execution to the operator.

## 1. Collect the Previous Failure Logs

Inspect the LSF logs for the two failed May 2026 jobs:

- `/work3/s204696/logs/lsf/ctc_kenlm_my_method_28394153.out`
- `/work3/s204696/logs/lsf/ctc_kenlm_my_method_28394153.err`
- `/work3/s204696/logs/lsf/coral_ctc_alexandra_matrix_28394154.out`
- `/work3/s204696/logs/lsf/coral_ctc_alexandra_matrix_28394154.err`

Also inspect any nested `run.log` files under the configured CTC output roots:

- `/work3/$USER/outputs/ctc_kenlm_my_method`
- `/work3/$USER/outputs/ctc_kenlm_coral_method`

The preflight now prints cache layout, tokenizer resolution, output roots, model checkpoints, KenLM binary, and parquet root before decoding.

## 2. Smoke CTC + Beam + KenLM

Run only tiny cases first, in this order:

```bash
MAX_SAMPLES=2 DECODERS=greedy OVERWRITE=true bsub < scripts/hpc/benchmark_ctc_kenlm_my_method.sh
MAX_SAMPLES=2 DECODERS=beam_no_lm OVERWRITE=true bsub < scripts/hpc/benchmark_ctc_kenlm_my_method.sh
MAX_SAMPLES=2 DECODERS=beam_lm OVERWRITE=true bsub < scripts/hpc/benchmark_ctc_kenlm_my_method.sh
```

Only submit the full matrix after all three smoke jobs produce `SUCCESS` markers and non-empty `scores.json` files.

## 3. Smoke LLM Zero-Shot Splits

The LLM base scripts now resolve the cached pretrained fairseq2 asset into an explicit `model.path` before preparing split configs. This keeps the run zero-shot while making the fairseq2 eval path behave like the finetuned checkpoint evals.

If the resolver reports a missing asset, pre-download the asset or set `BASE_MODEL_PATH` to the cached checkpoint path before submitting.

Run single-split checks first:

```bash
EVAL_CONFIG=configs/fairseq2/llm_1b/llm-eval-base-1b.yaml bsub < scripts/hpc/llm_1b/19_eval_base_1b.sh
EVAL_CONFIG=configs/fairseq2/llm_1b/llm-eval-base-1b-read-aloud.yaml bsub < scripts/hpc/llm_1b/19_eval_base_1b.sh
EVAL_CONFIG=configs/fairseq2/llm_1b/llm-eval-base-1b-conversation.yaml bsub < scripts/hpc/llm_1b/19_eval_base_1b.sh
```

Acceptance checks:

- Prepared subset configs contain `model.path`.
- Read-aloud and conversation configs point at different one-corpus parquet roots.
- Split runs report distinct `num_examples`.
- W&B tags include `combined`, `read_aloud`, or `conversation` as appropriate.
