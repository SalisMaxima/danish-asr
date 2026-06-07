# Finetuning Meta's Omnilingual ASR V2 Models On CoRal V3

Slide-by-slide outline for my informal walkthrough with Nicki.

The overall story I want to tell:

> I started out wanting to understand how modern ASR models actually work, and
> ended up building a full Danish ASR finetuning and evaluation setup around
> Meta's omniASR V2 models and the CoRal v3 dataset. The 1B LLM model currently
> looks best on my internal WER evaluations, while the 1B/3B CTC models are the
> most useful for comparing against Alexandra Institute's Røst CTC results. The
> main thing I learned is that the comparison is not just about model size. It is
> also about decoder choice, KenLM, utterance length, CER/WER, and normalization.

## 1. Title

**Finetuning Meta's Omnilingual ASR V2 Models on the CoRal v3 Dataset**

Subtitle:

- Danish ASR finetuning and evaluation.
- Comparison against Alexandra Institute's CoRal/Røst models where the
  methodology is close enough.
- Project journey, results, bugs, limitations, and what I would do next.

What I want to say:

- This is not only a leaderboard project.
- It is also a project about learning the ASR stack, understanding fairseq2 and
  omniASR, and building enough infrastructure that I could actually run
  experiments repeatedly on HPC.

## 2. Motivation And Goals

My intention with the project was to:

- Learn how ASR models work in general.
- Understand what types of ASR models there are:
  - CTC models
  - Wav2Vec2/XLS-R style models
  - Whisper / sequence-to-sequence models
  - CTC + LLM / autoregressive decoder models
  - CTC with beam search and external language models
- Explore and understand Meta's Omnilingual ASR V2 codebase and fairseq2.
- Build my own repo/codebase using some of the MLOps principles I have learned:
  configs, scripts, reproducibility, W&B tracking, preflight checks, and result
  documentation.
- Use the newly released CoRal v3 dataset from Alexandra Institute to finetune
  different model size variants.
- See how well my finetuned models perform compared with CoRal/Røst models, as
  long as I am careful about whether the comparison is actually 1:1.

What I want to emphasize:

- I had two goals running in parallel:
  - learn the architecture and tooling deeply
  - get real Danish ASR results
- A lot of the work was not just "train model", but figuring out how to make the
  whole pipeline run reliably.

## 3. Course Learning Goal Alignment

How I think the project maps to the special-course goals:

| Course goal | How I addressed it |
|---|---|
| Account for state-of-the-art ASR architectures | I looked at CTC, Wav2Vec2/XLS-R, Whisper, omniASR V2, LLM decoders, beam search, and KenLM |
| Identify and preprocess Danish audio datasets | I used CoRal v3 and prepared it for fairseq2/parquet-based training and evaluation |
| Understand memory-efficient refinement methods | I looked at LoRA, quantization, and gradient checkpointing; for the main omniASR runs I used full finetuning with bf16, batch shaping, gradient accumulation, and checkpoint pruning |
| Finetune a pretrained ASR model for Danish | I finetuned omniASR CTC V2 models at 300M, 1B, and 3B, plus LLM V2 models up to 1B |
| Evaluate quantitatively | I evaluated with WER/CER, and compared greedy, beam, and beam+KenLM decoding |
| Compare with existing solutions | I compared against Røst/Whisper anchors, while clearly marking where the comparison is still not 1:1 |
| Perspective future Danish ASR | I discuss data, dialect coverage, speech-like LMs, augmentation, compute, and benchmark alignment |

What I should say honestly:

- I did not use LoRA or quantized finetuning as the main omniASR training path.
- I did use other memory-efficient controls that were available in the
  fairseq2/omniASR recipe.
- LoRA is supported in my repo for HuggingFace Wav2Vec2/Whisper baselines, but
  not as a simple config switch in the omniASR/fairseq2 path.

## 4. Headline Results

Before going into the details, the main results are:

- Finetuning gave large gains over the base omniASR models.
- Scaling CTC from 300M to 1B was clearly useful.
- Scaling CTC from 1B to 3B worked, but the gain was small compared with the
  extra compute.
- The 1B LLM V2 model currently has the best internal WER result.
- Beam search helped the CTC models slightly.
- The current proxy KenLM setup was mixed:
  - it helped the 3B CTC model on read-aloud
  - it hurt conversation and combined results
- The final CoRal/Røst-style CER comparison is queued/running, and that is the
  table I should use for the cleanest Røst comparison when it finishes.

How I want to frame this:

- The most interesting outcome is not just "which number is best".
- The interesting outcome is that methodology alignment became just as important
  as model size.

## 5. ASR Theory

The simple ASR picture:

- Audio waveform goes into the model.
- An encoder turns the audio into hidden representations.
- A decoder or decoding rule turns those representations into text.

Wav2Vec2 / XLS-R:

- Self-supervised speech encoder family.
- Learns audio representations from large amounts of unlabeled speech.
- Often used with a CTC head.
- This matters because Alexandra's Røst CTC reference is Wav2Vec2/XLS-R-style.

CTC:

- CTC avoids needing exact frame-level alignments between audio and text.
- It predicts token probabilities over time, including a blank token.
- The final transcription comes from collapsing repeated tokens and blanks.
- It is efficient and relatively simple.
- The downside is that the decoder choice matters a lot.

Beam search:

- Greedy decoding just picks the best token path locally.
- Beam search keeps several candidate transcriptions alive.
- This can recover better sequence-level outputs.

KenLM:

- KenLM is an external n-gram language model.
- In CTC beam decoding, it adds a text prior.
- It can help if the LM domain matches the speech.
- It can hurt if the LM is more like written text than spoken conversation.

Whisper / sequence-to-sequence:

- Encoder-decoder model.
- Generates text autoregressively.
- Has a stronger built-in language modeling component.
- Very strong baseline family, but not the same kind of model as CTC.

Meta omniASR V2:

- Multilingual ASR system from Meta.
- Uses self-supervised speech encoders.
- Has CTC V2 models at different sizes.
- Also has LLM V2 variants with a more language-model-like decoding path.
- My project focused on finetuning both the CTC V2 and LLM V2 variants for
  Danish.

CTC + LLM:

- The LLM V2 models combine an acoustic front-end with a more autoregressive
  decoder style.
- In my internal results this track looks very promising.
- But I should not compare it too directly to Røst CTC until it is evaluated
  under the same CoRal-style benchmark.

Why Danish ASR is hard:

- Dialects and pronunciation variation.
- Spoken reductions and small function words.
- Compounds and morphology.
- Fillers and hesitations.
- Conversation noise, interruptions, and natural turn-taking.
- Less high-quality labelled Danish speech data than English.

## 6. Dataset

The dataset:

- CoRal v3 from Alexandra Institute.
- Danish speech dataset with both read-aloud and conversation speech.
- Important because it gives a modern Danish benchmark context.
- Røst v3 model cards give public comparison anchors.

Why it matters:

- It is directly relevant for Danish ASR.
- It includes both easier read-aloud speech and harder conversational speech.
- It lets me test whether model size, decoder choice, and training regime behave
  differently across the two speech styles.

One important thing I found:

- The data is not balanced.
- Read-aloud has about 301k samples and 525 hours.
- Conversation has about 149k samples and 146 hours.
- So conversation is both smaller and harder, which also matches my results.

Why the Røst comparison is tricky:

- Røst reports CER as the main metric.
- My first internal table was WER-first.
- Røst reports read-aloud and conversation separately.
- My early fairseq2 evaluation had combined and split-tagged WER.
- Røst-style evaluation uses short utterances, roughly `0.5s-10s`.
- My existing fairseq2 evaluation allowed longer utterances.
- So I need the CoRal-style benchmark path before claiming a true 1:1
  comparison.

## 7. Methodology And Setup

What I had to figure out first:

- How to download and prepare CoRal v3.
- How to convert the data into a fairseq2-compatible parquet layout.
- How fairseq2 expects dataset manifests, splits, languages, and cache paths.
- How to make the training and evaluation scripts repeatable on DTU HPC.

Repo/codebase work:

- I built reusable configs for model sizes and experiment variants.
- I added HPC scripts for training and evaluation.
- I added scripts for CTC decoding, beam search, and KenLM.
- I added result docs so I could actually remember what each run meant.
- I added preflight checks because many failures were boring but expensive:
  missing checkpoints, wrong tokenizer cache, missing KenLM binary, wrong output
  paths, or quota problems.

fairseq2 / omniASR hiccups:

- Tokenizer and asset cache paths were easy to get wrong.
- Checkpoint formats were not always the same.
- Some checkpoints were fixed `.pt` files, while fairseq2 training checkpoints
  were sharded directories.
- Batching was not ordinary "batch size"; it was controlled by total audio
  elements.
- Long audio made memory unstable even when the visible batch size looked small.

HPC constraints:

- A100 queue time shaped what experiments were realistic.
- Initial batches were too large and caused memory instability.
- For 1B and 3B, I had to reduce `max_num_elements` and compensate with gradient
  accumulation.
- Checkpoints could quickly fill `/work3`, so pruning mattered.
- A failed job could cost a lot of time because I might wait hours before seeing
  the error.

## 8. Memory-Efficient Training Choices

The formal methods from the course description:

- LoRA: train low-rank adapter weights instead of all parameters.
- Quantization: reduce precision/storage of model weights.
- Gradient checkpointing: save memory by recomputing activations during
  backpropagation.

What I actually used in the main omniASR/fairseq2 runs:

- `bfloat16` mixed precision.
- Smaller `max_num_elements` to control how much audio entered each batch.
- Gradient accumulation to keep the effective batch size reasonable.
- Checkpoint pruning:
  - keep only a few recent checkpoints
  - keep the best checkpoint
- Separate batch settings for 300M, 1B, 3B, and LLM V2 models.
- A100 40GB/80GB-specific tuning.

What I did not use in the main omniASR/fairseq2 runs:

- LoRA/PEFT.
- Quantized finetuning.
- Gradient checkpointing.

Why:

- The omniASR/fairseq2 recipe I used is oriented around full finetuning.
- LoRA is not exposed there as a simple config flag.
- It would probably be possible, but it would require changing the model/recipe
  path, inserting adapters, freezing base weights, changing optimizer behavior,
  and making checkpoint loading work.
- In contrast, my repo already supports LoRA for HuggingFace Wav2Vec2 and
  Whisper baselines through PEFT.
- That is a different path than the main omniASR experiments.

What I should say:

- I did not ignore LoRA/quantization; they just did not become the main path for
  the omniASR experiments.
- The main memory-efficiency work became practical fairseq2/HPC engineering:
  bf16, batch shaping, gradient accumulation, and checkpoint management.

## 9. Experiment Journey

300M CTC:

- I started with the smallest useful CTC V2 model.
- I tested learning rate, number of steps, scheduler choices, and shuffle
  windows.
- The shuffle window turned out to matter a lot.
- Freezing the encoder/backbone did not clearly help.

1B CTC:

- I then scaled the same basic recipe to 1B.
- This gave a large improvement over 300M.
- This became the most practical CTC model in my mind:
  - much better than 300M
  - much cheaper and easier than 3B

3B CTC:

- I managed to run 3B on A100 80GB.
- It needed a smaller batch shape.
- It improved slightly over 1B.
- But the improvement was not huge, so I do not think 3B is clearly worth the
  extra cost for this setup.

LLM V2 track:

- I also tried the CTC + LLM V2 models up to 1B.
- This model family gave the strongest internal WER.
- The parameter count is a little more subtle here because the model family is
  different from pure CTC.
- It looks very promising, but still needs CoRal-style evaluation before I can
  compare it cleanly to Røst.

Beam search and KenLM:

- Finally I tried to add beam search and KenLM to the CTC models.
- This was important because Alexandra's CTC results use LM-enabled decoding.
- The first beam jobs failed because logits still required gradients before
  NumPy conversion.
- The fix was to detach logits before decoding.
- After that, beam search worked.
- Beam without LM helped slightly.
- The proxy KenLM helped read-aloud in one important case, but hurt
  conversation.

## 10. Results

Internal combined WER results:

| Model | Family | Internal combined WER | Main read |
|---|---|---:|---|
| `omniASR_CTC_300M_v2` finetuned | CTC | 30.73% | finetuning works |
| `omniASR_CTC_1B_v2` finetuned | CTC | 23.43% | large scaling gain |
| `omniASR_CTC_3B_v2` finetuned | CTC | 23.06% | modest gain over 1B |
| `omniASR_LLM_300M_v2` finetuned | LLM V2 | 20.98% | strong autoregressive track |
| `omniASR_LLM_1B_v2` finetuned | LLM V2 | 17.83% | best internal WER |

CTC decoder comparison on the internal long-utterance evaluation:

| Model | Decoder | Combined WER | Combined CER | Main read |
|---|---|---:|---:|---|
| `3b_e6_30k` | greedy | 24.08% | 9.15% | strong baseline |
| `3b_e6_30k` | beam no LM | 23.82% | 9.09% | best current CTC long-utterance result |
| `3b_e6_30k` | beam + KenLM | 24.30% | 9.41% | proxy LM hurts combined |

Split-specific KenLM behavior for the 3B CTC model:

- Read-aloud:
  - beam no LM: 20.05% WER / 6.33% CER
  - beam + KenLM: 19.29% WER / 6.11% CER
  - so the proxy LM helps here
- Conversation:
  - beam no LM: 27.90% WER / 13.08% CER
  - beam + KenLM: 29.73% WER / 14.17% CER
  - so the proxy LM hurts here

How I interpret it:

- Beam search itself seems useful.
- The current KenLM is not tuned well enough for conversation.
- The LM is probably too written-Danish-like compared with spontaneous speech.
- So I should report beam-no-LM as the best current CTC long-utterance result,
  and treat beam+KenLM as an untuned proxy-LM experiment.

Røst comparison caveat:

- These internal WER numbers are not a true Røst comparison.
- A real Røst-style comparison needs:
  - CER as the main metric
  - read-aloud and conversation separately
  - short utterance filtering
  - CoRal-style normalization
  - comparable decoder labels like `CTC no_lm` and `CTC LM-enabled`

Current final alignment jobs:

- Short-utterance 1B CTC training.
- Full CoRal/Røst-style comparison matrix.
- KenLM alpha/beta tuning probe.

## 11. What I Learned

ASR:

- The model architecture matters, but the decoder and evaluation protocol also
  matter a lot.
- A better internal WER number does not automatically mean a fair public
  benchmark comparison.
- Conversation is much harder than read-aloud speech.

fairseq2 / omniASR:

- The framework is powerful, but there are many moving pieces.
- Dataset manifests, cache paths, tokenizers, checkpoint formats, and batch
  shapes all had to be understood.
- A lot of the real work was figuring out the system boundaries.

HPC:

- GPU memory and queue time really shape the project.
- A run that fails after waiting in queue is expensive.
- Smoke tests and preflight scripts are worth it.

Evaluation:

- WER vs CER matters.
- Long utterances vs short utterances matters.
- Normalization matters.
- Split-specific reporting matters.
- Comparing to a published model card is only meaningful if the evaluation
  setup is close enough.

Reproducibility:

- Configs, scripts, W&B, preflight checks, and result docs became part of the
  project, not just supporting material.

## 12. Why Direct Comparison Is Hard

The tempting question is:

> How close are my omniASR models to Alexandra Institute's Røst models?

But the careful answer needs alignment across:

- Metric:
  - my early tables are WER-first
  - Røst is CER-first
- Data filtering:
  - my internal fairseq2 eval allows longer utterances
  - Røst-style comparison uses short utterances
- Splits:
  - Røst reports read-aloud and conversation separately
  - combined WER can hide important differences
- Normalization:
  - CoRal-style normalization handles punctuation, symbols, fillers, and
    numerals differently
- Decoder:
  - greedy CTC, beam, and beam+LM are different systems
  - the CTC comparison should use comparable decoder labels

What I want to say:

- This became one of the main lessons of the project.
- I started by thinking mostly about model size and finetuning.
- I ended up learning that benchmark methodology is just as important.

## 13. Negative Results Are Useful

Things that did not simply work:

- Freezing the encoder/backbone did not clearly improve the CTC model.
- 3B CTC was feasible, but the extra gain over 1B was small.
- Proxy KenLM helped read-aloud but hurt conversation.
- Some early split-tagged evaluations were misleading until I audited the
  filtering.
- Beam/KenLM initially crashed because logits needed to be detached before NumPy
  conversion.

Why this still matters:

- Negative results narrowed the search space.
- They explain why I would prioritize 1B CTC, LLM V2 evaluation, better LM
  tuning, and methodology alignment rather than blindly scaling up.

## 14. Discussion

Main point:

- Methodology alignment matters as much as model size.

CTC track:

- 1B CTC looks like the practical sweet spot.
- 3B CTC works, but the improvement is modest.
- Beam search is worth keeping.
- KenLM needs better tuning and probably a more speech-like corpus.

LLM V2 track:

- 1B LLM is currently the strongest internal model.
- It may be the most promising model family for peak performance.
- But I still need to compare it under the same CoRal-style benchmark before
  making a strong Røst-style claim.

KenLM:

- The LM is active and wired correctly.
- The problem is likely domain and tuning.
- A written Danish LM can push conversation outputs away from natural spoken
  forms.
- So I see this as future work, not as evidence that beam search is bad.

## 15. Future Work

Immediate next things:

- Finish and collect the CoRal/Røst-style CER table.
- Evaluate the 1B short-utterance CTC model.
- Check whether short-utterance training helps read-aloud, conversation, or
  both.

Decoder and LM:

- Tune KenLM `alpha` and `beta` on dev.
- Add unigrams to pyctcdecode.
- Build a more speech-like Danish LM.
- Keep read-aloud and conversation LM results separate.

Training:

- Try augmentation inspired by Alexandra/Røst:
  - gain
  - background noise
  - colored noise
  - random filters
  - peak normalization
- Only try 3B short-utterance training if the 1B short-utterance run gives a
  clear reason to do so.
- Finetune and test other ASR model families.

Evaluation:

- Evaluate the LLM V2 models under the same CoRal-style benchmark.
- Keep internal WER tracking separate from public CER comparison.
- Do more qualitative error analysis using the saved prediction records.

What Danish ASR needs to catch up to English:

- More high-quality labelled Danish speech, especially spontaneous conversation.
- Better coverage of dialects, accents, age groups, speaking styles, and noise.
- Stronger Danish language models that are closer to speech, not only written
  text.
- Better augmentation and normalization.
- More compute for systematic ablations rather than only the highest-priority
  runs.
- Consistent public benchmarks like CoRal/Røst so progress can be compared
  cleanly.

## 16. Current Status

Completed:

- CTC finetuning for 300M, 1B, and 3B.
- LLM V2 finetuning up to 1B.
- Internal fairseq2 WER evaluations.
- Beam search and KenLM implementation for CTC.
- Full internal CTC decoder matrix.
- Documentation of results, bugs, and methodology caveats.

Running / queued:

- 1B short-utterance CTC training.
- Full CoRal/Røst-style CER comparison matrix.
- KenLM alpha/beta probe.

Still uncertain:

- Whether short-utterance training improves CoRal-style CER.
- Whether tuned KenLM helps conversation.
- Whether the LLM V2 track stays best under the CoRal-style benchmark.

## 17. Appendix: Exact Result Tables

Use these as backup slides rather than the main story.

Internal combined WER:

| Model | Training | Combined WER |
|---|---|---:|
| `omniASR_CTC_300M_v2` base | zero-shot | 68.18% |
| `omniASR_CTC_300M_v2` finetuned | E6 50k | 30.73% |
| `omniASR_CTC_1B_v2` base | zero-shot | 55.39% |
| `omniASR_CTC_1B_v2` finetuned | E6 50k | 23.43% |
| `omniASR_CTC_3B_v2` base | zero-shot | 52.87% |
| `omniASR_CTC_3B_v2` finetuned | E6 30k | 23.06% |
| `omniASR_LLM_300M_v2` finetuned | E1 20k | 20.98% |
| `omniASR_LLM_1B_v2` finetuned | E1 15k | 17.83% |

Internal CTC decoder matrix, combined split:

| Model | Greedy WER | Beam WER | Beam+KenLM WER |
|---|---:|---:|---:|
| `300m_e6_50k` | 31.75% | 31.30% | 33.60% |
| `1b_e6_50k` | 24.42% | 24.03% | 26.32% |
| `3b_e6_30k` | 24.08% | 23.82% | 24.30% |

Published Røst anchors:

| Model | Read-aloud CER | Conversation CER | Notes |
|---|---:|---:|---|
| `CoRal-project/roest-v3-whisper-1.5b` | 4.5% | 11.6% | published Røst v3 Whisper |
| `CoRal-project/roest-v3-wav2vec2-315m` | 5.9% | 13.7% | published Røst v3 CTC |
| `openai/whisper-large-v3` zero-shot | 10.1% | 27.5% | Røst model-card rerun |

Important caveat:

- Do not compare the internal WER rows directly against the Røst CER rows.
- Use the CoRal-style benchmark result when the full matrix finishes.

## 18. Appendix: LoRA Feasibility In fairseq2/omniASR

Short answer:

- LoRA is possible in PyTorch in principle.
- It is not currently a drop-in option for the omniASR/fairseq2 recipe I used
  for the main experiments.

What LoRA would require:

- Insert low-rank adapter modules into attention/projection layers.
- Freeze most base model weights.
- Train only adapter parameters and any selected heads.
- Make sure the fairseq2 optimizer only updates the intended parameters.
- Save and load adapter weights in a way that works with fairseq2 checkpoints
  and inference.

What exists in this repo:

- HuggingFace Wav2Vec2 and Whisper baselines support LoRA through PEFT.
- The main omniASR CTC/LLM V2 runs use the upstream fairseq2 recipe path.
- The omniASR/fairseq2 path currently uses full finetuning, not PEFT.

How I should present this:

> LoRA and quantization are relevant future work, especially if the goal becomes
> fitting larger Danish ASR models with less GPU memory. For this project, the
> reliable path was full finetuning with `bfloat16`, reduced audio-element
> batches, gradient accumulation, and careful checkpoint management.

## 19. Appendix: Useful Commands

Check running jobs:

```bash
bstat
```

Inspect training:

```bash
bpeek 28604701 | tail -120
```

Inspect CoRal/Røst-style matrix:

```bash
bpeek 28607129 | tail -120
```

Collect CTC + KenLM results:

```bash
uv run python scripts/hpc/collect_ctc_kenlm_results.py
```

Summarize internal decoder scores:

```bash
python - <<'PY'
import json
import os
from pathlib import Path

root = Path("/work3") / os.environ["USER"] / "outputs/ctc_kenlm_my_method"
for p in sorted(root.rglob("scores.json")):
    model, split, decoder, _ = p.relative_to(root).parts
    s = json.loads(p.read_text())
    print(
        f"{model:12s} {split:12s} {decoder:22s} "
        f"n={s.get('num_examples')} WER={s.get('wer'):.2f}% CER={s.get('cer'):.2f}%"
    )
PY
```
