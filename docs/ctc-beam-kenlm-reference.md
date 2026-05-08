# CTC Beam Search And KenLM Reference

This note records the two decoder references currently used for the CTC
beam-search + KenLM methodology discussion:

- ApX Machine Learning, [Implementing Beam Search with a Language Model](https://apxml.com/courses/applied-speech-recognition/chapter-5-language-modeling-decoding/implementing-beam-search-with-lm)
- ApX Machine Learning, [Building an N-gram Model with KenLM](https://apxml.com/courses/applied-speech-recognition/chapter-5-language-modeling-decoding/building-n-gram-model-kenlm)
- Alexandra Institute, [CoRal n-gram training code](https://github.com/alexandrainst/coral/blob/main/src/coral/ngram.py)
- Alexandra Institute, [ScandiWiki](https://huggingface.co/datasets/alexandrainst/scandi-wiki)
- Alexandra Institute, [ScandiReddit](https://huggingface.co/datasets/alexandrainst/scandi-reddit)

## Beam Search

Greedy CTC decoding chooses the most likely token at every acoustic timestep,
then applies the CTC cleanup step: collapse repeated tokens and remove blanks.
This is fast and useful as a baseline, but it commits locally. If the acoustic
model is uncertain, a locally best token can push the whole transcript toward a
worse final sentence.

Beam search keeps multiple candidate transcripts alive. At each timestep, it
extends the current candidates with likely next tokens, scores the extended
candidates, sorts them, and keeps only the top `beam_width` candidates. A beam
width of `64` means the decoder keeps up to 64 partial transcripts after each
pruning step.

Without a language model, beam search is still acoustic-only. It can recover
from local ambiguity better than greedy decoding, but it does not know whether a
word sequence is linguistically likely.

With a language model, each hypothesis combines acoustic and language evidence.
The ApX beam-search reference describes the common scoring shape as:

```text
score(W) = acoustic_score + alpha * lm_score + beta * word_count
```

Where:

- `acoustic_score` is the CTC/acoustic log-probability accumulated from model
  outputs.
- `lm_score` is the language-model log-probability of the word sequence.
- `alpha` controls how much the decoder trusts the language model.
- `beta` is a word insertion bonus that helps counter the tendency to prefer
  very short hypotheses.

For this repo, this means:

- `CTC no_lm` is greedy CTC.
- `beam` is CTC beam search without an external language model.
- `CTC LM-enabled` is CTC beam search with a KenLM language model.

## KenLM

KenLM is not a variant of beam search. KenLM is a toolkit for building and
querying n-gram language models. Beam search is the search algorithm; KenLM is
one possible scoring component used by that algorithm.

An n-gram language model estimates the probability of a word from the previous
`n - 1` words. A trigram model, for example, scores:

```text
P(next_word | previous_word_1, previous_word_2)
```

During beam search, when a candidate transcript completes a word, the decoder
can query KenLM for the probability of that word in its local context. More
natural word sequences receive better LM scores and are more likely to survive
beam pruning.

The KenLM build pipeline used in this repo follows the ApX KenLM reference:

1. Prepare a normalized text corpus.
2. Use `lmplz` to estimate an n-gram model and write an ARPA file.
3. Use `build_binary` to convert the ARPA model into a compact binary model for
   fast decoder queries.

The main Alexandra-proxy OmniASR KenLM artifact is:

- corpus: `alexandrainst/scandi-wiki` Danish train text plus
  `alexandrainst/scandi-reddit` Danish train text
- excluded: CoRal-v3 test transcripts
- order: 3-gram
- default binary path: `/work3/$USER/artifacts/lm/danish_lm_alexandra_proxy_3gram.bin`

This keeps the `CTC LM-enabled` rows useful without leaking evaluation text into
the decoder. A smaller CoRal-v3-train-only LM is still available as an ablation,
but it should not be described as Alexandra-aligned.

## Methodology Implication

The `CTC LM-enabled` row should be described as a documented beam+KenLM proxy
unless Alexandra's exact beam-search settings are recovered. The current
first-pass settings are:

```text
beam_width = 64
alpha = 0.5
beta = 1.5
```

If these settings are tuned, tune them on a validation split only, then freeze
the selected values before evaluating the CoRal-v3 test split.
