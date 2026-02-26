# CoRal Dataset Reference

Source: [CoRal-project/coral-v3](https://huggingface.co/datasets/CoRal-project/coral-v3) on HuggingFace.

## Overview

Danish ASR dataset covering **dialects, accents, genders, and age groups**. Two subsets: read-aloud (scripted) and conversation (spontaneous).

License: OpenRAIL-D — open for most uses, **except** speech synthesis and biometric person identification.

Access requires HuggingFace login + data sharing agreement.

## Collection Methodology

*Based on presentations at the CoRal 2026 conference.*

Data was collected by the CoRal project (Alexandra Institute and collaborators). The collection methodology was designed to maximize dialect diversity across Denmark:

- **Collection sites:** Primarily public libraries across Denmark, chosen as cultural gathering points that attract diverse local populations
- **Participants:** Over 1,000 Danes donated between roughly 2 and 6 hours of their time, depending on how many recording sessions they completed
- **Read-aloud subset:** Each participant completed up to 3 x 2-hour recording sessions of scripted text
- **Conversational subset:** Pairs of speakers from the same dialect region had unstructured conversations; pairing same-dialect speakers reinforces natural dialect use
  - Note: Conversation participants did not know each other beforehand, which creates a particular conversational pattern
- **Geographic coverage:** Data was collected from across Denmark, with some gaps on the west coast of Jutland
- **Validation:** Read-aloud recordings were originally planned for manual validation but were validated electronically due to the scale of the task. All conversations were manually reviewed and segmented

### Data Volume

The target was approximately 1,000 hours of recorded material. The final released dataset contains **~710 hours** after quality filtering — roughly a 20% reduction from removing problematic audio clips.

### Dialect Classification

Dialect information was captured through multiple signals per speaker:

- Recording location (where the interview took place)
- Current residential address
- Self-reported dialect ("which dialect do you think you speak?")
- Postal code of school attendance
- Childhood region
- Many subcategories available in metadata

### Challenges in Data Collection

- **Legal framework:** Datatilsynet (Danish Data Protection Agency) has a restrictive stance on data for AI development under GDPR. A new legislative proposal is underway to provide legal basis for projects like CoRal (presented by Marlene Winter, DLA Piper)
- **Perfectionism:** Some participants wanted to review and re-record their own recordings. This option was removed to prevent self-censoring of natural speech
- **Non-native speakers:** Recruiting speakers with Danish as a second language was difficult. Solved partly through university students
- **Contracts:** Individual contracts were established with each participant after consulting with Datatilsynet

## Dataset Versions

| Version | Status | Notes |
|---|---|---|
| CoRal-v2 | Released | Previous version |
| CoRal-v3 | Released (Feb 2026) | Used in this project |
| CoRal-v3.1 | Expected | Upcoming |

This project uses **v3**, released at the CoRal 2026 conference. Note: the subset previously called `conversational` in v2 is now called `conversation` in v3.

## Splits & Statistics

*Note: Split statistics below are approximate figures from the HuggingFace dataset card. The total (~498h across the tables) is less than the full ~710h corpus; the remainder reflects audio filtered out during quality processing or not yet finalised in individual splits.*

### read_aloud

| Split | Duration | Speakers | Unique Sentences |
|---|---|---|---|
| train | 425.90 h | 657 | 250,108 |
| validation | 3.48 h | 11 | 2,046 |
| test | 17.30 h | 35 | 9,123 |

### conversation

| Split | Duration | Speakers |
|---|---|---|
| train | 48.85 h | 160 |
| validation | 1.16 h | 4 |
| test | 1.40 h | 5 |

## Data Fields

| Field | Type | Description |
|---|---|---|
| `id_recording` | string | Unique recording ID |
| `id_sentence` | string | Unique sentence ID |
| `id_speaker` | string | Unique speaker ID |
| `text` | string | Transcription |
| `audio` | Audio | Audio waveform + sample rate |
| `age` | string | Speaker age |
| `gender` | string | male / female / non-binary |
| `dialect` | string | Self-reported dialect |
| `country_birth` | string | Speaker's birth country |
| `location` | string | Recording location |
| `location_roomdim` | string | Room dimensions |
| `noise_level` | float | Ambient noise (dB) |
| `noise_type` | string | Noise exposure type |
| `validated` | bool | Manual validation status |
| `asr_prediction` | string | ASR model output |
| `asr_wer` | float | Word Error Rate |
| `asr_cer` | float | Character Error Rate |

## Dialect Distribution (read_aloud train)

| Dialect | % |
|---|---|
| Ostjysk | 26.6 |
| Sjaellandsk | 15.7 |
| Nordjysk | 14.7 |
| Kobenhavnsk | 13.6 |
| Vestjysk | 11.7 |
| Sonderjysk | 4.9 |
| Bornholmsk | 4.4 |
| Fynsk | 4.4 |
| Non-native | 4.0 |
| Sydomaal | 0.2 |

## Demographics (read_aloud train)

- **Gender:** Female 70.6%, Male 27.5%, Non-binary 2.0%
  - Note: The overall collection has ~65/35 female/male split according to CoRal conference presentation
- **Age:** 50+ 56.4%, 25-49 37.2%, 0-24 6.4%

### Known Representation Gaps

- **Lolland-Falster:** Underrepresented in the dataset
- **West coast of Jutland:** Some coverage gaps
- **Non-native speakers:** Limited, partly addressed via university students
- **Audio conditions:** All recordings are studio-quality. Companies using CoRal have noted a lack of "real-life" audio conditions (bad microphones, echo, background noise). Data augmentation during training can partly address this

## Industry Feedback

*From Sif (anthropologist, Alexandra Institute) at the CoRal 2026 conference:*

- Companies report that CoRal captures the Danish language niche better than Microsoft's offerings
- Dataset is used both for model training and for validation/benchmarking
- Companies value the diversity of the dataset and audio quality
- Wish list: more "real-life" noisy audio, and domain-specific data with specialized terminology

## Loading

```python
from datasets import load_dataset

coral_read = load_dataset("CoRal-project/coral-v3", "read_aloud")
coral_conv = load_dataset("CoRal-project/coral-v3", "conversation")

sample = coral_read["train"][0]
audio_array = sample["audio"]["array"]      # numpy float32 waveform
sample_rate = sample["audio"]["sampling_rate"]  # 48000 Hz (needs resample to 16kHz)
text = sample["text"]
dialect = sample["dialect"]
```

## Key Notes for Our Project

- **Both subsets used:** `read_aloud` + `conversation`
- Audio sample rate is **48kHz** — must be resampled to **16kHz** for omnilingual ASR
- Language code for omnilingual ASR: **`dan_Latn`**
- Text field: `text` — needs normalization (lowercasing, punctuation removal) via omnilingual ASR's `text_normalize()`
- Split naming: HF uses `validation`, omnilingual ASR expects `dev` — map during Parquet conversion
- Rich metadata (dialect, age, gender) enables per-group evaluation and fairness analysis
- Conversational subset adds spontaneous speech diversity (different acoustic conditions, disfluencies)
- CoRal team's own benchmarks on coral_v3-conversation show CER of 15-25% for their fine-tuned models

## Related Pre-trained Models

Models already trained on CoRal (can serve as comparison baselines):

- `roest-wav2vec2-315m-v2` (315M params)
- `roest-wav2vec2-1B-v2` (1.0B params)
- `roest-wav2vec2-2B-v2` (2.0B params)

The CoRal team also trained Whisper models — their largest Whisper model outperformed all comparable Danish ASR models on their benchmarks.

### Dialect-Specific Training Findings (CoRal Conference)

Research presented at the conference explored whether dialect-specific models outperform general models:

- **Fynsk:** A model trained on all dialects actually performed better than a dialect-focused model (possibly because Fynsk is close to rigsdansk)
- **Sonderjysk:** A dialect-focused model performed better than a general model of the same data size, but the full-data general model (with more total training data) still performed best
- **Conclusion:** Data volume and diversity generally provide better robustness for ASR models. Using all available data is recommended over restricting to a single dialect
