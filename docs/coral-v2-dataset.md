# CoRal-v2 Dataset Reference

Source: [CoRal-project/coral-v2](https://huggingface.co/datasets/CoRal-project/coral-v2) on HuggingFace.

## Overview

Danish ASR dataset covering **dialects, accents, genders, and age groups**. Two subsets: read-aloud (scripted) and conversational (spontaneous).

License: OpenRAIL-D (commercial use allowed; speech synthesis and biometric ID prohibited).

Access requires HuggingFace login + data sharing agreement.

## Splits & Statistics

### read_aloud

| Split | Duration | Speakers | Unique Sentences |
|---|---|---|---|
| train | 425.90 h | 657 | 250,108 |
| validation | 3.48 h | 11 | 2,046 |
| test | 17.30 h | 35 | 9,123 |

### conversational

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
- **Age:** 50+ 56.4%, 25-49 37.2%, 0-24 6.4%

## Loading

```python
from datasets import load_dataset

coral_read = load_dataset("CoRal-project/coral-v2", "read_aloud")
coral_conv = load_dataset("CoRal-project/coral-v2", "conversational")

sample = coral_read["train"][0]
audio_array = sample["audio"]["array"]      # numpy float32 waveform
sample_rate = sample["audio"]["sampling_rate"]  # 48000 Hz (needs resample to 16kHz)
text = sample["text"]
dialect = sample["dialect"]
```

## Key Notes for Our Project

- **Both subsets used:** `read_aloud` (425h) + `conversational` (48h) = ~474h total
- Audio sample rate is **48kHz** — must be resampled to **16kHz** for omnilingual ASR
- Language code for omnilingual ASR: **`dan_Latn`**
- Text field: `text` — needs normalization (lowercasing, punctuation removal) via omnilingual ASR's `text_normalize()`
- Split naming: HF uses `validation`, omnilingual ASR expects `dev` — map during Parquet conversion
- Rich metadata (dialect, age, gender) enables per-group evaluation
- Conversational subset adds spontaneous speech diversity (different acoustic conditions, disfluencies)

## Related Pre-trained Models

Models already trained on CoRal-v2 (can serve as comparison baselines):
- `roest-wav2vec2-315m-v2` (315M params)
- `roest-wav2vec2-1B-v2` (1.0B params)
- `roest-wav2vec2-2B-v2` (2.0B params)
