# Data Preparation: CoRal-v3 Preprocessing

Unified preprocessing pipeline for CoRal-v3. Resamples audio (48→16kHz) and FLAC-encodes **once**, producing output for all three models.

## Quick Start

```bash
# Both formats (fairseq2 + universal)
invoke data.preprocess --subset all --target all

# Universal only (Wav2Vec2/Whisper baselines, no omnilingual-asr dep)
invoke data.preprocess --subset read_aloud --target universal

# fairseq2 only (equivalent to old convert-parquet)
invoke data.preprocess --subset all --target fairseq2

# Small test run
invoke data.preprocess --subset read_aloud --target universal --max-samples 50
```

Or directly:
```bash
uv run python -m danish_asr.preprocessing --subset all --target all --max-samples 50
```

## Two Output Formats

| | fairseq2 format | Universal format |
|---|---|---|
| Text | Raw (no normalization — see below) | Raw |
| Audio | `binary` (FLAC bytes) | `binary` (FLAC bytes) |
| Splits | `validation` → `dev` | Keep HF naming (`validation`) |
| Metadata | None (strict schema) | speaker_id, gender, age, dialect |
| Use | omniASR training | Wav2Vec2/Whisper baselines |

## Output Directory Structure

```
data/
├── parquet/version=0/                     # fairseq2
│   ├── corpus=coral_v3_read_aloud/
│   │   ├── split=train/language=dan_Latn/
│   │   ├── split=dev/language=dan_Latn/
│   │   └── split=test/language=dan_Latn/
│   ├── corpus=coral_v3_conversation/...
│   └── language_distribution_0.tsv
└── preprocessed/                          # universal
    ├── read_aloud/
    │   ├── train/part-00000.parquet
    │   ├── validation/part-00000.parquet
    │   └── test/part-00000.parquet
    └── conversation/...
```

Using separate corpus names (`coral_v3_read_aloud`, `coral_v3_conversation`) lets the fairseq2 dataloader weight them via `beta_corpus`.

## Using Preprocessed Data with Baselines

After preprocessing, set `use_preprocessed: true` in `configs/data/coral.yaml`:

```yaml
use_preprocessed: true
preprocessed_dir: data/preprocessed
```

This makes `CoRalDataModule` load from Parquet via `PreprocessedCoRalDataset` instead of resampling on-the-fly from HuggingFace each epoch.

### Required Schema (PyArrow)

| Column | PyArrow Type | Description |
|---|---|---|
| `text` | `pa.string()` | Raw transcription (no normalization needed — see Text Normalization section) |
| `audio_bytes` | `pa.binary()` | Compressed audio (FLAC bytes) |
| `audio_size` | `pa.int64()` | Decoded waveform length (samples). Duration = `audio_size / 16_000` |
| `corpus` | `pa.dictionary(pa.int32(), pa.string())` | Dataset name: `"coral_v3"` |
| `split` | `pa.dictionary(pa.int32(), pa.string())` | Partition: `"train"`, `"dev"`, `"test"` |
| `language` | `pa.dictionary(pa.int32(), pa.string())` | Language code: `"dan_Latn"` |

**Parquet settings:** `row_group_size=100` (required for efficient shuffling + memory control).

> **Note:** `audio_bytes` changed from `pa.list_(pa.int8())` to `pa.binary()` in omnilingual-asr 0.2.0.

## Field Mapping: CoRal-v3 to Parquet

| Target Field | Source | Transformation |
|---|---|---|
| `text` | `sample["text"]` | None — stored as-is (see Text Normalization section) |
| `audio_bytes` | `sample["audio"]["array"]` | Resample 48kHz to 16kHz, encode to FLAC, store as `binary` |
| `audio_size` | computed | Length of decoded 16kHz waveform (= `len(resampled_array)`) |
| `corpus` | subset name | `"coral_v3_read_aloud"` or `"coral_v3_conversation"` |
| `split` | split name | Map: `"train"` to `"train"`, `"validation"` to `"dev"`, `"test"` to `"test"` |
| `language` | constant | `"dan_Latn"` |

## Text Normalization

**No text normalization is applied.** Raw CoRal-v3 `text` is stored as-is in both the fairseq2 and universal Parquet formats.

### Why not normalize?

This project targets `omniASR_CTC_300M_v2`, which uses `omniASR_tokenizer_written_v2` — a tokenizer specifically designed for **written text**. Unlike the older `omniASR_tokenizer_v1` (used by `omniASR_CTC_300M`), the written tokenizer natively handles:
- Mixed case (uppercase and lowercase, including ÆØÅ)
- Digits (`42`, `CO2`, `9 meter`)
- Punctuation (`,`, `.`, `!`, `bl.a.`)

We verified this empirically: every character in CoRal-v3's `text` field tokenizes to a distinct, non-UNK token ID.

### Historical context (do not reintroduce)

`omnilingual-asr 0.1.0` exposed `omnilingual_asr.data.text_tools.text_normalize()`, which lowercased text and removed digits and punctuation. This was required for `omniASR_tokenizer_v1` because that tokenizer mapped all uppercase letters to UNK. **This function was removed in 0.2.0** because the v2 tokenizer made it unnecessary.

Do not add text normalization back unless you deliberately switch to a v1 model and its tokenizer.

## Audio Processing

```python
import io
import soundfile as sf
import torch
import torchaudio

def process_audio(audio_dict):
    """Convert HF audio dict to omnilingual ASR format."""
    array = audio_dict["array"]  # numpy float32
    sr = audio_dict["sampling_rate"]  # 48000

    # Resample to 16kHz
    waveform = torch.tensor(array, dtype=torch.float32).unsqueeze(0)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    audio_size = waveform.shape[1]  # decoded waveform length

    # Encode to FLAC bytes — stored as pa.binary() in Parquet
    buffer = io.BytesIO()
    sf.write(buffer, waveform.squeeze(0).numpy(), 16000, format="FLAC")
    flac_bytes = buffer.getvalue()

    return flac_bytes, audio_size
```

## Conversion Script Outline

`scripts/convert_coral_to_parquet.py`:

```python
"""Convert CoRal-v3 HuggingFace dataset to omnilingual ASR Parquet format."""

from datasets import load_dataset
import pyarrow as pa
import pyarrow.parquet as pq

SUBSETS = {
    "read_aloud": "coral_v3_read_aloud",
    "conversation": "coral_v3_conversation",
}
SPLIT_MAP = {"train": "train", "validation": "dev", "test": "test"}
ROWS_PER_FILE = 5000  # samples per Parquet part file
ROW_GROUP_SIZE = 100

SCHEMA = pa.schema([
    ("text", pa.string()),
    ("audio_bytes", pa.list_(pa.int8())),
    ("audio_size", pa.int64()),
    ("corpus", pa.dictionary(pa.int32(), pa.string())),
    ("split", pa.dictionary(pa.int32(), pa.string())),
    ("language", pa.dictionary(pa.int32(), pa.string())),
])

for hf_subset, corpus_name in SUBSETS.items():
    for hf_split, parquet_split in SPLIT_MAP.items():
        ds = load_dataset("CoRal-project/coral-v3", hf_subset, split=hf_split)

        # Process in chunks → write part-XXXXX.parquet files
        # Each file: ROWS_PER_FILE samples, row_group_size=100
        # Output: data/parquet/version=0/corpus={corpus_name}/split={parquet_split}/language=dan_Latn/
        ...
```

## Dataset Stats File

After conversion, generate `language_distribution_0.tsv` required by the training recipe:

```bash
python hf_dataset_ingestion_example.py compute_stats \
    data/parquet/version=0 \
    data/parquet/version=0/language_distribution_0.tsv
```

Or create it manually — it's a TSV with columns: `corpus`, `language`, `split`, `num_samples`, `total_audio_seconds`.

Example for our single-language, two-corpus setup (sample counts TBD after download):

```tsv
corpus	language	split	num_samples	total_audio_seconds
coral_v3_read_aloud	dan_Latn	train	TBD	TBD
coral_v3_read_aloud	dan_Latn	dev	TBD	TBD
coral_v3_read_aloud	dan_Latn	test	TBD	TBD
coral_v3_conversation	dan_Latn	train	TBD	TBD
coral_v3_conversation	dan_Latn	dev	TBD	TBD
coral_v3_conversation	dan_Latn	test	TBD	TBD
```

## fairseq2 Asset Card

Create `src/omnilingual_asr/cards/datasets/coral_v3_danish.yaml`:

```yaml
name: coral_v3_danish
dataset_family: mixture_parquet_asr_dataset
dataset_config:
  data: /path/to/data/parquet/version=0
tokenizer_ref: omniASR_tokenizer_v1
```

The `data` path must point to the directory containing the `corpus=coral_v3_*/` subdirectories.

## Verification

Test the converted dataset loads correctly:

```bash
python -m workflows.dataprep.dataloader_example \
    --dataset_path="data/parquet/version=0" \
    --split="train" \
    --num_iterations=10
```

## Estimated Storage

- CoRal-v3 both subsets raw: ~80-120 GB (HF cache, 48kHz audio)
- Parquet output (16kHz FLAC compressed): ~40-60 GB estimated
- Plan for **~180 GB** total disk space (raw + converted)

## Processing Time Estimate

- ~710h of audio at ~1-2x realtime processing: **8-15 hours** on a single machine
- Consider using Ray for parallel processing on DTU HPC (see [dtu-hpc-setup.md](dtu-hpc-setup.md))
