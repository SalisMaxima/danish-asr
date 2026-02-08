# Data Preparation: CoRal-v2 to Omnilingual ASR Parquet

Guide for converting CoRal-v2 (HuggingFace format) to the Parquet format required by the omnilingual ASR training pipeline.

## Target Format

The omnilingual ASR dataloader expects Parquet files in a specific directory hierarchy with a fixed PyArrow schema.

### Directory Structure

```
data/parquet/version=0/
├── corpus=coral_v2_read_aloud/
│   ├── split=train/
│   │   └── language=dan_Latn/
│   │       ├── part-00000.parquet
│   │       └── ...
│   ├── split=dev/
│   │   └── language=dan_Latn/
│   │       └── part-00000.parquet
│   └── split=test/
│       └── language=dan_Latn/
│           └── part-00000.parquet
└── corpus=coral_v2_conversational/
    ├── split=train/
    │   └── language=dan_Latn/
    │       └── part-00000.parquet
    ├── split=dev/
    │   └── language=dan_Latn/
    │       └── part-00000.parquet
    └── split=test/
        └── language=dan_Latn/
            └── part-00000.parquet
```

Using separate corpus names (`coral_v2_read_aloud`, `coral_v2_conversational`) lets the dataloader weight them via `beta_corpus`.

### Required Schema (PyArrow)

| Column | PyArrow Type | Description |
|---|---|---|
| `text` | `pa.string()` | Normalized transcription |
| `audio_bytes` | `pa.list_(pa.int8())` | Compressed audio (FLAC), stored as int8 list |
| `audio_size` | `pa.int64()` | Decoded waveform length (samples). Duration = `audio_size / 16_000` |
| `corpus` | `pa.dictionary(pa.int32(), pa.string())` | Dataset name: `"coral_v2"` |
| `split` | `pa.dictionary(pa.int32(), pa.string())` | Partition: `"train"`, `"dev"`, `"test"` |
| `language` | `pa.dictionary(pa.int32(), pa.string())` | Language code: `"dan_Latn"` |

**Parquet settings:** `row_group_size=100` (required for efficient shuffling + memory control).

## Field Mapping: CoRal-v2 → Parquet

| Target Field | Source | Transformation |
|---|---|---|
| `text` | `sample["text"]` | `text_normalize(text, "dan", lower_case=True, remove_numbers=True)` |
| `audio_bytes` | `sample["audio"]["array"]` | Resample 48kHz→16kHz, encode to FLAC, convert to `list<int8>` via `binary_to_list_int8()` |
| `audio_size` | computed | Length of decoded 16kHz waveform (= `len(resampled_array)`) |
| `corpus` | subset name | `"coral_v2_read_aloud"` or `"coral_v2_conversational"` |
| `split` | split name | Map: `"train"→"train"`, `"validation"→"dev"`, `"test"→"test"` |
| `language` | constant | `"dan_Latn"` |

## Text Normalization

The omnilingual ASR package provides `text_normalize()`:

```python
from omnilingual_asr.data.text_tools import text_normalize

normalized = text_normalize(
    text="Hej, verden! Der er 42 mennesker.",
    iso_code="dan",
    lower_case=True,
    remove_numbers=True,
    remove_brackets=False,
)
# → "hej verden der er mennesker"
```

This handles:
- Punctuation removal (language-specific rules)
- Lowercasing
- Digit-only word removal
- Language remapping

## Audio Processing

```python
import torchaudio
import io
import soundfile as sf
import numpy as np
from omnilingual_asr.data.audio_tools import binary_to_list_int8

def process_audio(audio_dict):
    """Convert HF audio dict to omnilingual ASR format."""
    array = audio_dict["array"]  # numpy float32
    sr = audio_dict["sampling_rate"]  # 48000

    # Resample to 16kHz
    waveform = torch.tensor(array, dtype=torch.float32).unsqueeze(0)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    audio_size = waveform.shape[1]  # decoded waveform length

    # Encode to FLAC bytes
    buffer = io.BytesIO()
    sf.write(buffer, waveform.squeeze(0).numpy(), 16000, format="FLAC")
    flac_bytes = buffer.getvalue()

    # Convert to list<int8> for Parquet
    # Use binary_to_list_int8() for batch processing with PyArrow,
    # or manually: list(np.frombuffer(flac_bytes, dtype=np.int8))
    audio_int8 = list(np.frombuffer(flac_bytes, dtype=np.int8))

    return audio_int8, audio_size
```

## Conversion Script Outline

`scripts/convert_coral_to_parquet.py`:

```python
"""Convert CoRal-v2 HuggingFace dataset to omnilingual ASR Parquet format."""

from datasets import load_dataset
import pyarrow as pa
import pyarrow.parquet as pq

SUBSETS = {
    "read_aloud": "coral_v2_read_aloud",
    "conversational": "coral_v2_conversational",
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
        ds = load_dataset("CoRal-project/coral-v2", hf_subset, split=hf_split)

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
    data/parquet/language_distribution_0.tsv
```

Or create it manually — it's a TSV with columns: `corpus`, `language`, `split`, `num_samples`, `total_audio_seconds`.

Example for our single-language, two-corpus setup:

```tsv
corpus	language	split	num_samples	total_audio_seconds
coral_v2_read_aloud	dan_Latn	train	250108	1533240.0
coral_v2_read_aloud	dan_Latn	dev	2046	12528.0
coral_v2_read_aloud	dan_Latn	test	9123	62280.0
coral_v2_conversational	dan_Latn	train	TBD	175860.0
coral_v2_conversational	dan_Latn	dev	TBD	4176.0
coral_v2_conversational	dan_Latn	test	TBD	5040.0
```

## fairseq2 Asset Card

Create `src/omnilingual_asr/cards/datasets/coral_v2_danish.yaml`:

```yaml
name: coral_v2_danish
dataset_family: mixture_parquet_asr_dataset
dataset_config:
  data: /path/to/data/parquet/version=0
tokenizer_ref: omniASR_tokenizer_v1
```

The `data` path must point to the directory containing the `corpus=coral_v2_*/` subdirectories.

## Verification

Test the converted dataset loads correctly:

```bash
python -m workflows.dataprep.dataloader_example \
    --dataset_path="data/parquet/version=0" \
    --split="train" \
    --num_iterations=10
```

## Estimated Storage

- CoRal-v2 both subsets raw: ~60-100 GB (HF cache, 48kHz audio)
- Parquet output (16kHz FLAC compressed): ~30-50 GB estimated
- Plan for **~150 GB** total disk space (raw + converted)

## Processing Time Estimate

- ~474h of audio at ~1-2x realtime processing: **5-10 hours** on a single machine
- Consider using Ray for parallel processing on DTU HPC (see [dtu-hpc-setup.md](dtu-hpc-setup.md))
