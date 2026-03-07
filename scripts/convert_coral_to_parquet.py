"""Convert CoRal-v3 HuggingFace dataset to omnilingual ASR Parquet format.

Reads CoRal-v3 from HuggingFace, resamples audio to 16kHz, encodes as FLAC,
normalizes text, and writes Parquet files following the omnilingual ASR directory hierarchy.

Usage:
    uv run python scripts/convert_coral_to_parquet.py --subset all
    uv run python scripts/convert_coral_to_parquet.py --subset read_aloud --max-samples 50
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torchaudio
from loguru import logger

SUBSETS = {
    "read_aloud": "coral_v3_read_aloud",
    "conversation": "coral_v3_conversation",
}
SPLIT_MAP = {"train": "train", "val": "dev", "test": "test"}
ROW_GROUP_SIZE = 100
TARGET_SR = 16000
LANGUAGE = "dan_Latn"
MAX_SKIP_RATE = 0.05

SCHEMA = pa.schema(
    [
        ("text", pa.string()),
        ("audio_bytes", pa.list_(pa.int8())),
        ("audio_size", pa.int64()),
        ("corpus", pa.dictionary(pa.int32(), pa.string())),
        ("split", pa.dictionary(pa.int32(), pa.string())),
        ("language", pa.dictionary(pa.int32(), pa.string())),
    ]
)


def normalize_text(text: str) -> str:
    """Normalize text using omnilingual ASR text normalizer."""
    return _get_text_normalize()(text, "dan", lower_case=True, remove_numbers=True)


def _get_text_normalize():
    """Lazy-load and cache the text_normalize function."""
    global _text_normalize_fn  # noqa: PLW0603
    if _text_normalize_fn is None:
        from omnilingual_asr.data.text_tools import text_normalize

        _text_normalize_fn = text_normalize
    return _text_normalize_fn


_text_normalize_fn = None


def process_audio(audio_dict: dict) -> tuple[np.ndarray, int]:
    """Resample audio to 16kHz and encode as FLAC.

    Returns (int8_array, audio_size) where audio_size is the decoded waveform length.
    """
    array = audio_dict["array"]
    sr = audio_dict["sampling_rate"]

    waveform = torch.tensor(array, dtype=torch.float32).unsqueeze(0)
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

    audio_size = waveform.shape[1]

    buffer = io.BytesIO()
    sf.write(buffer, waveform.squeeze(0).numpy(), TARGET_SR, format="FLAC")
    flac_bytes = buffer.getvalue()

    return np.frombuffer(flac_bytes, dtype=np.int8), audio_size


def write_parquet(rows: list[dict], path: Path) -> None:
    """Write rows to a Parquet file with the required schema."""
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {
        "text": pa.array([r["text"] for r in rows], type=pa.string()),
        "audio_bytes": pa.array([r["audio_bytes"] for r in rows], type=pa.list_(pa.int8())),
        "audio_size": pa.array([r["audio_size"] for r in rows], type=pa.int64()),
        "corpus": pa.array([r["corpus"] for r in rows]).dictionary_encode(),
        "split": pa.array([r["split"] for r in rows]).dictionary_encode(),
        "language": pa.array([r["language"] for r in rows]).dictionary_encode(),
    }
    table = pa.table(arrays, schema=SCHEMA)
    pq.write_table(table, path, row_group_size=ROW_GROUP_SIZE)


def convert_split(
    hf_subset: str,
    corpus_name: str,
    hf_split: str,
    parquet_split: str,
    output_dir: Path,
    rows_per_file: int,
    max_samples: int | None = None,
    cache_dir: str | None = None,
) -> dict:
    """Convert one HF split to Parquet part files.

    Returns stats dict with num_samples and total_audio_seconds.
    """
    from datasets import load_dataset

    logger.info(f"Loading {hf_subset}/{hf_split}...")
    try:
        load_kwargs: dict[str, object] = {}
        if cache_dir is not None:
            load_kwargs["cache_dir"] = cache_dir
        ds = load_dataset("CoRal-project/coral-v3", hf_subset, split=hf_split, **load_kwargs)
    except Exception as e:
        logger.error(
            f"Failed to load dataset {hf_subset}/{hf_split}: {e}. "
            "Check your network connection and HF authentication "
            "(run 'huggingface-cli login' or set HF_TOKEN env var)."
        )
        raise

    split_dir = output_dir / f"corpus={corpus_name}" / f"split={parquet_split}" / f"language={LANGUAGE}"

    rows: list[dict] = []
    part_idx = 0
    total_samples = 0
    total_audio_seconds = 0.0
    skipped = 0
    if max_samples is not None:
        ds = ds.select(range(min(len(ds), max_samples)))

    total_to_process = len(ds)

    for i, sample in enumerate(ds):
        try:
            normalized_text = normalize_text(sample["text"])
            audio_int8, audio_size = process_audio(sample["audio"])
        except (ValueError, RuntimeError, OSError, KeyError, TypeError, AttributeError) as e:
            logger.warning(f"Skipping sample {i} in {hf_subset}/{hf_split}: {type(e).__name__}: {e}")
            skipped += 1
            continue

        rows.append(
            {
                "text": normalized_text,
                "audio_bytes": audio_int8,
                "audio_size": audio_size,
                "corpus": corpus_name,
                "split": parquet_split,
                "language": LANGUAGE,
            }
        )
        total_audio_seconds += audio_size / TARGET_SR
        total_samples += 1

        if len(rows) >= rows_per_file:
            part_path = split_dir / f"part-{part_idx:05d}.parquet"
            write_parquet(rows, part_path)
            logger.info(f"Wrote {len(rows)} rows to {part_path}")
            rows = []
            part_idx += 1

        if (i + 1) % 1000 == 0:
            logger.info(f"  Processed {i + 1}/{total_to_process} samples...")

    # Write remaining rows
    if rows:
        part_path = split_dir / f"part-{part_idx:05d}.parquet"
        write_parquet(rows, part_path)
        logger.info(f"Wrote {len(rows)} rows to {part_path}")

    # Check skip rate
    if total_to_process > 0 and skipped / total_to_process > MAX_SKIP_RATE:
        raise RuntimeError(
            f"Skip rate {skipped}/{total_to_process} ({skipped / total_to_process:.1%}) "
            f"exceeds maximum allowed {MAX_SKIP_RATE:.0%} for {hf_subset}/{hf_split}"
        )

    logger.info(
        f"Done {corpus_name}/{parquet_split}: {total_samples} samples, "
        f"{total_audio_seconds:.1f}s audio, {skipped} skipped"
    )
    return {
        "corpus": corpus_name,
        "language": LANGUAGE,
        "split": parquet_split,
        "num_samples": total_samples,
        "total_audio_seconds": round(total_audio_seconds, 2),
    }


def write_stats_tsv(stats: list[dict], path: Path) -> None:
    """Write language distribution stats TSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path.open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["corpus", "language", "split", "num_samples", "total_audio_seconds"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(stats)
    logger.info(f"Wrote stats to {path}")


def main() -> None:
    # Early check for omnilingual_asr dependency
    try:
        from omnilingual_asr.data.text_tools import text_normalize  # noqa: F401
    except ImportError:
        logger.error("omnilingual-asr not installed. Run: uv sync --group omni")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Convert CoRal-v3 to omnilingual ASR Parquet format")
    parser.add_argument(
        "--subset",
        choices=["read_aloud", "conversation", "all"],
        default="all",
        help="Which subset(s) to convert",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/parquet/version=0"),
        help="Output directory for Parquet files",
    )
    parser.add_argument(
        "--rows-per-file",
        type=int,
        default=5000,
        help="Number of samples per Parquet part file",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per split (for testing)",
    )
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="Skip generating language_distribution_0.tsv",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory (default: HF_HOME or ~/.cache/huggingface)",
    )
    args = parser.parse_args()

    subsets_to_process = SUBSETS if args.subset == "all" else {args.subset: SUBSETS[args.subset]}
    all_stats: list[dict] = []

    for hf_subset, corpus_name in subsets_to_process.items():
        for hf_split, parquet_split in SPLIT_MAP.items():
            stats = convert_split(
                hf_subset=hf_subset,
                corpus_name=corpus_name,
                hf_split=hf_split,
                parquet_split=parquet_split,
                output_dir=args.output_dir,
                rows_per_file=args.rows_per_file,
                max_samples=args.max_samples,
                cache_dir=args.cache_dir,
            )
            all_stats.append(stats)

    if not args.skip_stats:
        stats_path = args.output_dir / "language_distribution_0.tsv"
        write_stats_tsv(all_stats, stats_path)

    logger.info("Conversion complete!")


if __name__ == "__main__":
    main()
