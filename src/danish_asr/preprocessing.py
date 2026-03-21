"""Unified preprocessing for CoRal-v3: fairseq2 + universal Parquet formats.

Resamples audio 48→16kHz and encodes as FLAC once per sample,
then writes one or both output formats:

- **fairseq2**: strict schema for omnilingual ASR training
- **universal**: metadata-rich schema for Wav2Vec2/Whisper baselines

Usage:
    uv run python -m danish_asr.preprocessing --subset all --target all
    uv run python -m danish_asr.preprocessing --subset read_aloud --target universal --max-samples 50
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from loguru import logger

# ---------------------------------------------------------------------------
# Optional pyarrow import (only required for Parquet output)
# ---------------------------------------------------------------------------

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    _PYARROW_AVAILABLE = True
except ImportError:
    pa = None  # type: ignore[assignment]
    pc = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]
    _PYARROW_AVAILABLE = False


def _require_pyarrow():
    """Return (pa, pc, pq) or raise a clear ImportError if pyarrow is missing."""
    if not _PYARROW_AVAILABLE:
        msg = (
            "pyarrow is required for Parquet output but is not installed.\n"
            "Install it with one of the following commands:\n"
            "  uv add pyarrow\n"
            "  or sync the 'omni' dependency group: uv sync --group omni"
        )
        raise ImportError(msg)
    return pa, pc, pq  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUBSETS = {
    "read_aloud": "coral_v3_read_aloud",
    "conversation": "coral_v3_conversation",
}
SPLIT_MAP = {"train": "train", "validation": "dev", "test": "test"}
HF_SPLITS = ("train", "validation", "test")
TARGET_SR = 16000
LANGUAGE = "dan_Latn"
ROW_GROUP_SIZE = 100
MAX_SKIP_RATE = 0.05
METADATA_FIELDS = ("speaker_id", "gender", "age", "dialect")

# ---------------------------------------------------------------------------
# Schemas (only defined when pyarrow is available; None otherwise)
# ---------------------------------------------------------------------------

if _PYARROW_AVAILABLE:
    FAIRSEQ2_SCHEMA = pa.schema(  # type: ignore[union-attr]
        [
            ("text", pa.string()),  # type: ignore[union-attr]
            ("audio_bytes", pa.binary()),  # type: ignore[union-attr]
            ("audio_size", pa.int64()),  # type: ignore[union-attr]
            ("corpus", pa.dictionary(pa.int32(), pa.string())),  # type: ignore[union-attr]
            ("split", pa.dictionary(pa.int32(), pa.string())),  # type: ignore[union-attr]
            ("language", pa.dictionary(pa.int32(), pa.string())),  # type: ignore[union-attr]
        ]
    )
    UNIVERSAL_SCHEMA = pa.schema(  # type: ignore[union-attr]
        [
            ("text", pa.string()),  # type: ignore[union-attr]
            ("audio", pa.binary()),  # type: ignore[union-attr]
            ("audio_samples", pa.int64()),  # type: ignore[union-attr]
            ("duration_s", pa.float32()),  # type: ignore[union-attr]
            ("subset", pa.string()),  # type: ignore[union-attr]
            ("split", pa.string()),  # type: ignore[union-attr]
            ("speaker_id", pa.string()),  # type: ignore[union-attr]
            ("gender", pa.string()),  # type: ignore[union-attr]
            ("age", pa.string()),  # type: ignore[union-attr]
            ("dialect", pa.string()),  # type: ignore[union-attr]
        ]
    )
else:
    FAIRSEQ2_SCHEMA = None  # type: ignore[assignment]
    UNIVERSAL_SCHEMA = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Audio processing
# ---------------------------------------------------------------------------


def process_audio(audio_dict: dict) -> tuple[bytes, int]:
    """Resample to 16kHz and FLAC-encode once.

    Returns:
        (flac_bytes, audio_samples)
    """
    array = audio_dict["array"]
    sr = audio_dict["sampling_rate"]

    waveform = torch.tensor(array, dtype=torch.float32).unsqueeze(0)
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

    audio_samples = waveform.shape[1]

    buffer = io.BytesIO()
    sf.write(buffer, waveform.squeeze(0).numpy(), TARGET_SR, format="FLAC")
    flac_bytes = buffer.getvalue()

    return flac_bytes, audio_samples


# ---------------------------------------------------------------------------
# Parquet writers
# ---------------------------------------------------------------------------


def write_fairseq2_parquet(rows: list[dict], path: Path) -> None:
    """Write rows to a fairseq2-format Parquet file."""
    pa, pc, pq = _require_pyarrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "text": pa.array([r["text"] for r in rows], type=pa.string()),
        "audio_bytes": pa.array([r["audio_bytes"] for r in rows], type=pa.binary()),
        "audio_size": pa.array([r["audio_size"] for r in rows], type=pa.int64()),
        "corpus": pa.array([r["corpus"] for r in rows]).dictionary_encode(),
        "split": pa.array([r["split"] for r in rows]).dictionary_encode(),
        "language": pa.array([r["language"] for r in rows]).dictionary_encode(),
    }
    table = pa.table(arrays, schema=FAIRSEQ2_SCHEMA)
    tmp_path = path.with_suffix(".tmp")
    pq.write_table(table, tmp_path, row_group_size=ROW_GROUP_SIZE)
    tmp_path.rename(path)


def write_universal_parquet(rows: list[dict], path: Path) -> None:
    """Write rows to a universal-format Parquet file."""
    pa, pc, pq = _require_pyarrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "text": pa.array([r["text"] for r in rows], type=pa.string()),
        "audio": pa.array([r["audio"] for r in rows], type=pa.binary()),
        "audio_samples": pa.array([r["audio_samples"] for r in rows], type=pa.int64()),
        "duration_s": pa.array([r["duration_s"] for r in rows], type=pa.float32()),
        "subset": pa.array([r["subset"] for r in rows], type=pa.string()),
        "split": pa.array([r["split"] for r in rows], type=pa.string()),
        "speaker_id": pa.array([r["speaker_id"] for r in rows], type=pa.string()),
        "gender": pa.array([r["gender"] for r in rows], type=pa.string()),
        "age": pa.array([r["age"] for r in rows], type=pa.string()),
        "dialect": pa.array([r["dialect"] for r in rows], type=pa.string()),
    }
    table = pa.table(arrays, schema=UNIVERSAL_SCHEMA)
    tmp_path = path.with_suffix(".tmp")
    pq.write_table(table, tmp_path, row_group_size=ROW_GROUP_SIZE)
    tmp_path.rename(path)


# ---------------------------------------------------------------------------
# Core conversion loop
# ---------------------------------------------------------------------------


def convert_split(
    hf_subset: str,
    corpus_name: str,
    hf_split: str,
    targets: set[str],
    fairseq2_dir: Path | None = None,
    universal_dir: Path | None = None,
    rows_per_file: int = 5000,
    max_samples: int | None = None,
    cache_dir: str | None = None,
    revision: str | None = None,
) -> dict:
    """Convert one HF split, writing enabled target formats.

    Returns stats dict with num_samples and total_audio_seconds.
    """
    from datasets import load_dataset

    parquet_split = SPLIT_MAP[hf_split]

    logger.info(f"Loading {hf_subset}/{hf_split}...")
    try:
        load_kwargs: dict[str, object] = {}
        if cache_dir is not None:
            load_kwargs["cache_dir"] = cache_dir
        if revision is not None:
            load_kwargs["revision"] = revision
        ds = load_dataset("CoRal-project/coral-v3", hf_subset, split=hf_split, **load_kwargs)  # nosec B615
    except Exception as e:
        logger.error(
            f"Failed to load dataset {hf_subset}/{hf_split}: {e}. "
            "Check your network connection and HF authentication "
            "(run 'huggingface-cli login' or set HF_TOKEN env var)."
        )
        raise

    if max_samples is not None:
        ds = ds.select(range(min(len(ds), max_samples)))

    total_to_process = len(ds)

    # Prepare output dirs
    fairseq2_split_dir = None
    if "fairseq2" in targets and fairseq2_dir is not None:
        fairseq2_split_dir = fairseq2_dir / f"corpus={corpus_name}" / f"split={parquet_split}" / f"language={LANGUAGE}"

    universal_split_dir = None
    if "universal" in targets and universal_dir is not None:
        universal_split_dir = universal_dir / hf_subset / hf_split

    fairseq2_rows: list[dict] = []
    universal_rows: list[dict] = []
    part_idx_f = 0
    part_idx_u = 0
    total_samples = 0
    total_audio_seconds = 0.0
    skipped = 0

    for i, sample in enumerate(ds):
        try:
            flac_bytes, audio_samples = process_audio(sample["audio"])
        except (ValueError, RuntimeError, OSError, KeyError, TypeError, AttributeError) as e:
            logger.warning(f"Skipping sample {i} in {hf_subset}/{hf_split}: {type(e).__name__}: {e}")
            skipped += 1
            continue

        text = sample.get("text", sample.get("sentence", ""))

        # Build fairseq2 row — raw text, no normalization needed.
        # omniASR_CTC_300M_v2 uses omniASR_tokenizer_written_v2, which natively
        # handles mixed case, digits, punctuation, and Danish characters.
        if fairseq2_split_dir is not None:
            fairseq2_rows.append(
                {
                    "text": text,
                    "audio_bytes": flac_bytes,
                    "audio_size": audio_samples,
                    "corpus": corpus_name,
                    "split": parquet_split,
                    "language": LANGUAGE,
                }
            )

        # Build universal row
        if universal_split_dir is not None:
            universal_rows.append(
                {
                    "text": text,
                    "audio": flac_bytes,
                    "audio_samples": audio_samples,
                    "duration_s": audio_samples / TARGET_SR,
                    "subset": hf_subset,
                    "split": hf_split,
                    "speaker_id": str(sample.get("speaker_id", "")),
                    "gender": str(sample.get("gender", "")),
                    "age": str(sample.get("age", "")),
                    "dialect": str(sample.get("dialect", "")),
                }
            )

        total_audio_seconds += audio_samples / TARGET_SR
        total_samples += 1

        # Flush fairseq2 rows
        if fairseq2_split_dir is not None and len(fairseq2_rows) >= rows_per_file:
            part_path = fairseq2_split_dir / f"part-{part_idx_f:05d}.parquet"
            write_fairseq2_parquet(fairseq2_rows, part_path)
            logger.info(f"Wrote {len(fairseq2_rows)} rows to {part_path}")
            fairseq2_rows = []
            part_idx_f += 1

        # Flush universal rows
        if universal_split_dir is not None and len(universal_rows) >= rows_per_file:
            part_path = universal_split_dir / f"part-{part_idx_u:05d}.parquet"
            write_universal_parquet(universal_rows, part_path)
            logger.info(f"Wrote {len(universal_rows)} rows to {part_path}")
            universal_rows = []
            part_idx_u += 1

        if (i + 1) % 1000 == 0:
            logger.info(f"  Processed {i + 1}/{total_to_process} samples...")

    # Write remaining rows
    if fairseq2_rows and fairseq2_split_dir is not None:
        part_path = fairseq2_split_dir / f"part-{part_idx_f:05d}.parquet"
        write_fairseq2_parquet(fairseq2_rows, part_path)
        logger.info(f"Wrote {len(fairseq2_rows)} rows to {part_path}")

    if universal_rows and universal_split_dir is not None:
        part_path = universal_split_dir / f"part-{part_idx_u:05d}.parquet"
        write_universal_parquet(universal_rows, part_path)
        logger.info(f"Wrote {len(universal_rows)} rows to {part_path}")

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


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def write_stats_tsv(stats: list[dict], path: Path) -> None:
    """Write language distribution stats TSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["corpus", "language", "split", "num_samples", "total_audio_seconds"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(stats)
    logger.info(f"Wrote stats to {path}")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS = {
    "text",
    "audio",
    "audio_samples",
    "duration_s",
    "subset",
    "split",
    "speaker_id",
    "gender",
    "age",
    "dialect",
}
# Approximate total hours per subset (train+val+test combined); tolerance ±20h
EXPECTED_SUBSET_HOURS = {"read_aloud": 542, "conversation": 155}


def verify_preprocessed_data(preprocessed_dir: str = "data/preprocessed") -> None:
    """Verify schema, row counts, audio duration, and audio integrity of preprocessed Parquet files.

    Checks all 6 split directories in `preprocessed_dir` ({read_aloud,conversation}/{train,validation,test}).
    Exits with code 1 if any check fails.
    """
    pa, pc, pq = _require_pyarrow()
    base = Path(preprocessed_dir)
    splits = ("train", "validation", "test")
    subsets = ("read_aloud", "conversation")

    errors: list[str] = []
    summary: list[tuple[str, str, int, float]] = []

    for subset in subsets:
        subset_hours = 0.0
        for split in splits:
            split_dir = base / subset / split
            if not split_dir.exists():
                errors.append(f"MISSING dir: {split_dir}")
                continue
            files = sorted(split_dir.glob("*.parquet"))
            if not files:
                errors.append(f"NO FILES in {split_dir}")
                continue

            # Schema check (read from first file only)
            schema = pq.read_schema(files[0])
            missing_cols = EXPECTED_COLUMNS - set(schema.names)
            if missing_cols:
                errors.append(f"SCHEMA {subset}/{split}: missing columns {missing_cols}")

            # Row count + duration + null checks — process in chunks to limit memory
            total_rows = sum(pq.read_metadata(f).num_rows for f in files)
            split_duration_s = 0.0
            null_counts: dict[str, int] = {"text": 0, "audio": 0, "duration_s": 0}
            audio_spot_checked = False
            chunk_size = max(1, len(files) // 20)

            for chunk_start in range(0, len(files), chunk_size):
                chunk_files = files[chunk_start : chunk_start + chunk_size]
                chunk_tables = [pq.read_table(f, columns=["duration_s", "text", "audio"]) for f in chunk_files]
                chunk = pa.concat_tables(chunk_tables)

                split_duration_s += pc.sum(chunk["duration_s"].cast(pa.float64())).as_py()

                for col in null_counts:
                    null_counts[col] += chunk[col].null_count

                # Audio spot-check on first chunk only
                if not audio_spot_checked and len(chunk) > 0:
                    spot_bytes = chunk["audio"][0].as_py()
                    try:
                        sf.read(io.BytesIO(spot_bytes))
                    except Exception as e:
                        errors.append(f"AUDIO DECODE {subset}/{split}: {e}")
                    audio_spot_checked = True

                del chunk, chunk_tables

            split_hours_total = split_duration_s / 3600
            subset_hours += split_hours_total

            for col, nc in null_counts.items():
                if nc > 0:
                    errors.append(f"NULLS {subset}/{split} col={col}: {nc} nulls")

            summary.append((subset, split, total_rows, split_hours_total))

        expected_h = EXPECTED_SUBSET_HOURS[subset]
        if abs(subset_hours - expected_h) > 20:
            errors.append(f"DURATION {subset}: {subset_hours:.1f}h (expected ~{expected_h}h)")

    # Print summary table
    print(f"{'Subset':<15} {'Split':<8} {'Rows':>10} {'Hours':>8}")
    print("-" * 45)
    for subset, split, n, h in summary:
        print(f"{subset:<15} {split:<8} {n:>10,} {h:>7.1f}h")
    print()

    if errors:
        print("FAILURES:")
        for err in errors:
            print(f"  [FAIL] {err}")
        sys.exit(1)
    else:
        print("All checks passed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified CoRal-v3 preprocessing")
    parser.add_argument(
        "--subset",
        choices=["read_aloud", "conversation", "all"],
        default="all",
        help="Which subset(s) to convert",
    )
    parser.add_argument(
        "--target",
        choices=["fairseq2", "universal", "all"],
        default="all",
        help="Output format(s) to produce",
    )
    parser.add_argument(
        "--fairseq2-dir",
        type=Path,
        default=Path("data/parquet/version=0"),
        help="fairseq2 output directory",
    )
    parser.add_argument(
        "--universal-dir",
        type=Path,
        default=Path("data/preprocessed"),
        help="Universal output directory",
    )
    parser.add_argument(
        "--rows-per-file",
        type=int,
        default=5000,
        help="Samples per Parquet part file",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit per split (for testing)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Pin HuggingFace dataset revision (commit hash or tag) for reproducibility",
    )
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="Skip generating language_distribution_0.tsv",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    targets: set[str] = set()
    if args.target in ("fairseq2", "all"):
        targets.add("fairseq2")
    if args.target in ("universal", "all"):
        targets.add("universal")

    subsets_to_process = SUBSETS if args.subset == "all" else {args.subset: SUBSETS[args.subset]}
    all_stats: list[dict] = []

    for hf_subset, corpus_name in subsets_to_process.items():
        for hf_split in HF_SPLITS:
            stats = convert_split(
                hf_subset=hf_subset,
                corpus_name=corpus_name,
                hf_split=hf_split,
                targets=targets,
                fairseq2_dir=args.fairseq2_dir,
                universal_dir=args.universal_dir,
                rows_per_file=args.rows_per_file,
                max_samples=args.max_samples,
                cache_dir=args.cache_dir,
                revision=args.revision,
            )
            all_stats.append(stats)

    if not args.skip_stats and "fairseq2" in targets:
        stats_path = args.fairseq2_dir / "language_distribution_0.tsv"
        write_stats_tsv(all_stats, stats_path)

    logger.info("Preprocessing complete!")


if __name__ == "__main__":
    main()
