"""Step 2: Convert universal Parquet → fairseq2 Parquet format.

Reads preprocessed universal Parquet files and converts them to the fairseq2
schema required by omniASR training, without re-downloading or resampling audio.

For each row:
- Normalizes text via omnilingual ASR text normalizer
- Converts pa.binary() FLAC → list<int8> for fairseq2
- Maps audio_samples → audio_size
- Maps subset → corpus name, split: val → dev

Usage:
    python scripts/hpc/convert_to_fairseq2.py
    python scripts/hpc/convert_to_fairseq2.py --subset read_aloud --rows-per-file 3000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from loguru import logger

from danish_asr.preprocessing import (
    MAX_SKIP_RATE,
    normalize_text_fairseq2,
    write_fairseq2_parquet,
    write_stats_tsv,
)
from scripts.hpc.common import (
    FAIRSEQ2_DIR,
    HF_SPLITS,
    LANGUAGE,
    SPLIT_MAP,
    SUBSETS,
    UNIVERSAL_DIR,
    log_system_info,
    setup_hpc_environment,
    setup_logging,
)


def convert_split(
    subset: str,
    corpus_name: str,
    hf_split: str,
    universal_dir: Path,
    fairseq2_dir: Path,
    rows_per_file: int,
    max_samples: int | None = None,
) -> dict:
    """Convert one split from universal → fairseq2 Parquet."""
    parquet_split = SPLIT_MAP[hf_split]
    split_dir = universal_dir / subset / hf_split
    if not split_dir.exists():
        logger.error(f"Source directory not found: {split_dir}")
        sys.exit(1)

    source_files = sorted(split_dir.glob("*.parquet"))
    if not source_files:
        logger.error(f"No Parquet files in {split_dir}")
        sys.exit(1)

    out_dir = fairseq2_dir / f"corpus={corpus_name}" / f"split={parquet_split}" / f"language={LANGUAGE}"
    logger.info(f"Converting {subset}/{hf_split} → {out_dir}")

    fairseq2_rows: list[dict] = []
    part_idx = 0
    total_samples = 0
    total_audio_seconds = 0.0
    skipped = 0
    total_to_process = 0

    for source_file in source_files:
        table = pq.read_table(source_file)
        total_to_process += len(table)

        for i in range(len(table)):
            if max_samples is not None and total_samples >= max_samples:
                break

            text = table["text"][i].as_py()
            audio_bytes = table["audio"][i].as_py()
            audio_samples = table["audio_samples"][i].as_py()
            duration_s = table["duration_s"][i].as_py()

            # Normalize text
            try:
                normalized_text = normalize_text_fairseq2(text)
            except Exception as e:
                logger.warning(f"Skipping row (text normalization): {type(e).__name__}: {e}")
                skipped += 1
                continue

            if not normalized_text.strip():
                logger.debug(f"Skipping row with empty normalized text (original: {text!r})")
                skipped += 1
                continue

            # Convert binary FLAC → int8 array
            flac_int8 = np.frombuffer(audio_bytes, dtype=np.int8)

            fairseq2_rows.append(
                {
                    "text": normalized_text,
                    "audio_bytes": flac_int8,
                    "audio_size": audio_samples,
                    "corpus": corpus_name,
                    "split": parquet_split,
                    "language": LANGUAGE,
                }
            )
            total_samples += 1
            total_audio_seconds += duration_s

            # Flush every rows_per_file
            if len(fairseq2_rows) >= rows_per_file:
                part_path = out_dir / f"part-{part_idx:05d}.parquet"
                write_fairseq2_parquet(fairseq2_rows, part_path)
                logger.info(f"Wrote {len(fairseq2_rows)} rows to {part_path}")
                fairseq2_rows = []
                part_idx += 1

        del table

        if max_samples is not None and total_samples >= max_samples:
            break

    # Write remaining rows
    if fairseq2_rows:
        part_path = out_dir / f"part-{part_idx:05d}.parquet"
        write_fairseq2_parquet(fairseq2_rows, part_path)
        logger.info(f"Wrote {len(fairseq2_rows)} rows to {part_path}")

    # Check skip rate
    processed = total_samples + skipped
    if processed > 0 and skipped / processed > MAX_SKIP_RATE:
        raise RuntimeError(
            f"Skip rate {skipped}/{processed} ({skipped / processed:.1%}) "
            f"exceeds maximum allowed {MAX_SKIP_RATE:.0%} for {subset}/{hf_split}"
        )

    logger.info(
        f"Done {corpus_name}/{parquet_split}: {total_samples} samples, "
        f"{total_audio_seconds / 3600:.1f}h audio, {skipped} skipped"
    )
    return {
        "corpus": corpus_name,
        "language": LANGUAGE,
        "split": parquet_split,
        "num_samples": total_samples,
        "total_audio_seconds": round(total_audio_seconds, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert universal Parquet → fairseq2 format")
    parser.add_argument("--universal-dir", type=Path, default=UNIVERSAL_DIR, help="Universal Parquet directory")
    parser.add_argument("--fairseq2-dir", type=Path, default=FAIRSEQ2_DIR, help="fairseq2 output directory")
    parser.add_argument("--rows-per-file", type=int, default=5000, help="Rows per output Parquet file")
    parser.add_argument("--subset", choices=["read_aloud", "conversation", "all"], default="all")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit per split (for testing)")
    args = parser.parse_args()

    setup_logging("convert_to_fairseq2")
    setup_hpc_environment()
    log_system_info()

    # Early check for omnilingual_asr
    try:
        from omnilingual_asr.data.text_tools import text_normalize  # noqa: F401
    except ImportError:
        logger.error("omnilingual-asr not installed. Run: uv sync --group omni")
        sys.exit(1)

    subsets = SUBSETS if args.subset == "all" else {args.subset: SUBSETS[args.subset]}
    all_stats: list[dict] = []

    start_time = time.time()
    for subset, corpus_name in subsets.items():
        for hf_split in HF_SPLITS:
            stats = convert_split(
                subset=subset,
                corpus_name=corpus_name,
                hf_split=hf_split,
                universal_dir=args.universal_dir,
                fairseq2_dir=args.fairseq2_dir,
                rows_per_file=args.rows_per_file,
                max_samples=args.max_samples,
            )
            all_stats.append(stats)

    # Write stats TSV
    stats_path = args.fairseq2_dir / "language_distribution_0.tsv"
    write_stats_tsv(all_stats, stats_path)

    elapsed = time.time() - start_time
    logger.info(f"Conversion complete in {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
