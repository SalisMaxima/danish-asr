"""Repair fairseq2 Parquet files by removing in-file partition columns.

The legacy converter (scripts/convert_coral_to_parquet.py) wrote corpus, split,
and language columns *inside* the Parquet files.  These columns are already
encoded in the Hive partition directory paths (corpus=X/split=Y/language=Z/),
so PyArrow adds them again on read → duplicate columns → pandas warning:

    "DataFrame columns are not unique, some columns will be omitted."

This script reads each Parquet file, keeps only the three required columns
(text, audio_bytes, audio_size), and rewrites the file in-place.  Files that
already have the correct schema are skipped.

Usage:
    python scripts/hpc/repair_parquet_schema.py
    python scripts/hpc/repair_parquet_schema.py --fairseq2-dir /work3/$USER/data/parquet/version=0
    python scripts/hpc/repair_parquet_schema.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError as exc:
    # Provide a helpful message instead of a bare ModuleNotFoundError.
    # This script relies on the "omni" dependency group, which includes pyarrow.
    if exc.name == "pyarrow":
        print(
            "Error: This script requires the 'pyarrow' package.\n"
            "Install it by syncing the 'omni' dependency group:\n"
            "    uv sync --group omni",
            file=sys.stderr,
        )
        sys.exit(1)
    raise
from loguru import logger

from scripts.hpc.common import FAIRSEQ2_DIR, setup_logging

REQUIRED_COLUMNS = ("text", "audio_bytes", "audio_size")
PARTITION_COLUMNS = {"corpus", "split", "language"}


def repair_file(path: Path, dry_run: bool = False) -> bool:
    """Remove partition columns from a single Parquet file.

    Returns True if the file was repaired (or would be in dry-run mode).
    """
    schema = pq.read_schema(path)
    file_columns = set(schema.names)
    extra_columns = file_columns & PARTITION_COLUMNS

    if not extra_columns:
        return False

    if dry_run:
        logger.info(f"[DRY RUN] Would repair {path} (extra columns: {extra_columns})")
        return True

    table = pq.read_table(path, columns=list(REQUIRED_COLUMNS))
    tmp_path = path.with_suffix(".tmp")
    pq.write_table(table, tmp_path, row_group_size=100)
    tmp_path.rename(path)
    logger.info(f"Repaired {path} (removed {extra_columns})")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair fairseq2 Parquet schema")
    parser.add_argument(
        "--fairseq2-dir",
        type=Path,
        default=FAIRSEQ2_DIR,
        help="Root of the fairseq2 Parquet directory (default: %(default)s)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report what would be changed without modifying files")
    args = parser.parse_args()

    setup_logging("repair_parquet_schema")

    if not args.fairseq2_dir.exists():
        logger.error(f"Directory not found: {args.fairseq2_dir}")
        sys.exit(1)

    parquet_files = sorted(args.fairseq2_dir.rglob("*.parquet"))
    if not parquet_files:
        logger.error(f"No Parquet files found in {args.fairseq2_dir}")
        sys.exit(1)

    logger.info(f"Scanning {len(parquet_files)} Parquet files in {args.fairseq2_dir}")

    repaired = 0
    for pf in parquet_files:
        if repair_file(pf, dry_run=args.dry_run):
            repaired += 1

    if repaired == 0:
        logger.info("All files already have the correct schema — nothing to repair.")
    elif args.dry_run:
        logger.info(f"[DRY RUN] {repaired}/{len(parquet_files)} files would be repaired.")
    else:
        logger.info(f"Repaired {repaired}/{len(parquet_files)} files.")


if __name__ == "__main__":
    main()
