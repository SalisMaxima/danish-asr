"""Step 1: Verify universal Parquet data integrity on HPC.

Enhanced version of preprocessing.verify_preprocessed_data() with loguru logging,
FLAC spot-checks, null checks, and disk usage reporting.

Usage:
    python scripts/hpc/verify_data.py
    python scripts/hpc/verify_data.py --data-dir /work3/$USER/data/preprocessed
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
import soundfile as sf
from loguru import logger

from scripts.hpc.common import (
    SUBSETS,
    UNIVERSAL_DIR,
    log_system_info,
    setup_hpc_environment,
    setup_logging,
)

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
EXPECTED_SUBSET_HOURS = {"read_aloud": 542, "conversation": 155}
HF_SPLITS = ("train", "validation", "test")
CRITICAL_COLUMNS = ("text", "audio", "duration_s")


def verify(data_dir: Path) -> bool:
    """Run all verification checks. Returns True if all pass."""
    errors: list[str] = []
    summary: list[tuple[str, str, int, float]] = []

    for subset in SUBSETS:
        subset_hours = 0.0
        for split in HF_SPLITS:
            split_dir = data_dir / subset / split
            if not split_dir.exists():
                errors.append(f"MISSING dir: {split_dir}")
                continue

            files = sorted(split_dir.glob("*.parquet"))
            if not files:
                errors.append(f"NO FILES in {split_dir}")
                continue

            # Schema check
            schema = pq.read_schema(files[0])
            missing_cols = EXPECTED_COLUMNS - set(schema.names)
            if missing_cols:
                errors.append(f"SCHEMA {subset}/{split}: missing columns {missing_cols}")

            # Row count + duration + null checks
            total_rows = sum(pq.read_metadata(f).num_rows for f in files)
            split_duration_s = 0.0
            null_counts: dict[str, int] = dict.fromkeys(CRITICAL_COLUMNS, 0)
            audio_spot_checked = False

            for f in files:
                table = pq.read_table(f, columns=list(CRITICAL_COLUMNS))
                split_duration_s += pc.sum(table["duration_s"].cast("float64")).as_py()

                for col in CRITICAL_COLUMNS:
                    null_counts[col] += table[col].null_count

                # Spot-check: decode first FLAC sample from first file
                if not audio_spot_checked and len(table) > 0:
                    spot_bytes = table["audio"][0].as_py()
                    try:
                        sf.read(io.BytesIO(spot_bytes))
                        logger.info(f"FLAC spot-check OK: {subset}/{split}")
                    except Exception as e:
                        errors.append(f"AUDIO DECODE {subset}/{split}: {e}")
                    audio_spot_checked = True

                del table

            split_hours = split_duration_s / 3600
            subset_hours += split_hours

            for col, nc in null_counts.items():
                if nc > 0:
                    errors.append(f"NULLS {subset}/{split} col={col}: {nc} nulls")

            summary.append((subset, split, total_rows, split_hours))
            logger.info(f"  {subset}/{split}: {total_rows:,} rows, {split_hours:.1f}h")

        # Duration sanity check
        expected_h = EXPECTED_SUBSET_HOURS.get(subset, 0)
        if expected_h and abs(subset_hours - expected_h) > 20:
            errors.append(f"DURATION {subset}: {subset_hours:.1f}h (expected ~{expected_h}h)")
        logger.info(f"  {subset} total: {subset_hours:.1f}h (expected ~{expected_h}h)")

    # Summary table
    logger.info("")
    logger.info(f"{'Subset':<15} {'Split':<8} {'Rows':>10} {'Hours':>8}")
    logger.info("-" * 45)
    for subset, split, n, h in summary:
        logger.info(f"{subset:<15} {split:<8} {n:>10,} {h:>7.1f}h")

    # Disk usage
    try:
        import shutil

        usage = shutil.disk_usage(data_dir)
        logger.info(f"Data dir disk usage: {usage.used / (1024**3):.1f} GB")
    except OSError:
        pass

    if errors:
        logger.error(f"{len(errors)} check(s) FAILED:")
        for err in errors:
            logger.error(f"  [FAIL] {err}")
        return False

    logger.info("All checks passed.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify universal Parquet data integrity")
    parser.add_argument("--data-dir", type=Path, default=UNIVERSAL_DIR, help="Preprocessed data directory")
    args = parser.parse_args()

    setup_logging("verify_data")
    setup_hpc_environment()
    log_system_info()

    logger.info(f"Verifying data in: {args.data_dir}")
    ok = verify(args.data_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
