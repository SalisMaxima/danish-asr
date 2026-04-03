"""Filter language_distribution_0.tsv to a single CoRal-v3 subset.

The fairseq2 MIXTURE_PARQUET loader uses the TSV to know which parquet
fragments exist and how to sample from them. Per-subset evaluation requires
a TSV that lists only the rows for that subset so the loader doesn't mix
in the other corpus.

Usage (run from project root on HPC):
    python scripts/hpc/make_subset_tsv.py --subset read_aloud
    python scripts/hpc/make_subset_tsv.py --subset conversation
    python scripts/hpc/make_subset_tsv.py --subset read_aloud \\
        --input  data/parquet/version=0/language_distribution_0.tsv \\
        --output data/parquet/version=0/language_distribution_read_aloud.tsv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Matches SUBSETS in src/danish_asr/preprocessing.py
_CORPUS_NAME = {
    "read_aloud": "coral_v3_read_aloud",
    "conversation": "coral_v3_conversation",
}

_DEFAULT_INPUT = Path("data/parquet/version=0/language_distribution_0.tsv")
_DEFAULT_OUTPUT_TEMPLATE = "data/parquet/version=0/language_distribution_{subset}.tsv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter language_distribution TSV to a single corpus subset")
    parser.add_argument(
        "--subset",
        choices=list(_CORPUS_NAME),
        required=True,
        help="Which subset to keep: 'read_aloud' or 'conversation'",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_DEFAULT_INPUT,
        help=f"Source TSV (default: {_DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output TSV path (default: language_distribution_<subset>.tsv next to input)",
    )
    args = parser.parse_args()

    corpus = _CORPUS_NAME[args.subset]

    if not args.input.exists():
        print(f"ERROR: input TSV not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    output = args.output or args.input.parent / f"language_distribution_{args.subset}.tsv"

    input_path = args.input.resolve()
    output_path = output.resolve(strict=False)
    if input_path == output_path:
        print(
            f"ERROR: input and output paths must differ: {args.input} == {output}",
            file=sys.stderr,
        )
        sys.exit(1)
    kept = 0
    with args.input.open(newline="") as fh_in, output.open("w", newline="") as fh_out:
        reader = csv.DictReader(fh_in, delimiter="\t")
        fieldnames = reader.fieldnames
        if fieldnames is None:
            print(f"ERROR: input TSV is empty or missing a header row: {args.input}", file=sys.stderr)
            output.unlink(missing_ok=True)
            sys.exit(1)
        if "corpus" not in fieldnames:
            print(f"ERROR: input TSV is missing required 'corpus' column: {args.input}", file=sys.stderr)
            output.unlink(missing_ok=True)
            sys.exit(1)
        writer = csv.DictWriter(fh_out, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in reader:
            if row["corpus"] == corpus:
                writer.writerow(row)
                kept += 1

    if kept == 0:
        print(f"ERROR: no rows matched corpus='{corpus}' in {args.input}", file=sys.stderr)
        output.unlink(missing_ok=True)
        sys.exit(1)

    print(f"Wrote {kept} row(s) for corpus='{corpus}' → {output}")


if __name__ == "__main__":
    main()
