"""Prepare a Fairseq2 eval config that opens a one-corpus parquet root.

Fairseq2's ASR eval recipe opens the parquet root derived from
``dataset_summary_path.parent``. A corpus-filtered TSV inside the full root is
not sufficient for split-tagged evals because the loader can still discover all
``corpus=*`` directories. This helper creates a small subset root with symlinks
to only the requested corpus directory and writes an eval config that points at
that root's summary TSV.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import yaml


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


def _safe_symlink(source: Path, target: Path) -> None:
    if target.is_symlink():
        if target.resolve() == source.resolve():
            return
        target.unlink()

    if target.exists():
        msg = f"Refusing to replace non-symlink path: {target}"
        raise FileExistsError(msg)

    try:
        target.symlink_to(source, target_is_directory=True)
    except OSError as exc:
        msg = (
            f"Failed to create subset corpus symlink {target} -> {source}. "
            "This helper is intended for Linux HPC filesystems with symlink support."
        )
        raise OSError(msg) from exc


def _write_filtered_summary(source_summary: Path, output_summary: Path, corpus: str) -> None:
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    with (
        source_summary.open("r", encoding="utf-8", newline="") as source,
        output_summary.open("w", encoding="utf-8", newline="") as output,
    ):
        reader = csv.reader(source, delimiter="\t")
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        header = next(reader)
        writer.writerow(header)
        kept = 0
        for row in reader:
            if row and row[0] == corpus:
                writer.writerow(row)
                kept += 1

    if kept == 0:
        msg = f"No rows for corpus={corpus!r} in {source_summary}"
        raise ValueError(msg)


def _source_summary_for_subset(configured_summary: Path) -> Path:
    """Return a full summary TSV to filter for a subset config.

    Existing subset configs may point at generated files such as
    ``language_distribution_read_aloud.tsv``. Those files are intentionally no
    longer generated in-place, so fall back to the canonical full summary next
    to them when needed.
    """
    if configured_summary.exists():
        return configured_summary

    fallback = configured_summary.parent / "language_distribution_0.tsv"
    if fallback.exists():
        return fallback

    return configured_summary


def prepare_config(
    *,
    source_config: Path,
    output_config: Path,
    subset_root_parent: Path,
    subset_corpus: str | None,
) -> Path:
    config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    storage_config = config["dataset"]["mixture_parquet_storage_config"]
    configured_summary = _resolve_project_path(storage_config["dataset_summary_path"])

    if subset_corpus is None:
        summary_path = configured_summary
    else:
        source_summary = _source_summary_for_subset(configured_summary)
        source_root = configured_summary.parent
        source_corpus_dir = source_root / f"corpus={subset_corpus}"
        if not source_corpus_dir.is_dir():
            msg = f"Corpus directory not found: {source_corpus_dir}"
            raise FileNotFoundError(msg)

        subset_root = subset_root_parent / subset_corpus / source_root.name
        subset_summary = subset_root / configured_summary.name
        _write_filtered_summary(source_summary, subset_summary, subset_corpus)
        _safe_symlink(source_corpus_dir, subset_root / f"corpus={subset_corpus}")
        summary_path = subset_summary

    storage_config["dataset_summary_path"] = str(summary_path)
    output_config.parent.mkdir(parents=True, exist_ok=True)
    tmp_config = output_config.with_suffix(output_config.suffix + ".tmp")
    try:
        tmp_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        tmp_config.replace(output_config)
    except BaseException:
        if tmp_config.exists():
            tmp_config.unlink()
        raise

    return output_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--subset-root-parent", type=Path, required=True)
    parser.add_argument("--subset-corpus", default=None)
    parser.add_argument("--copy-only", action="store_true")
    args = parser.parse_args()

    if args.copy_only and args.subset_corpus:
        msg = "--copy-only cannot be combined with --subset-corpus"
        raise ValueError(msg)

    if args.copy_only:
        args.output_config.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.config, args.output_config)
        print(args.output_config)
        return

    output_config = prepare_config(
        source_config=args.config,
        output_config=args.output_config,
        subset_root_parent=args.subset_root_parent,
        subset_corpus=args.subset_corpus,
    )
    print(output_config)


if __name__ == "__main__":
    main()
