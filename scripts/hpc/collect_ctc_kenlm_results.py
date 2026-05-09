"""Collect CTC + beam + KenLM eval outputs into CSV and Markdown summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from scripts.hpc.check_ctc_kenlm_eval_ready import DEFAULT_MANIFEST, load_manifest

FIELDS = [
    "methodology",
    "model",
    "split_or_subset",
    "decoder",
    "success",
    "num_examples",
    "wer",
    "cer",
    "wer_coral",
    "cer_coral",
    "wer_jiwer",
    "cer_jiwer",
    "beam_width",
    "alpha",
    "beta",
    "output_path",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _scores_and_metadata(scores_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _read_json(scores_path)
    if "scores" in payload:
        return dict(payload["scores"]), dict(payload.get("metadata", {}))

    metadata_path = scores_path.with_name("metadata.json")
    metadata = _read_json(metadata_path) if metadata_path.exists() else {}
    return dict(payload), metadata


def _collect_root(root: Path, methodology: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows

    for scores_path in sorted(root.glob("*/*/*/scores.json")):
        model, split_or_subset, decoder = scores_path.parent.relative_to(root).parts[:3]
        scores, metadata = _scores_and_metadata(scores_path)
        rows.append(
            {
                "methodology": methodology,
                "model": model,
                "split_or_subset": split_or_subset,
                "decoder": decoder,
                "success": (scores_path.parent / "SUCCESS").exists(),
                "num_examples": scores.get("num_examples"),
                "wer": scores.get("wer"),
                "cer": scores.get("cer"),
                "wer_coral": scores.get("wer_coral"),
                "cer_coral": scores.get("cer_coral"),
                "wer_jiwer": scores.get("wer_jiwer"),
                "cer_jiwer": scores.get("cer_jiwer"),
                "beam_width": metadata.get("beam_width"),
                "alpha": metadata.get("alpha"),
                "beta": metadata.get("beta"),
                "output_path": str(scores_path.parent),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# CTC + Beam + KenLM Results", ""]
    if not rows:
        lines.append("No result files found.")
    else:
        lines.append("| " + " | ".join(FIELDS) + " |")
        lines.append("| " + " | ".join(["---"] * len(FIELDS)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(_format_value(row.get(field)) for field in FIELDS) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    output_roots = manifest["output_roots"]
    result_root = Path(args.output_dir or output_roots["results"])

    rows = [
        *_collect_root(Path(output_roots["my_method"]), "my_method"),
        *_collect_root(Path(output_roots["coral_method"]), "coral_method"),
    ]
    rows.sort(key=lambda row: (row["methodology"], row["model"], row["split_or_subset"], row["decoder"]))

    csv_path = result_root / "results.csv"
    markdown_path = result_root / "results.md"
    _write_csv(csv_path, rows)
    _write_markdown(markdown_path, rows)
    print(f"Wrote {len(rows)} rows to {csv_path}")
    print(f"Wrote Markdown summary to {markdown_path}")


if __name__ == "__main__":
    main()
