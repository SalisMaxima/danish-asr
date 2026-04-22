"""Compute WER from saved prediction and reference text files."""

from __future__ import annotations

import argparse
import json

from danish_asr.lm import read_text_lines, score_predictions
from danish_asr.utils import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = read_text_lines(args.predictions)
    references = read_text_lines(args.references)
    summary = score_predictions(predictions, references)

    if args.output_json:
        output_path = resolve_project_path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
