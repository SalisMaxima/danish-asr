"""Build a cleaned pyctcdecode unigram list for OmniASR CTC + KenLM decoding."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from danish_asr.lm import build_pyctcdecode_unigrams, load_yaml_config, write_text_lines
from danish_asr.utils import configure_project_cache_environment, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lm/alexandra_proxy_v1.yaml")
    parser.add_argument("--text-path", default=None)
    parser.add_argument("--tokenizer-model-path", required=True)
    parser.add_argument("--output-path", default=None)
    return parser.parse_args()


def main() -> None:
    configure_project_cache_environment()
    args = parse_args()
    config = load_yaml_config(args.config)

    text_path = resolve_project_path(args.text_path or config["output"]["corpus_text_path"])
    tokenizer_model_path = resolve_project_path(args.tokenizer_model_path)
    output_path = resolve_project_path(args.output_path or f"artifacts/lm/{config['name']}_pyctcdecode_unigrams.txt")

    if not text_path.is_file():
        msg = f"LM corpus text file not found: {text_path}. Run scripts/lm/build_danish_lm_corpus.py first."
        raise FileNotFoundError(msg)

    logger.info("Reading LM corpus text from {}", text_path)
    with text_path.open(encoding="utf-8") as handle:
        unigrams = build_pyctcdecode_unigrams(handle, tokenizer_model_path=Path(tokenizer_model_path))

    write_text_lines(output_path, unigrams)
    logger.info("Wrote {} cleaned unigrams to {}", len(unigrams), output_path)


if __name__ == "__main__":
    main()
