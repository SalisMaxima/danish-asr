"""Build a deterministic Danish KenLM text corpus from local fairseq2 parquet shards."""

from __future__ import annotations

import argparse

from loguru import logger

from danish_asr.lm import (
    build_lm_corpus_from_parquet,
    load_yaml_config,
    write_corpus_stats,
    write_lm_corpus,
)
from danish_asr.utils import configure_project_cache_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lm/coral_train_only_v1.yaml")
    return parser.parse_args()


def main() -> None:
    configure_project_cache_environment()
    args = parse_args()
    config = load_yaml_config(args.config)

    source = config["source"]
    output = config["output"]

    texts, stats = build_lm_corpus_from_parquet(
        source["dataset_root"],
        split=source.get("split", "train"),
        language=source.get("language", "dan_Latn"),
        corpora=tuple(source["corpora"]),
    )

    write_lm_corpus(texts, output["corpus_text_path"])
    write_corpus_stats(stats, output["stats_path"])

    logger.info(
        "Built LM corpus with {} unique lines / {} tokens from {} raw examples.",
        stats.unique_examples,
        stats.token_count,
        stats.raw_examples,
    )


if __name__ == "__main__":
    main()
