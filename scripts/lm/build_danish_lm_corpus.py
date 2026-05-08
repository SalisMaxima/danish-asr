"""Build a deterministic Danish KenLM text corpus."""

from __future__ import annotations

import argparse

from loguru import logger

from danish_asr.lm import (
    build_hf_text_lm_corpus,
    build_lm_corpus_from_parquet,
    load_yaml_config,
    write_corpus_stats,
    write_lm_corpus,
)
from danish_asr.utils import configure_project_cache_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lm/alexandra_proxy_v1.yaml")
    return parser.parse_args()


def main() -> None:
    configure_project_cache_environment()
    args = parse_args()
    config = load_yaml_config(args.config)

    source = config["source"]
    output = config["output"]

    source_kind = source.get("kind", "parquet")
    if source_kind == "parquet":
        texts, stats = build_lm_corpus_from_parquet(
            source["dataset_root"],
            split=source.get("split", "train"),
            language=source.get("language", "dan_Latn"),
            corpora=tuple(source["corpora"]),
        )

        write_lm_corpus(texts, output["corpus_text_path"])
        write_corpus_stats(stats, output["stats_path"])
    elif source_kind == "hf_text":
        if "exclude_datasets" not in source:
            msg = (
                "hf_text source must set `exclude_datasets` explicitly (use `[]` to opt out). "
                "A typo or omitted key would silently train the LM on held-out eval transcripts."
            )
            raise ValueError(msg)
        stats = build_hf_text_lm_corpus(
            datasets_config=source["datasets"],
            output_path=output["corpus_text_path"],
            stats_path=output["stats_path"],
            version=config["name"],
            cache_dir=source.get("cache_dir"),
            streaming=source.get("streaming", True),
            exclude_datasets_config=source["exclude_datasets"],
        )
    else:
        msg = f"Unsupported LM corpus source kind: {source_kind}"
        raise ValueError(msg)

    logger.info(
        "Built LM corpus with {} unique lines / {} tokens from {} raw examples.",
        stats.unique_examples,
        stats.token_count,
        stats.raw_examples,
    )


if __name__ == "__main__":
    main()
