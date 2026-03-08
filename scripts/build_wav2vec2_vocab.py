"""Build a Danish CTC vocabulary for the Wav2Vec2 baseline from CoRal transcripts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable

from loguru import logger

from danish_asr.text import normalize_ctc_text
from danish_asr.utils import configure_project_cache_environment, get_project_hf_cache_dir, resolve_project_path

DEFAULT_SUBSETS = ("read_aloud", "conversation")
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
WORD_DELIMITER_TOKEN = "|"


def build_ctc_vocab(
    texts: Iterable[str],
    pad_token: str = PAD_TOKEN,
    unk_token: str = UNK_TOKEN,
    word_delimiter_token: str = WORD_DELIMITER_TOKEN,
) -> dict[str, int]:
    """Build a deterministic CTC vocabulary from transcript texts."""
    charset: set[str] = set()
    for text in texts:
        normalized = normalize_ctc_text(text)
        if not normalized:
            continue
        charset.update(character for character in normalized if character != " ")

    vocab = {
        pad_token: 0,
        unk_token: 1,
        word_delimiter_token: 2,
    }
    for character in sorted(charset):
        if character not in vocab:
            vocab[character] = len(vocab)
    return vocab


def iter_training_texts(
    dataset_name: str,
    subsets: tuple[str, ...],
    split: str,
    cache_dir: str | None,
    max_examples_per_subset: int | None = None,
) -> Iterable[str]:
    """Yield transcript texts from the selected CoRal subsets."""
    from datasets import Audio, load_dataset

    for subset in subsets:
        logger.info(f"Loading {dataset_name}/{subset} split={split} for vocab generation")
        dataset = load_dataset(dataset_name, subset, split=split, cache_dir=cache_dir, streaming=True)
        if "audio" in dataset.column_names:
            dataset = dataset.cast_column("audio", Audio(decode=False))
        if "audio" in dataset.column_names:
            dataset = dataset.remove_columns("audio")
        for idx, item in enumerate(dataset):
            if max_examples_per_subset is not None and idx >= max_examples_per_subset:
                break
            yield item.get("text", item.get("sentence", ""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="CoRal-project/coral-v3")
    parser.add_argument("--subsets", default=",".join(DEFAULT_SUBSETS))
    parser.add_argument("--split", default="train")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-path", default="configs/vocab/coral_wav2vec2_ctc.json")
    parser.add_argument("--max-examples-per-subset", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    configure_project_cache_environment()
    args = parse_args()
    subsets = tuple(part.strip() for part in args.subsets.split(",") if part.strip())
    cache_dir = args.cache_dir or str(get_project_hf_cache_dir())
    texts = iter_training_texts(
        args.dataset_name,
        subsets,
        args.split,
        cache_dir,
        max_examples_per_subset=args.max_examples_per_subset,
    )
    vocab = build_ctc_vocab(texts)

    output_path = resolve_project_path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    logger.info(f"Saved Wav2Vec2 CTC vocab to {output_path} with {len(vocab)} tokens")
    print(f"num_labels={len(vocab)}")


if __name__ == "__main__":
    main()
