"""CoRal-style benchmark helpers compatible with Alexandra's public eval setup."""

from __future__ import annotations

import csv
import json
import os
import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from unicodedata import normalize

import jiwer
import numpy as np

from danish_asr.utils import resolve_project_path

CORAL_DATASET_ID = "CoRal-project/coral-v3"
CORAL_SUBSETS = ("read_aloud", "conversation")
TARGET_SAMPLE_RATE = 16_000
MIN_SECONDS = 0.5
MAX_SECONDS = 10.0
CORAL_CHARACTERS_TO_KEEP = "abcdefghijklmnopqrstuvwxyzæøå0123456789éü"

NUMERAL_REGEX = re.compile(r"\b(0|[1-9]\d{0,2}(?:(?:\.\d{3})*|\d*)(?:,\d+)?)\b")
FILLER_WORDS_PATTERN = re.compile(pattern=r"\b(eh+m*|øh+m*|h+m+|m+h+)\b", flags=re.IGNORECASE)

DEFAULT_CONVERSION_DICT = {
    "aa": "å",
    "ğ": "g",
    "ñ": "n",
    "ń": "n",
    "è": "e",
    "kg": " kilo ",
    "μg": " mikrogram ",
    "hhv": "henholdsvis",
    "fx": "for eksempel",
    "f.eks.": "for eksempel",
    "-": " minus ",
    "+": " plus ",
    "μ": " mikro ",
    "§": " paragraf ",
    "%": " procent ",
    "‰": " promille ",
    "ú": "u",
    "ş": "s",
    "ê": "e",
    "ã": "a",
    "ë": "e",
    "ć": "c",
    "ä": "æ",
    "í": "i",
    "š": "s",
    "î": "i",
    "ě": "e",
    "ð": "d",
    "á": "a",
    "ó": "o",
    "þ": "th",
    "ı": "i",
    "ö": "ø",
    "ç": "c",
    "ș": "s",
    "\u0301": " ",
    "\u200b": " ",
}

SUB_DIALECT_TO_DIALECT = {
    "midtøstjysk": "Østjysk",
    "østjysk": "Østjysk",
    "amagermål": "Københavnsk",
    "nørrejysk": "Nordjysk",
    "vestjysk": "Vestjysk",
    "nordsjællandsk": "Sjællandsk",
    "sjællandsk": "Sjællandsk",
    "fynsk": "Fynsk",
    "bornholmsk": "Bornholmsk",
    "sønderjysk": "Sønderjysk",
    "vendsysselsk (m. hanherred og læsø)": "Nordjysk",
    "østligt sønderjysk (m. als)": "Sønderjysk",
    "nordvestsjællandsk": "Sjællandsk",
    "thybomål": "Vestjysk",
    "himmerlandsk": "Nordjysk",
    "djurslandsk (nord-, syddjurs m. nord- og sydsamsø, anholt)": "Østjysk",
    "sydsjællandsk (sydligt sydsjællandsk)": "Sjællandsk",
    "sydfynsk": "Fynsk",
    "morsingmål": "Vestjysk",
    "sydøstjysk": "Østjysk",
    "østsjællandsk": "Sjællandsk",
    "syd for rigsgrænsen: mellemslesvisk, angelmål, fjoldemål": "Sønderjysk",
    "vestfynsk (nordvest-, sydvestfynsk)": "Fynsk",
    "vestlig sønderjysk (m. mandø og rømø)": "Sønderjysk",
    "sydvestjysk (m. fanø)": "Vestjysk",
    "sallingmål": "Vestjysk",
    "nordfalstersk": "Sydømål",
    "langelandsk": "Fynsk",
    "sydvestsjællandsk": "Sjællandsk",
    "lollandsk": "Sydømål",
    "sydømål": "Sydømål",
    "ommersysselsk": "Østjysk",
    "sydfalstersk": "Sydømål",
    "fjandbomål": "Vestjysk",
}


@dataclass(frozen=True)
class CoRalBenchmarkExample:
    """A filtered CoRal-v3 benchmark example."""

    audio: np.ndarray
    sampling_rate: int
    text: str
    subset: str
    row_index: int
    duration_s: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class FilterStats:
    """Counts collected while applying CoRal-style benchmark filters."""

    seen: int = 0
    kept: int = 0
    too_short: int = 0
    too_long: int = 0
    empty_text: int = 0
    rejected: int = 0
    max_samples_reached: bool = False


def convert_numeral_to_words(numeral: str, inside_larger_numeral: bool = False) -> str:
    """Convert Danish numerals to words using Alexandra's CoRal helper logic."""
    if re.fullmatch(pattern=NUMERAL_REGEX, string=numeral) is None:
        return numeral

    numeral = numeral.replace(".", "")

    if "," in numeral:
        if numeral.count(",") != 1:
            return numeral
        major, minor = numeral.split(",")
        major_words = convert_numeral_to_words(numeral=major)
        minor_words = " ".join(convert_numeral_to_words(numeral=char) for char in minor)
        return f"{major_words} komma {minor_words.replace('en', 'et')}"

    match len(numeral):
        case 0:
            return ""
        case 1:
            result = {
                "0": "nul",
                "1": "en",
                "2": "to",
                "3": "tre",
                "4": "fire",
                "5": "fem",
                "6": "seks",
                "7": "syv",
                "8": "otte",
                "9": "ni",
            }[numeral]
        case 2:
            mapping = {
                "10": "ti",
                "11": "elleve",
                "12": "tolv",
                "13": "tretten",
                "14": "fjorten",
                "15": "femten",
                "16": "seksten",
                "17": "sytten",
                "18": "atten",
                "19": "nitten",
                "20": "tyve",
                "30": "tredive",
                "40": "fyrre",
                "50": "halvtreds",
                "60": "tres",
                "70": "halvfjerds",
                "80": "firs",
                "90": "halvfems",
            }
            if numeral in mapping:
                return mapping[numeral]
            minor = convert_numeral_to_words(numeral=numeral[1], inside_larger_numeral=True)
            major = convert_numeral_to_words(numeral=numeral[0] + "0", inside_larger_numeral=True)
            result = f"{minor}og{major}"
        case 3:
            if not inside_larger_numeral and numeral == "100":
                return "hundrede"
            major = convert_numeral_to_words(numeral=numeral[0], inside_larger_numeral=True).replace("en", "et")
            minor = convert_numeral_to_words(numeral=numeral[1:].lstrip("0"), inside_larger_numeral=True)
            infix = "hundrede"
            if minor:
                infix += " og"
            result = f"{major} {infix} {minor}"
        case 4:
            if not inside_larger_numeral and numeral == "1000":
                return "tusind"
            major = convert_numeral_to_words(numeral=numeral[0], inside_larger_numeral=True).replace("en", "et")
            minor = convert_numeral_to_words(numeral=numeral[1:].lstrip("0"), inside_larger_numeral=True)
            infix = "tusind"
            if minor and len(str(int(numeral[1:]))) <= 2:
                infix += " og"
            result = f"{major} {infix} {minor}".strip()
        case 5:
            major = convert_numeral_to_words(numeral=numeral[:2], inside_larger_numeral=True)
            minor = convert_numeral_to_words(numeral=numeral[2:].lstrip("0"), inside_larger_numeral=True)
            infix = "tusind"
            if minor and len(str(int(numeral[2:]))) <= 2:
                infix += " og"
            result = f"{major} {infix} {minor}"
        case 6:
            major = convert_numeral_to_words(numeral=numeral[:3], inside_larger_numeral=True)
            minor = convert_numeral_to_words(numeral=numeral[3:].lstrip("0"), inside_larger_numeral=True)
            infix = "tusind"
            if minor and len(str(int(numeral[3:]))) <= 2:
                infix += " og"
            result = f"{major} {infix} {minor}"
        case 7:
            major = convert_numeral_to_words(numeral=numeral[0], inside_larger_numeral=True)
            minor = convert_numeral_to_words(numeral=numeral[1:].lstrip("0"), inside_larger_numeral=True)
            infix = "million" if int(numeral[0]) == 1 else "millioner"
            if minor and len(str(int(numeral[1:]))) <= 2:
                infix += " og"
            result = f"{major} {infix} {minor}"
        case 8:
            major = convert_numeral_to_words(numeral=numeral[:2], inside_larger_numeral=True)
            minor = convert_numeral_to_words(numeral=numeral[2:].lstrip("0"), inside_larger_numeral=True)
            infix = "millioner"
            if minor and len(str(int(numeral[2:]))) <= 2:
                infix += " og"
            result = f"{major} {infix} {minor}"
        case 9:
            major = convert_numeral_to_words(numeral=numeral[:3], inside_larger_numeral=True)
            minor = convert_numeral_to_words(numeral=numeral[3:].lstrip("0"), inside_larger_numeral=True)
            infix = "millioner"
            if minor and len(str(int(numeral[3:]))) <= 2:
                infix += " og"
            result = f"{major} {infix} {minor}"
        case _:
            return numeral

    return re.sub(r" +", " ", result).strip()


def normalize_coral_benchmark_text(
    text: str,
    *,
    convert_numerals: bool = True,
    characters_to_keep: Iterable[str] | None = CORAL_CHARACTERS_TO_KEEP,
) -> str:
    """Normalize text following Alexandra's public CoRal evaluation conventions."""
    doc = text
    if convert_numerals and re.search(pattern=NUMERAL_REGEX, string=doc):
        doc = "".join(
            convert_numeral_to_words(numeral=maybe_numeral)
            for maybe_numeral in re.split(pattern=NUMERAL_REGEX, string=doc)
            if maybe_numeral is not None
        )

    doc = doc.lower()
    doc = FILLER_WORDS_PATTERN.sub(repl="", string=doc)
    doc = normalize("NFKC", doc)

    for key, value in DEFAULT_CONVERSION_DICT.items():
        doc = doc.replace(key, value)

    if characters_to_keep is not None:
        allowed = "".join(char for char in characters_to_keep)
        non_standard_characters_regex = re.compile(f"[^{re.escape(allowed + ' |')}]", flags=re.IGNORECASE)
        doc = re.sub(non_standard_characters_regex, " ", doc.strip())

    doc = re.sub(r" +", " ", doc)
    return "\n".join(line.strip() for line in doc.split("\n")).strip("\n")


def bounded_error_rate(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    unit: str,
    normalise: bool = True,
) -> float:
    """Compute Alexandra-style bounded edit error rate as a fraction."""
    if len(predictions) != len(references):
        msg = "Predictions and references must have the same number of items."
        raise ValueError(msg)

    incorrect = 0
    total = 0
    process = jiwer.process_characters if unit == "char" else jiwer.process_words

    for prediction, reference in zip(predictions, references, strict=True):
        measures = process(reference=reference, hypothesis=prediction)
        incorrect += measures.substitutions + measures.deletions + measures.insertions
        total += measures.substitutions + measures.deletions + measures.hits
        if normalise:
            total += measures.insertions

    return 0.0 if total == 0 else incorrect / total


def score_coral_style(predictions: Sequence[str], references: Sequence[str]) -> dict[str, float | int]:
    """Return CoRal-style and plain jiwer metrics as percentages."""
    if len(predictions) != len(references):
        msg = "Predictions and references must have the same number of items."
        raise ValueError(msg)

    return {
        "num_examples": len(predictions),
        "cer_coral": bounded_error_rate(predictions, references, unit="char") * 100.0,
        "wer_coral": bounded_error_rate(predictions, references, unit="word") * 100.0,
        "cer_jiwer": jiwer.cer(list(references), list(predictions)) * 100.0,
        "wer_jiwer": jiwer.wer(list(references), list(predictions)) * 100.0,
    }


def _example_duration_seconds(sample: dict[str, Any]) -> float:
    audio = sample["audio"]
    return len(audio["array"]) / float(audio["sampling_rate"])


def _empty_filter_stats(max_samples_reached: bool = False) -> FilterStats:
    return FilterStats(max_samples_reached=max_samples_reached)


def _replace_filter_stats(stats: FilterStats, **updates: int | bool) -> FilterStats:
    data = asdict(stats)
    data.update(updates)
    return FilterStats(**data)


def _iter_dataset_rows(dataset: Iterable[dict[str, Any]]) -> Iterator[tuple[int, dict[str, Any]]]:
    yield from enumerate(dataset)


def load_coral_v3_test_subset(
    subset: str,
    *,
    max_samples: int | None = None,
    cache_dir: str | None = None,
    dataset_loader: Callable[..., Iterable[dict[str, Any]]] | None = None,
    min_seconds: float = MIN_SECONDS,
    max_seconds: float = MAX_SECONDS,
) -> tuple[list[CoRalBenchmarkExample], FilterStats]:
    """Load and filter a CoRal-v3 test subset for direct benchmark evaluation."""
    if subset not in CORAL_SUBSETS:
        msg = f"Unsupported CoRal subset {subset!r}. Expected one of: {', '.join(CORAL_SUBSETS)}"
        raise ValueError(msg)

    if dataset_loader is None:
        from datasets import load_dataset

        dataset_loader = load_dataset

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or True
    load_kwargs: dict[str, Any] = {
        "path": CORAL_DATASET_ID,
        "name": subset,
        "split": "test",
        "streaming": True,
        "token": token,
    }
    if cache_dir is not None:
        load_kwargs["cache_dir"] = cache_dir

    dataset = dataset_loader(**load_kwargs)
    examples: list[CoRalBenchmarkExample] = []
    stats = _empty_filter_stats()

    for row_index, sample in _iter_dataset_rows(dataset):
        stats = _replace_filter_stats(stats, seen=stats.seen + 1)

        duration_s = _example_duration_seconds(sample)
        if duration_s <= min_seconds:
            stats = _replace_filter_stats(stats, too_short=stats.too_short + 1)
            continue
        if duration_s >= max_seconds:
            stats = _replace_filter_stats(stats, too_long=stats.too_long + 1)
            continue

        text = str(sample.get("text", ""))
        if len(text.strip()) == 0:
            stats = _replace_filter_stats(stats, empty_text=stats.empty_text + 1)
            continue
        if sample.get("validated") == "rejected":
            stats = _replace_filter_stats(stats, rejected=stats.rejected + 1)
            continue

        audio = sample["audio"]
        metadata = {key: value for key, value in sample.items() if key not in {"audio", "text"}}
        examples.append(
            CoRalBenchmarkExample(
                audio=np.asarray(audio["array"], dtype=np.float32),
                sampling_rate=int(audio["sampling_rate"]),
                text=text,
                subset=subset,
                row_index=row_index,
                duration_s=duration_s,
                metadata=metadata,
            )
        )
        stats = _replace_filter_stats(stats, kept=stats.kept + 1)

        if max_samples is not None and len(examples) >= max_samples:
            stats = _replace_filter_stats(stats, max_samples_reached=True)
            break

    return examples, stats


def normalized_prediction_reference_pairs(
    predictions: Sequence[str], examples: Sequence[CoRalBenchmarkExample]
) -> tuple[list[str], list[str]]:
    """Normalize model predictions and CoRal references with the same benchmark transform."""
    references = [normalize_coral_benchmark_text(example.text) for example in examples]
    normalized_predictions = [normalize_coral_benchmark_text(prediction) for prediction in predictions]
    return normalized_predictions, references


def _age_group(age: Any) -> str | None:
    if age is None:
        return None
    try:
        age_int = int(age)
    except (TypeError, ValueError):
        return None
    if age_int < 25:
        return "0-25"
    if age_int < 50:
        return "25-50"
    return "50+"


def _dialect(metadata: dict[str, Any]) -> str | None:
    country_birth = metadata.get("country_birth")
    if country_birth is not None and country_birth != "DK":
        return "Non-native"

    raw_dialect = metadata.get("dialect")
    if raw_dialect is None:
        return None
    dialect_key = str(raw_dialect).lower()
    return SUB_DIALECT_TO_DIALECT.get(dialect_key, str(raw_dialect))


def example_group_metadata(example: CoRalBenchmarkExample) -> dict[str, str | None]:
    """Return benchmark grouping metadata matching Alexandra's age/gender/dialect axes."""
    gender = example.metadata.get("gender")
    return {
        "age_group": _age_group(example.metadata.get("age")),
        "gender": None if gender is None else str(gender),
        "dialect": _dialect(example.metadata),
    }


def score_by_group(
    predictions: Sequence[str],
    references: Sequence[str],
    examples: Sequence[CoRalBenchmarkExample],
    *,
    categories: Sequence[str] = ("age_group", "gender", "dialect"),
) -> list[dict[str, Any]]:
    """Score overall and grouped category combinations using CoRal-style metrics."""
    if not (len(predictions) == len(references) == len(examples)):
        msg = "Predictions, references, and examples must have the same length."
        raise ValueError(msg)

    group_rows = [example_group_metadata(example) for example in examples]
    values_by_category = [
        sorted({row[category] for row in group_rows if row[category] is not None}) + [None] for category in categories
    ]

    def product(items: Sequence[Sequence[str | None]]) -> Iterator[tuple[str | None, ...]]:
        if not items:
            yield ()
            return
        for value in items[0]:
            for rest in product(items[1:]):
                yield (value, *rest)

    records: list[dict[str, Any]] = []
    all_indices = list(range(len(examples)))
    for combination in product(values_by_category):
        selected = all_indices
        skip = False
        for category, value in zip(categories, combination, strict=True):
            if value is None:
                continue
            filtered = [idx for idx in selected if group_rows[idx][category] == value]
            if len(filtered) == len(selected) or len(filtered) == 0:
                skip = True
                break
            selected = filtered
        if skip:
            continue

        selected_predictions = [predictions[idx] for idx in selected]
        selected_references = [references[idx] for idx in selected]
        scores = score_coral_style(selected_predictions, selected_references)
        records.append(dict(zip(categories, combination, strict=True)) | scores | {"num_examples": len(selected)})

    return records


def write_benchmark_outputs(
    output_dir: str | Path,
    *,
    predictions: Sequence[str],
    references: Sequence[str],
    raw_predictions: Sequence[str],
    examples: Sequence[CoRalBenchmarkExample],
    scores: dict[str, Any],
    by_group: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    """Write benchmark artifacts in a stable, inspectable format."""
    output_path = resolve_project_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    (output_path / "predictions.txt").write_text("".join(f"{line}\n" for line in predictions), encoding="utf-8")
    (output_path / "references.txt").write_text("".join(f"{line}\n" for line in references), encoding="utf-8")

    with (output_path / "records.jsonl").open("w", encoding="utf-8") as handle:
        for raw_prediction, prediction, reference, example in zip(
            raw_predictions, predictions, references, examples, strict=True
        ):
            record = {
                "row_index": example.row_index,
                "subset": example.subset,
                "duration_s": example.duration_s,
                "raw_reference": example.text,
                "raw_prediction": raw_prediction,
                "reference": reference,
                "prediction": prediction,
                "metadata": example.metadata,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    (output_path / "scores.json").write_text(
        json.dumps({"scores": scores, "metadata": metadata}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    fieldnames = ["age_group", "gender", "dialect", "num_examples", "cer_coral", "wer_coral", "cer_jiwer", "wer_jiwer"]
    with (output_path / "by_group.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(by_group)
