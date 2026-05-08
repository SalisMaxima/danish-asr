"""Helpers for Danish LM corpus building and CTC + KenLM decoding."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from danish_asr.utils import configure_project_cache_environment, get_project_fairseq2_cache_dir, resolve_project_path

try:
    import pyarrow.parquet as pq

    _PYARROW_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency in tests/CI
    pq = None  # type: ignore[assignment]
    _PYARROW_AVAILABLE = False

try:
    from fairseq2.assets import get_asset_store
    from fairseq2.data.tokenizers.char import load_char_tokenizer
    from fairseq2.data.tokenizers.hub import load_tokenizer
    from fairseq2.data.tokenizers.sentencepiece import load_sentencepiece_model
    from fairseq2.models.wav2vec2.asr import get_wav2vec2_asr_model_hub

    _FAIRSEQ2_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency in tests/CI
    get_asset_store = None  # type: ignore[assignment]
    load_char_tokenizer = None  # type: ignore[assignment]
    load_tokenizer = None  # type: ignore[assignment]
    load_sentencepiece_model = None  # type: ignore[assignment]
    get_wav2vec2_asr_model_hub = None  # type: ignore[assignment]
    _FAIRSEQ2_AVAILABLE = False

try:
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    _OMNIASR_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency in tests/CI
    ASRInferencePipeline = None  # type: ignore[assignment]
    _OMNIASR_AVAILABLE = False

if TYPE_CHECKING:
    from fairseq2.data.tokenizers import Tokenizer
    from fairseq2.models.wav2vec2.asr import Wav2Vec2AsrModel
else:
    Tokenizer = Any
    Wav2Vec2AsrModel = Any
from loguru import logger

LM_VERSION = "danish_lm_v1"
DEFAULT_CORPORA = ("coral_v3_read_aloud", "coral_v3_conversation")
DEFAULT_LANGUAGE = "dan_Latn"

_WHITESPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_METADATA_TOKEN_RE = re.compile(
    r"\b(?:speaker_id|speaker|spk|id|uid|utt|segment|timestamp)\b(?:\s*[:=_-]\s*[\w-]+)?",
    re.IGNORECASE,
)
_ONLY_PUNCT_RE = re.compile(r"^[\W_]+$")


@dataclass
class CorpusStats:
    """Compact statistics for an LM text build."""

    version: str
    split: str
    language: str
    corpora: list[str]
    raw_examples: int
    written_examples: int
    unique_examples: int
    token_count: int
    source_counts: dict[str, int]
    normalization: dict[str, Any]


@dataclass
class DecodeResult:
    """Decoded hypothesis with reference metadata."""

    prediction: str
    reference: str
    corpus: str
    file: str
    row_index: int


def _require_pyarrow() -> None:
    if not _PYARROW_AVAILABLE:
        msg = "pyarrow is required for parquet-backed LM corpus and decoding scripts. Install the 'omni' dependency group."
        raise ImportError(msg)


def _require_fairseq2() -> None:
    if not _FAIRSEQ2_AVAILABLE:
        msg = (
            "fairseq2 is required for tokenizer/model-backed LM helpers. "
            "Install the 'omni' dependency group with `uv sync --group omni`."
        )
        raise ImportError(msg)


def _require_omniasr() -> None:
    if not _OMNIASR_AVAILABLE:
        msg = "omnilingual_asr is required for OmniASR inference pipeline helpers. Install the 'omni' dependency group."
        raise ImportError(msg)


def normalize_lm_text(text: str) -> str:
    """Normalize transcript text for LM training without changing Danish orthography."""
    text = unicodedata.normalize("NFKC", text)
    text = _URL_RE.sub(" ", text)
    text = _METADATA_TOKEN_RE.sub(" ", text)
    text = text.lower()
    text = _WHITESPACE_RE.sub(" ", text).strip()

    if not text or _ONLY_PUNCT_RE.fullmatch(text):
        return ""

    return text


def parse_valid_split(valid_split: str) -> tuple[str, str | None]:
    """Parse fairseq2 split names such as ``test`` or ``test_coral_v3_read_aloud``."""
    if valid_split in {"train", "dev", "test"}:
        return valid_split, None

    split, _, corpus = valid_split.partition("_")
    if split not in {"train", "dev", "test"} or not corpus:
        msg = f"Unsupported valid_split value: {valid_split}"
        raise ValueError(msg)

    return split, corpus


def iter_fairseq2_rows(
    dataset_root: Path,
    *,
    split: str,
    language: str = DEFAULT_LANGUAGE,
    corpora: Sequence[str] | None = None,
    columns: Sequence[str] = ("text",),
) -> Iterator[dict[str, Any]]:
    """Yield rows from fairseq2 parquet shards for the requested corpora/split."""
    _require_pyarrow()

    root = resolve_project_path(dataset_root)
    selected_corpora = (
        list(corpora)
        if corpora is not None
        else sorted(path.name.removeprefix("corpus=") for path in root.glob("corpus=*") if path.is_dir())
    )

    for corpus in selected_corpora:
        corpus_dir = root / f"corpus={corpus}" / f"split={split}" / f"language={language}"
        if not corpus_dir.exists():
            logger.warning(f"Skipping missing corpus split directory: {corpus_dir}")
            continue

        for parquet_file in sorted(corpus_dir.glob("*.parquet")):
            parquet = pq.ParquetFile(parquet_file)
            row_offset = 0
            for batch in parquet.iter_batches(columns=list(columns)):
                data = batch.to_pydict()
                batch_size = len(next(iter(data.values()))) if data else 0
                for idx in range(batch_size):
                    row = {column: data[column][idx] for column in columns}
                    row["corpus"] = corpus
                    row["file"] = str(parquet_file)
                    row["row_index"] = row_offset + idx
                    yield row
                row_offset += batch_size


def build_lm_corpus_from_parquet(
    dataset_root: Path,
    *,
    split: str = "train",
    language: str = DEFAULT_LANGUAGE,
    corpora: Sequence[str] = DEFAULT_CORPORA,
) -> tuple[list[str], CorpusStats]:
    """Build a deterministic LM text corpus from fairseq2 parquet transcripts."""
    seen: set[str] = set()
    lines: list[str] = []
    source_counts = dict.fromkeys(corpora, 0)
    raw_examples = 0
    written_examples = 0
    token_count = 0

    for row in iter_fairseq2_rows(dataset_root, split=split, language=language, corpora=corpora, columns=("text",)):
        raw_examples += 1
        normalized = normalize_lm_text(str(row["text"]))
        if not normalized:
            continue

        if normalized in seen:
            continue

        seen.add(normalized)
        lines.append(normalized)
        written_examples += 1
        token_count += len(normalized.split())
        source_counts[row["corpus"]] = source_counts.get(row["corpus"], 0) + 1

    stats = CorpusStats(
        version=LM_VERSION,
        split=split,
        language=language,
        corpora=list(corpora),
        raw_examples=raw_examples,
        written_examples=written_examples,
        unique_examples=len(lines),
        token_count=token_count,
        source_counts=source_counts,
        normalization={
            "unicode_normalization": "NFKC",
            "lowercase": True,
            "collapse_whitespace": True,
            "strip_urls": True,
            "strip_metadata_tokens": True,
            "deduplicate_exact_lines": True,
        },
    )
    return lines, stats


def write_lm_corpus(texts: Iterable[str], output_path: Path) -> None:
    """Write normalized LM text, one line per example."""
    output_path = resolve_project_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(f"{line}\n" for line in texts), encoding="utf-8")


def write_corpus_stats(stats: CorpusStats, output_path: Path) -> None:
    """Write corpus stats as pretty JSON."""
    output_path = resolve_project_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(stats), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a small YAML config file."""
    import yaml

    config_path = resolve_project_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _get_cached_tokenizer_path(tokenizer_name: str) -> Path | None:
    """Find a cached tokenizer model file using the asset card basename."""
    _require_fairseq2()

    store = get_asset_store()
    card = store.retrieve_card(tokenizer_name)
    tokenizer_uri = card.field("tokenizer").as_uri()
    model_name = Path(tokenizer_uri.path).name

    candidate_roots = [
        get_project_fairseq2_cache_dir() / "assets",
        Path.home() / ".cache" / "fairseq2" / "assets",
    ]

    for root in candidate_roots:
        if not root.exists():
            continue

        matches = sorted(root.glob(f"*/{model_name}"))
        if matches:
            return matches[0]

    return None


def load_omniasr_tokenizer(
    tokenizer_name: str,
    *,
    tokenizer_model_path: str | Path | None = None,
) -> tuple[Tokenizer, Path]:
    """Load the OmniASR tokenizer, preferring an explicit or cached local model file."""
    _require_fairseq2()
    configure_project_cache_environment()

    if tokenizer_model_path is not None:
        tokenizer_path = resolve_project_path(tokenizer_model_path)
        return load_char_tokenizer(tokenizer_path, None), tokenizer_path

    cached_path = _get_cached_tokenizer_path(tokenizer_name)
    if cached_path is not None:
        return load_char_tokenizer(cached_path, None), cached_path

    tokenizer = load_tokenizer(tokenizer_name)
    cached_path = _get_cached_tokenizer_path(tokenizer_name)
    if cached_path is None:
        msg = (
            f"Tokenizer {tokenizer_name} loaded but a local tokenizer model path could not be resolved. "
            "Pass --tokenizer-model-path explicitly."
        )
        raise FileNotFoundError(msg)

    return tokenizer, cached_path


def build_pyctcdecode_labels(tokenizer_model_path: Path) -> tuple[list[str], set[str]]:
    """Build pyctcdecode labels in the exact OmniASR logit order."""
    _require_fairseq2()
    model = load_sentencepiece_model(tokenizer_model_path)
    labels: list[str] = []
    removable_tokens: set[str] = set()

    for idx in range(model.vocabulary_size):
        token = model.index_to_token(idx)

        if idx == 0:
            labels.append("")
            continue

        labels.append(token)

        if token in {"<pad>", "</s>", "<s>"}:
            removable_tokens.add(token)

    return labels, removable_tokens


def strip_special_tokens(text: str, removable_tokens: set[str]) -> str:
    """Remove special-token text artifacts after beam decoding."""
    cleaned = text
    for token in removable_tokens:
        cleaned = cleaned.replace(token, "")

    return _WHITESPACE_RE.sub(" ", cleaned).strip()


def load_custom_omniasr_ctc_model(
    checkpoint_path: str | Path,
    *,
    model_arch: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Wav2Vec2AsrModel:
    """Load a custom OmniASR CTC checkpoint using fairseq2's registered family."""
    _require_fairseq2()
    configure_project_cache_environment()

    hub = get_wav2vec2_asr_model_hub()
    config = hub.get_arch_config(model_arch)
    return hub.load_custom_model(
        resolve_project_path(checkpoint_path),
        config,
        device=device,
        dtype=dtype,
        progress=True,
    )


def make_inference_pipeline(
    *,
    checkpoint_path: str | Path,
    model_arch: str,
    tokenizer_name: str,
    tokenizer_model_path: str | Path | None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[ASRInferencePipeline, Path]:
    """Create an OmniASR inference pipeline for CTC decoding."""
    _require_omniasr()
    model = load_custom_omniasr_ctc_model(checkpoint_path, model_arch=model_arch, device=device, dtype=dtype)
    tokenizer, resolved_tokenizer_path = load_omniasr_tokenizer(
        tokenizer_name,
        tokenizer_model_path=tokenizer_model_path,
    )

    pipeline = ASRInferencePipeline(
        model_card=None,
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
    )
    return pipeline, resolved_tokenizer_path


def chunked(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    """Yield fixed-size chunks from a sequence."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def read_text_lines(path: str | Path) -> list[str]:
    """Read UTF-8 text lines without trailing newlines."""
    return resolve_project_path(path).read_text(encoding="utf-8").splitlines()


def write_text_lines(path: str | Path, lines: Iterable[str]) -> None:
    """Write UTF-8 text lines with trailing newlines."""
    output_path = resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def infer_split_from_eval_config(config_path: str | Path) -> dict[str, Any]:
    """Resolve checkpoint, tokenizer, dataset root, and split metadata from an eval config."""
    config = load_yaml_config(config_path)

    model_cfg = config["model"]
    dataset_cfg = config["dataset"]
    tokenizer_cfg = config["tokenizer"]

    summary_path = resolve_project_path(dataset_cfg["mixture_parquet_storage_config"]["dataset_summary_path"])

    return {
        "checkpoint_path": Path(model_cfg["path"]),
        "model_arch": model_cfg["arch"],
        "tokenizer_name": tokenizer_cfg["name"],
        "dataset_root": summary_path.parent,
        "valid_split": dataset_cfg["valid_split"],
    }


def make_decoder_factory(
    labels: Sequence[str],
    *,
    kenlm_model_path: str | Path | None,
    alpha: float,
    beta: float,
) -> Any:
    """Construct a pyctcdecode decoder lazily."""
    try:
        from pyctcdecode import build_ctcdecoder
    except ImportError as ex:  # pragma: no cover - optional dependency in CI
        msg = "pyctcdecode is required for beam decoding. Install it before using --decoder beam."
        raise ImportError(msg) from ex

    decoder_kwargs: dict[str, Any] = {
        "labels": list(labels),
        "alpha": alpha,
        "beta": beta,
    }
    if kenlm_model_path is not None:
        decoder_kwargs["kenlm_model_path"] = str(resolve_project_path(kenlm_model_path))

    return build_ctcdecoder(**decoder_kwargs)


def score_predictions(predictions: Sequence[str], references: Sequence[str]) -> dict[str, Any]:
    """Compute simple WER summary from aligned prediction/reference lists."""
    if len(predictions) != len(references):
        msg = "Predictions and references must have the same number of lines."
        raise ValueError(msg)

    from jiwer import wer

    score = wer(list(references), list(predictions))
    return {
        "num_examples": len(predictions),
        "wer": score * 100.0,
    }


def resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    """Resolve a dtype string with a CPU-safe fallback."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[dtype_name]
    if device.type == "cpu" and dtype != torch.float32:
        logger.warning(f"Falling back to float32 on CPU instead of requested {dtype_name}.")
        return torch.float32
    return dtype


def collate_decode_records(records: Sequence[DecodeResult]) -> tuple[list[str], list[str]]:
    """Split decode records into aligned prediction and reference lists."""
    return [record.prediction for record in records], [record.reference for record in records]


def decode_logits_with_argmax(
    logits: torch.Tensor,
    *,
    seq_len: int,
    token_decoder: Callable[[torch.Tensor], str],
) -> str:
    """Apply greedy CTC decoding to a single logit sequence."""
    pred_ids = torch.argmax(logits[:seq_len], dim=-1)
    mask = torch.ones(pred_ids.shape[0], dtype=torch.bool, device=pred_ids.device)
    if pred_ids.shape[0] > 1:
        mask[1:] = pred_ids[1:] != pred_ids[:-1]

    return _WHITESPACE_RE.sub(" ", token_decoder(pred_ids[mask])).strip()


def decode_ctc_logits(
    logits: torch.Tensor,
    *,
    seq_len: int,
    token_decoder: Callable[[torch.Tensor], str],
    decoder_kind: str,
    beam_decoder: Any = None,
    beam_width: int = 64,
    removable_tokens: set[str] | None = None,
) -> str:
    """Decode one CTC logit sequence with greedy or beam search."""
    if decoder_kind == "greedy":
        return decode_logits_with_argmax(logits, seq_len=seq_len, token_decoder=token_decoder)

    if decoder_kind != "beam":
        msg = f"Unsupported decoder kind: {decoder_kind}"
        raise ValueError(msg)

    if beam_decoder is None:
        msg = "Beam decoder must be initialized when decoder_kind='beam'."
        raise ValueError(msg)

    hypothesis = beam_decoder.decode(logits[:seq_len].float().cpu().numpy(), beam_width=beam_width)
    return strip_special_tokens(hypothesis, removable_tokens or set())
