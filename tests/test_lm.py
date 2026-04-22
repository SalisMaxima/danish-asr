from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytest.importorskip("pyarrow", reason="pyarrow not installed (omni group); skipping LM parquet tests")

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from danish_asr.lm import (
    build_lm_corpus_from_parquet,
    decode_logits_with_argmax,
    normalize_lm_text,
    parse_valid_split,
    score_predictions,
    strip_special_tokens,
)
from scripts.lm.build_kenlm import run_kenlm_command


def _write_fairseq2_split(base: Path, corpus: str, split: str, texts: list[str]) -> None:
    split_dir = base / f"corpus={corpus}" / f"split={split}" / "language=dan_Latn"
    split_dir.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "text": texts,
            "audio_bytes": [b"flac"] * len(texts),
            "audio_size": [16000] * len(texts),
        }
    )
    pq.write_table(table, split_dir / "part-00000.parquet")


def test_normalize_lm_text_lowercases_and_strips_urls_and_metadata() -> None:
    assert normalize_lm_text("Speaker_ID=abc123 HEJ https://example.com") == "hej"


def test_parse_valid_split_handles_combined_and_subset() -> None:
    assert parse_valid_split("test") == ("test", None)
    assert parse_valid_split("test_coral_v3_read_aloud") == ("test", "coral_v3_read_aloud")


def test_build_lm_corpus_from_parquet_uses_train_only_and_deduplicates(tmp_path: Path) -> None:
    _write_fairseq2_split(tmp_path, "coral_v3_read_aloud", "train", ["Hej verden", "Hej verden"])
    _write_fairseq2_split(tmp_path, "coral_v3_conversation", "train", ["URL https://example.com test"])
    _write_fairseq2_split(tmp_path, "coral_v3_read_aloud", "dev", ["should not appear"])
    _write_fairseq2_split(tmp_path, "coral_v3_conversation", "test", ["should not appear"])

    texts, stats = build_lm_corpus_from_parquet(tmp_path)

    assert texts == ["hej verden", "url test"]
    assert stats.raw_examples == 3
    assert stats.unique_examples == 2
    assert stats.source_counts["coral_v3_read_aloud"] == 1
    assert stats.source_counts["coral_v3_conversation"] == 1


def test_run_kenlm_command_redirects_stdin_and_stdout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_run(command: list[str], check: bool, stdin, stdout) -> None:
        seen["command"] = command
        seen["stdin_name"] = None if stdin is None else Path(stdin.name).name
        seen["stdout_name"] = None if stdout is None else Path(stdout.name).name
        assert check is True

    monkeypatch.setattr(subprocess, "run", fake_run)

    stdin_path = tmp_path / "input.txt"
    stdout_path = tmp_path / "output.arpa"
    stdin_path.write_text("hej\n", encoding="utf-8")

    run_kenlm_command(["lmplz", "-o", "3"], stdin_path=stdin_path, stdout_path=stdout_path)

    assert seen["command"] == ["lmplz", "-o", "3"]
    assert seen["stdin_name"] == "input.txt"
    assert seen["stdout_name"] == "output.arpa"


def test_decode_logits_with_argmax_collapses_repeats() -> None:
    logits = torch.tensor(
        [
            [5.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 6.0, 0.0],
            [0.0, 0.0, 7.0],
            [0.0, 0.0, 7.0],
        ]
    )

    def token_decoder(indices: torch.Tensor) -> str:
        mapping = {0: "", 1: "a", 2: "b"}
        return "".join(mapping[int(idx)] for idx in indices)

    assert decode_logits_with_argmax(logits, seq_len=5, token_decoder=token_decoder) == "ab"


def test_strip_special_tokens_removes_known_artifacts() -> None:
    assert strip_special_tokens("<s> hej </s> <pad>", {"<s>", "</s>", "<pad>"}) == "hej"


def test_score_predictions_returns_percent_wer() -> None:
    summary = score_predictions(["hej verden"], ["hej verden"])
    assert summary["num_examples"] == 1
    assert summary["wer"] == 0.0
