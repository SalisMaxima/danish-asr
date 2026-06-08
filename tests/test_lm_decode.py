from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from danish_asr import lm as lm_module
from danish_asr.lm import build_pyctcdecode_labels, build_pyctcdecode_unigrams, decode_ctc_logits


def _token_decoder(indices: torch.Tensor) -> str:
    mapping = {0: "", 1: "a", 2: "b"}
    return "".join(mapping[int(idx)] for idx in indices)


def test_decode_ctc_logits_greedy_delegates_to_argmax_collapse() -> None:
    logits = torch.tensor(
        [
            [5.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 6.0, 0.0],
            [0.0, 0.0, 7.0],
        ]
    )

    assert decode_ctc_logits(logits, seq_len=4, token_decoder=_token_decoder, decoder_kind="greedy") == "ab"


def test_decode_ctc_logits_rejects_unknown_decoder_kind() -> None:
    with pytest.raises(ValueError, match="Unsupported decoder kind"):
        decode_ctc_logits(torch.zeros((2, 3)), seq_len=2, token_decoder=_token_decoder, decoder_kind="nope")


def test_decode_ctc_logits_requires_beam_decoder_for_beam() -> None:
    with pytest.raises(ValueError, match="Beam decoder must be initialized"):
        decode_ctc_logits(torch.zeros((2, 3)), seq_len=2, token_decoder=_token_decoder, decoder_kind="beam")


def test_decode_ctc_logits_beam_uses_seq_len_width_and_strips_tokens() -> None:
    class FakeBeamDecoder:
        def __init__(self) -> None:
            self.seen_logits: np.ndarray | None = None
            self.seen_beam_width: int | None = None

        def decode(self, logits: np.ndarray, *, beam_width: int) -> str:
            self.seen_logits = logits
            self.seen_beam_width = beam_width
            return "<s> hej </s> <pad>"

    decoder = FakeBeamDecoder()
    logits = torch.arange(15, dtype=torch.float32, requires_grad=True).reshape(5, 3)

    result = decode_ctc_logits(
        logits,
        seq_len=3,
        token_decoder=_token_decoder,
        decoder_kind="beam",
        beam_decoder=decoder,
        beam_width=32,
        removable_tokens={"<s>", "</s>", "<pad>"},
    )

    assert result == "hej"
    assert decoder.seen_beam_width == 32
    assert decoder.seen_logits is not None
    assert decoder.seen_logits.shape == (3, 3)


def test_build_pyctcdecode_labels_replaces_multichar_special_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSentencePieceModel:
        vocabulary_size = 5

        def index_to_token(self, index: int) -> str:
            return ["<s>", "<pad>", "</s>", "<unk>", "a"][index]

    monkeypatch.setattr(lm_module, "_FAIRSEQ2_AVAILABLE", True)
    monkeypatch.setattr(lm_module, "load_sentencepiece_model", lambda path: FakeSentencePieceModel())

    labels, removable_tokens = build_pyctcdecode_labels(Path("tokenizer.model"))

    assert labels == ["", "\ue000", "\ue001", "⁇", "a"]
    assert removable_tokens == {"\ue000", "\ue001", "⁇"}
    assert not any(len(label) > 1 for label in labels)


def test_build_pyctcdecode_unigrams_filters_noise() -> None:
    unigrams = build_pyctcdecode_unigrams(
        ["Hej, verden! \x03bad 🎸", "Paracetamol og Ibuprofen"],
        tokenizer_model_path=None,
    )

    assert "hej" in unigrams
    assert "verden" in unigrams
    assert "paracetamol" in unigrams
    assert "ibuprofen" in unigrams
    assert "bad" in unigrams
    assert "🎸" not in unigrams
