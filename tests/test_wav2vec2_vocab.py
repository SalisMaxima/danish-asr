"""Tests for Wav2Vec2 vocabulary generation."""

from __future__ import annotations

from scripts.build_wav2vec2_vocab import build_ctc_vocab


def test_build_ctc_vocab_is_deterministic():
    vocab = build_ctc_vocab(["Hej verden", "Hej igen"])

    assert vocab["[PAD]"] == 0
    assert vocab["[UNK]"] == 1
    assert vocab["|"] == 2
    assert " " not in vocab
    assert "h" in vocab
    assert "e" in vocab


def test_build_ctc_vocab_normalizes_case():
    vocab = build_ctc_vocab(["ABC", "abc"])

    assert "a" in vocab
    assert "A" not in vocab
