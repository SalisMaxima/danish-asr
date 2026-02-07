"""Tests for ASR metrics."""

from __future__ import annotations

from danish_asr.metrics import compute_cer, compute_metrics, compute_wer


def test_compute_wer_perfect():
    """WER should be 0 for identical predictions."""
    assert compute_wer(["hello world"], ["hello world"]) == 0.0


def test_compute_wer_mismatch():
    """WER should be > 0 for different predictions."""
    wer = compute_wer(["hello world"], ["hello earth"])
    assert wer > 0.0


def test_compute_cer_perfect():
    """CER should be 0 for identical predictions."""
    assert compute_cer(["hello"], ["hello"]) == 0.0


def test_compute_metrics_returns_both():
    """compute_metrics should return both WER and CER."""
    result = compute_metrics(["hello world"], ["hello world"])
    assert "wer" in result
    assert "cer" in result
    assert result["wer"] == 0.0
    assert result["cer"] == 0.0
