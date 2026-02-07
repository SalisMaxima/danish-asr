"""ASR evaluation metrics (WER, CER)."""

from __future__ import annotations

import jiwer


def compute_wer(predictions: list[str], references: list[str]) -> float:
    """Compute Word Error Rate.

    Args:
        predictions: Model transcriptions
        references: Ground truth transcriptions

    Returns:
        WER as a float between 0.0 and 1.0+
    """
    return jiwer.wer(references, predictions)


def compute_cer(predictions: list[str], references: list[str]) -> float:
    """Compute Character Error Rate.

    Args:
        predictions: Model transcriptions
        references: Ground truth transcriptions

    Returns:
        CER as a float between 0.0 and 1.0+
    """
    return jiwer.cer(references, predictions)


def compute_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute all ASR metrics."""
    return {
        "wer": compute_wer(predictions, references),
        "cer": compute_cer(predictions, references),
    }
