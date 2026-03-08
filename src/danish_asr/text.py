"""Shared text normalization helpers for ASR baselines."""

from __future__ import annotations

import re

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_ctc_text(text: str) -> str:
    """Normalize text for CTC training and evaluation.

    Keep punctuation as-is for now, but make casing and whitespace deterministic so
    the tracked vocab asset and training-time labels stay aligned.
    """
    return _WHITESPACE_RE.sub(" ", text.strip().lower())
