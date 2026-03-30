"""Shared text normalization helpers for ASR baselines."""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")

# Danish filler words/sounds common in conversational speech
_FILLER_RE = re.compile(r"\b(?:øh+m*|hmm*|mm+|eh+m*|ah+)\b", re.IGNORECASE)

# Typographic character substitutions
_CHAR_SUBS: dict[str, str] = {
    "\u2018": "'",  # left single curly quote
    "\u2019": "'",  # right single curly quote
    "\u201c": '"',  # left double curly quote
    "\u201d": '"',  # right double curly quote
    "\u2013": "-",  # en-dash
    "\u2014": "-",  # em-dash
    "\u2026": "...",  # ellipsis
}
_CHAR_SUBS_RE = re.compile("|".join(re.escape(k) for k in _CHAR_SUBS))


def normalize_coral_text(text: str) -> str:
    """Normalize text following CoRal benchmark conventions.

    Applied before model-specific tokenization for both CTC and seq2seq baselines.
    For Wav2Vec2 CTC, compose with :func:`normalize_ctc_text` afterwards.

    Steps:
        1. NFKC unicode normalization (e.g. ``fi`` ligature -> ``fi``, ``²`` -> ``2``)
        2. Remove Danish filler words/sounds (``øh``, ``hmm``, ``mm``, etc.)
        3. Typographic character substitutions (curly quotes, dashes)
        4. Lowercase
        5. Collapse whitespace and strip
    """
    text = unicodedata.normalize("NFKC", text)
    text = _FILLER_RE.sub("", text)
    text = _CHAR_SUBS_RE.sub(lambda m: _CHAR_SUBS[m.group()], text)
    text = text.lower()
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_ctc_text(text: str) -> str:
    """Normalize text for CTC training and evaluation.

    Keep punctuation as-is for now, but make casing and whitespace deterministic so
    the tracked vocab asset and training-time labels stay aligned.
    """
    return _WHITESPACE_RE.sub(" ", text.strip().lower())
