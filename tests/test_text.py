"""Tests for text normalization helpers."""

from danish_asr.text import normalize_coral_text, normalize_ctc_text


def test_normalize_coral_text_applies_nfkc_and_substitutions() -> None:
    text = "ﬁ ² “Hej”—..."
    assert normalize_coral_text(text) == 'fi 2 "hej"-...'


def test_normalize_coral_text_removes_fillers_and_normalizes_spaces() -> None:
    text = "  Øhm   det   er   hmm   fint  "
    assert normalize_coral_text(text) == "det er fint"


def test_normalize_ctc_text_composition_is_deterministic() -> None:
    text = "  Hallo   VERDEN  "
    assert normalize_ctc_text(normalize_coral_text(text)) == "hallo verden"
