"""Tests for model module."""

from __future__ import annotations

import pytest

from danish_asr.model import MODEL_REGISTRY, build_model


def test_model_registry_has_models():
    """Model registry should contain registered ASR models."""
    assert isinstance(MODEL_REGISTRY, dict)
    assert "wav2vec2_asr" in MODEL_REGISTRY
    assert "whisper_asr" in MODEL_REGISTRY


def test_build_model_unknown_raises():
    """build_model should raise ValueError for unknown model."""
    with pytest.raises(ValueError, match="Unknown model"):
        build_model({"name": "nonexistent_model_xyz"})
