"""Tests for model module."""

from __future__ import annotations

import pytest

from danish_asr.model import MODEL_REGISTRY, build_model


def test_model_registry_exists():
    """Model registry should be a dict."""
    assert isinstance(MODEL_REGISTRY, dict)


def test_build_model_unknown_raises():
    """build_model should raise ValueError for unknown model."""
    from unittest.mock import MagicMock

    cfg = MagicMock()
    cfg.model.name = "nonexistent_model_xyz"
    with pytest.raises(ValueError, match="Unknown model"):
        build_model(cfg)
