"""Tests for model module."""

from __future__ import annotations

import pytest
import torch.nn as nn

from danish_asr.model import MODEL_REGISTRY, Wav2Vec2ASR, build_model


def test_model_registry_has_models():
    """Model registry should contain registered ASR models."""
    assert isinstance(MODEL_REGISTRY, dict)
    assert "wav2vec2_asr" in MODEL_REGISTRY
    assert "whisper_asr" in MODEL_REGISTRY


def test_build_model_unknown_raises():
    """build_model should raise ValueError for unknown model."""
    with pytest.raises(ValueError, match="Unknown model"):
        build_model({"name": "nonexistent_model_xyz"})


def test_wav2vec2_model_passes_vocab_size(monkeypatch):
    """Wav2Vec2 should size the CTC head from config instead of checkpoint defaults."""
    captured: dict[str, object] = {}

    class DummyHFModel(nn.Module):
        def freeze_feature_encoder(self) -> None:
            captured["froze_feature_encoder"] = True

    from transformers import Wav2Vec2ForCTC

    monkeypatch.setattr(
        Wav2Vec2ForCTC,
        "from_pretrained",
        lambda *args, **kwargs: captured.update({"args": args, "kwargs": kwargs}) or DummyHFModel(),
    )

    Wav2Vec2ASR.from_config(
        {
            "model_name": "facebook/wav2vec2-large-xlsr-53",
            "revision": None,
            "num_labels": 17,
            "pad_token_id": 0,
            "use_lora": False,
            "freeze_feature_extractor": True,
        }
    )

    assert captured["kwargs"]["vocab_size"] == 17
    assert captured["kwargs"]["pad_token_id"] == 0
    assert captured["froze_feature_encoder"] is True
