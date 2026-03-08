"""Tests for the training module."""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from danish_asr import train as train_module


class _DummyProcessor:
    def __init__(self):
        self.calls: list[tuple[torch.Tensor, bool]] = []

    def batch_decode(self, tokens: torch.Tensor, skip_special_tokens: bool = False) -> list[str]:
        self.calls.append((tokens, skip_special_tokens))
        return ["hej verden"]


class _DummyTokenizer:
    pass


class _DummySeq2SeqInner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.generate_calls: list[dict] = []

    def generate(self, input_features: torch.Tensor, **kwargs) -> torch.Tensor:
        self.generate_calls.append({"input_features": input_features, **kwargs})
        return torch.tensor([[1, 2, 3]])


class _DummySeq2SeqModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _DummySeq2SeqInner()

    def forward(self, input_features: torch.Tensor, labels: torch.Tensor) -> dict:
        del input_features, labels
        return {"loss": torch.tensor(0.25), "logits": torch.randn(1, 3, 4)}


def test_seq2seq_eval_uses_configured_language(monkeypatch):
    cfg = OmegaConf.create(
        {
            "model": {
                "name": "whisper_asr",
                "model_name": "openai/whisper-large-v3",
                "language": "kl",
            },
            "train": {
                "optimizer": {"lr": 1e-4, "weight_decay": 0.01, "betas": [0.9, 0.999]},
                "scheduler": {"warmup_steps": 10},
            },
            "hardware": {},
        }
    )
    processor = _DummyProcessor()
    tokenizer = _DummyTokenizer()
    model = _DummySeq2SeqModel()

    monkeypatch.setattr(train_module, "build_model", lambda cfg: model)
    monkeypatch.setattr(train_module.ASRLitModel, "_build_processor", lambda self: (processor, tokenizer))

    lit_model = train_module.ASRLitModel(cfg)
    lit_model.log = lambda *args, **kwargs: None
    batch = {
        "input_features": torch.randn(1, 80, 4),
        "labels": torch.tensor([[1, 2, 3]]),
        "text": ["hej verden"],
    }

    lit_model.validation_step(batch, 0)
    lit_model.test_step(batch, 0)

    assert model.model.generate_calls[0]["language"] == "kl"
    assert model.model.generate_calls[1]["language"] == "kl"
    assert lit_model._val_predictions == ["hej verden"]
    assert lit_model._test_predictions == ["hej verden"]


def test_epoch_end_metrics_clear_accumulators(monkeypatch):
    cfg = OmegaConf.create(
        {
            "model": {
                "name": "whisper_asr",
                "model_name": "openai/whisper-large-v3",
                "language": "da",
            },
            "train": {
                "optimizer": {"lr": 1e-4, "weight_decay": 0.01, "betas": [0.9, 0.999]},
                "scheduler": {"warmup_steps": 10},
            },
            "hardware": {},
        }
    )

    monkeypatch.setattr(train_module, "build_model", lambda cfg: _DummySeq2SeqModel())
    monkeypatch.setattr(
        train_module.ASRLitModel,
        "_build_processor",
        lambda self: (_DummyProcessor(), _DummyTokenizer()),
    )
    monkeypatch.setattr(train_module, "compute_wer", lambda predictions, references: 0.1)
    monkeypatch.setattr(train_module, "compute_cer", lambda predictions, references: 0.05)

    logged: list[tuple[str, float]] = []
    lit_model = train_module.ASRLitModel(cfg)
    lit_model.log = lambda name, value, **kwargs: logged.append((name, float(value)))
    lit_model._val_predictions = ["hej"]
    lit_model._val_references = ["hej"]
    lit_model._test_predictions = ["hej"]
    lit_model._test_references = ["hej"]

    lit_model.on_validation_epoch_end()
    lit_model.on_test_epoch_end()

    assert ("val_wer", 0.1) in logged
    assert ("val_cer", 0.05) in logged
    assert ("test_wer", 0.1) in logged
    assert ("test_cer", 0.05) in logged
    assert lit_model._val_predictions == []
    assert lit_model._val_references == []
    assert lit_model._test_predictions == []
    assert lit_model._test_references == []
