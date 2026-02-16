"""ASR model definitions with LoRA fine-tuning support."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch
from loguru import logger
from torch import nn

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, type[ModelConfigFactory]] = {}


def register_model(name: str):
    """Decorator to register a model class."""

    def decorator(cls: type[ModelConfigFactory]) -> type[ModelConfigFactory]:
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


@runtime_checkable
class ModelConfigFactory(Protocol):
    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> nn.Module: ...


def build_model(cfg: dict[str, Any]) -> nn.Module:
    """Build a model from config."""
    name = cfg.get("name", "wav2vec2_asr")
    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        msg = f"Unknown model '{name}'. Available: {available}"
        raise ValueError(msg)
    model_cls = MODEL_REGISTRY[name]
    return model_cls.from_config(cfg)


@register_model("wav2vec2_asr")
class Wav2Vec2ASR(nn.Module):
    """Wav2Vec2 for CTC-based ASR with optional LoRA fine-tuning.

    Uses HuggingFace Wav2Vec2ForCTC as a baseline Danish ASR model.
    Supports LoRA via PEFT for parameter-efficient fine-tuning.
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-large-xlsr-53",
        revision: str | None = None,
        num_labels: int = 32,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: list[str] | None = None,
        freeze_feature_extractor: bool = True,
    ):
        super().__init__()
        from transformers import Wav2Vec2ForCTC

        # Build kwargs for from_pretrained
        kwargs = {
            "ctc_loss_reduction": "mean",
            "pad_token_id": 0,
        }
        if revision is not None:
            kwargs["revision"] = revision

        # Security note: revision can be None (use latest checkpoint) or pinned.
        # For research baselines, using standard checkpoints is intentional.
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name, **kwargs)  # nosec B615

        if freeze_feature_extractor:
            self.model.freeze_feature_encoder()
            logger.info("Froze feature extractor")

        if use_lora:
            self._apply_lora(lora_r, lora_alpha, lora_dropout, lora_target_modules)

    def _apply_lora(self, r: int, alpha: int, dropout: float, target_modules: list[str] | None) -> None:
        from peft import LoraConfig, get_peft_model

        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "k_proj", "out_proj"]

        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_config)
        trainable, total = self.model.get_nb_trainable_parameters()
        logger.info(f"LoRA applied: {trainable:,} trainable / {total:,} total ({100 * trainable / total:.1f}%)")

    def forward(
        self, input_values: torch.Tensor, attention_mask: torch.Tensor | None = None, labels: torch.Tensor | None = None
    ) -> dict:
        outputs = self.model(input_values=input_values, attention_mask=attention_mask, labels=labels)
        return {"loss": outputs.loss, "logits": outputs.logits}

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Wav2Vec2ASR:
        return cls(
            model_name=cfg.get("model_name", "facebook/wav2vec2-large-xlsr-53"),
            revision=cfg.get("revision", "main"),
            num_labels=cfg.get("num_labels", 32),
            use_lora=cfg.get("use_lora", True),
            lora_r=cfg.get("lora_r", 8),
            lora_alpha=cfg.get("lora_alpha", 16),
            lora_dropout=cfg.get("lora_dropout", 0.1),
            lora_target_modules=cfg.get("lora_target_modules"),
            freeze_feature_extractor=cfg.get("freeze_feature_extractor", True),
        )


@register_model("whisper_asr")
class WhisperASR(nn.Module):
    """OpenAI Whisper for Danish ASR with optional LoRA fine-tuning.

    Uses HuggingFace WhisperForConditionalGeneration.
    Supports LoRA for parameter-efficient fine-tuning.
    """

    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3",
        revision: str | None = None,
        language: str = "da",
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: list[str] | None = None,
    ):
        super().__init__()
        from transformers import WhisperForConditionalGeneration

        # Build kwargs for from_pretrained
        kwargs = {}
        if revision is not None:
            kwargs["revision"] = revision

        # Security note: revision can be None (use latest checkpoint) or pinned.
        # For research baselines, using standard checkpoints is intentional.
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name, **kwargs)  # nosec B615
        self.language = language

        if use_lora:
            self._apply_lora(lora_r, lora_alpha, lora_dropout, lora_target_modules)

    def _apply_lora(self, r: int, alpha: int, dropout: float, target_modules: list[str] | None) -> None:
        from peft import LoraConfig, get_peft_model

        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "k_proj", "out_proj"]

        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_config)
        trainable, total = self.model.get_nb_trainable_parameters()
        logger.info(f"LoRA applied: {trainable:,} trainable / {total:,} total ({100 * trainable / total:.1f}%)")

    def forward(self, input_features: torch.Tensor, labels: torch.Tensor | None = None) -> dict:
        outputs = self.model(input_features=input_features, labels=labels)
        return {"loss": outputs.loss, "logits": outputs.logits}

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> WhisperASR:
        return cls(
            model_name=cfg.get("model_name", "openai/whisper-large-v3"),
            revision=cfg.get("revision", "main"),
            language=cfg.get("language", "da"),
            use_lora=cfg.get("use_lora", True),
            lora_r=cfg.get("lora_r", 8),
            lora_alpha=cfg.get("lora_alpha", 16),
            lora_dropout=cfg.get("lora_dropout", 0.1),
            lora_target_modules=cfg.get("lora_target_modules"),
        )
