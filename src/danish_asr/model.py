"""Model definitions and registry.

Uses a decorator-based registry pattern for extensible model building.
Register new models with @register_model("name") and implement from_config().
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch.nn as nn
from omegaconf import DictConfig

# Model registry for extensible model building


class ModelConfigFactory(Protocol):
    @classmethod
    def from_config(cls, cfg: DictConfig) -> nn.Module: ...


MODEL_REGISTRY: dict[str, type[ModelConfigFactory]] = {}


def register_model(*names: str) -> Callable[[type[nn.Module]], type[nn.Module]]:
    """Decorator to register a model class with one or more names.

    Args:
        *names: One or more names to register the model under (case-insensitive).

    Returns:
        Decorator function that registers the class and returns it unchanged.

    Example:
        @register_model("my_model")
        class MyModel(nn.Module):
            ...
    """

    def decorator(cls: type[nn.Module]) -> type[nn.Module]:
        for name in names:
            MODEL_REGISTRY[name.lower()] = cls
        return cls

    return decorator


def build_model(cfg: DictConfig) -> nn.Module:
    """Build model from Hydra config using the model registry.

    Args:
        cfg: Hydra config containing model parameters.
             Expected to have cfg.model.name specifying which model to build.

    Returns:
        Configured model instance.

    Raises:
        ValueError: If the model name is not found in the registry.
    """
    name = cfg.model.name.lower()

    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model: '{name}'. Available models: {available}")

    model_cls = MODEL_REGISTRY[name]
    return model_cls.from_config(cfg)
