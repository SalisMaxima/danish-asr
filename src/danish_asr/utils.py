"""Shared utility functions."""

from __future__ import annotations

import os
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_project_root() -> Path:
    """Return the repository root."""
    return _PROJECT_ROOT


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the repository root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return _PROJECT_ROOT / candidate


def get_project_hf_cache_dir() -> Path:
    """Return the project-local Hugging Face cache directory."""
    return _PROJECT_ROOT / ".cache" / "huggingface"


def configure_project_cache_environment() -> None:
    """Pin model and dataset caches to the project drive.

    This keeps large Hugging Face and Torch assets out of the user's home/root
    filesystem, which is too small for CoRal and model checkpoints.
    """
    hf_home = get_project_hf_cache_dir()
    hub_cache = hf_home / "hub"
    datasets_cache = hf_home / "datasets"
    torch_home = _PROJECT_ROOT / ".cache" / "torch"

    for directory in (hf_home, hub_cache, datasets_cache, torch_home):
        directory.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(hub_cache)
    os.environ["TORCH_HOME"] = str(torch_home)


def get_device() -> torch.device:
    """Get the best available device. Priority: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
