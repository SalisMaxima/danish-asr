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


def get_project_fairseq2_cache_dir() -> Path:
    """Return the project-local fairseq2 cache directory."""
    return _PROJECT_ROOT / ".cache" / "fairseq2"


def configure_project_cache_environment() -> None:
    """Pin model and dataset caches to configured cache directories.

    Existing environment variables win so HPC jobs can place large assets on
    scratch storage. When unset, default to project-local cache directories to
    keep assets out of the user's home/root filesystem.
    """
    hf_home = Path(os.environ.get("HF_HOME", get_project_hf_cache_dir())).expanduser()
    hub_cache = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", hf_home / "hub")).expanduser()
    datasets_cache = Path(os.environ.get("HF_DATASETS_CACHE", hf_home / "datasets")).expanduser()
    torch_home = Path(os.environ.get("TORCH_HOME", _PROJECT_ROOT / ".cache" / "torch")).expanduser()
    fairseq2_cache = Path(os.environ.get("FAIRSEQ2_CACHE_DIR", get_project_fairseq2_cache_dir())).expanduser()
    fairseq2_assets = fairseq2_cache / "assets"
    tmp_dir = Path(os.environ.get("TMPDIR", _PROJECT_ROOT / ".cache" / "tmp")).expanduser()

    for directory in (hf_home, hub_cache, datasets_cache, torch_home, fairseq2_cache, fairseq2_assets, tmp_dir):
        directory.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(hub_cache)
    os.environ["TORCH_HOME"] = str(torch_home)
    os.environ["FAIRSEQ2_CACHE_DIR"] = str(fairseq2_cache)
    os.environ["TMPDIR"] = str(tmp_dir)


def get_device() -> torch.device:
    """Get the best available device. Priority: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
