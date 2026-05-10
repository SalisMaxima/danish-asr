from __future__ import annotations

import os
from pathlib import Path

from danish_asr.utils import configure_project_cache_environment


def test_configure_project_cache_environment_preserves_existing_cache_dirs(tmp_path: Path, monkeypatch) -> None:
    fairseq2_cache = tmp_path / "fairseq2_cache"
    hf_home = tmp_path / "hf_cache"
    tmp_dir = tmp_path / "tmp"

    monkeypatch.setenv("FAIRSEQ2_CACHE_DIR", str(fairseq2_cache))
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("TMPDIR", str(tmp_dir))

    configure_project_cache_environment()

    assert fairseq2_cache.is_dir()
    assert (fairseq2_cache / "assets").is_dir()
    assert hf_home.is_dir()
    assert tmp_dir.is_dir()


def test_configure_project_cache_environment_treats_blank_env_vars_as_unset(tmp_path: Path, monkeypatch) -> None:
    import danish_asr.utils as utils

    monkeypatch.setattr(utils, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("HF_HOME", " ")
    monkeypatch.setenv("FAIRSEQ2_CACHE_DIR", "")

    configure_project_cache_environment()

    assert Path(os.environ["HF_HOME"]).name == "huggingface"
    assert Path(os.environ["FAIRSEQ2_CACHE_DIR"]).name == "fairseq2"


def test_configure_project_cache_environment_normalizes_fairseq2_assets_dir(tmp_path: Path, monkeypatch) -> None:
    fairseq2_assets = tmp_path / "fairseq2_cache" / "assets"
    monkeypatch.setenv("FAIRSEQ2_CACHE_DIR", str(fairseq2_assets))

    configure_project_cache_environment()

    assert os.environ["FAIRSEQ2_CACHE_DIR"] == str(tmp_path / "fairseq2_cache")
    assert fairseq2_assets.is_dir()
