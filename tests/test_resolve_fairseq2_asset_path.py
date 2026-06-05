from __future__ import annotations

from pathlib import Path

from scripts.hpc.resolve_fairseq2_asset_path import cache_dir_for_uri, resolve_cached_asset_uri


def test_resolve_cached_asset_uri_returns_single_cached_file(tmp_path: Path) -> None:
    uri = "https://example.test/model.pt"
    asset_dir = cache_dir_for_uri(uri, tmp_path)
    asset_dir.mkdir(parents=True)
    model_path = asset_dir / "model.pt"
    model_path.write_bytes(b"model")

    assert resolve_cached_asset_uri(uri, tmp_path) == model_path


def test_resolve_cached_asset_uri_returns_none_when_missing(tmp_path: Path) -> None:
    assert resolve_cached_asset_uri("https://example.test/model.pt", tmp_path) is None


def test_resolve_cached_asset_uri_supports_path_parameter(tmp_path: Path) -> None:
    uri = "https://example.test/archive.tar;path=nested%2Fmodel.pt"
    asset_dir = cache_dir_for_uri("https://example.test/archive.tar", tmp_path)
    model_path = asset_dir / "nested" / "model.pt"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"model")

    assert resolve_cached_asset_uri(uri, tmp_path) == model_path


def test_resolve_cached_asset_uri_accepts_file_uri(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"model")

    assert resolve_cached_asset_uri(model_path.as_uri(), tmp_path) == model_path
