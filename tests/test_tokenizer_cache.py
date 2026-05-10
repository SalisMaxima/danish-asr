from __future__ import annotations

from pathlib import Path

import pytest


class FakeUri:
    path = "/remote/omniasr_tokenizer.model"


class FakeField:
    def as_uri(self) -> FakeUri:
        return FakeUri()


class FakeCard:
    def field(self, name: str) -> FakeField:
        assert name == "tokenizer"
        return FakeField()


class FakeStore:
    def retrieve_card(self, name: str) -> FakeCard:
        assert name == "omniASR_tokenizer_written_v2"
        return FakeCard()


@pytest.fixture
def fake_fairseq2_store(monkeypatch: pytest.MonkeyPatch) -> None:
    import danish_asr.lm as lm

    monkeypatch.setattr(lm, "_require_fairseq2", lambda: None)
    monkeypatch.setattr(lm, "get_asset_store", lambda: FakeStore())


def test_get_cached_tokenizer_path_uses_active_fairseq2_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_fairseq2_store: None
) -> None:
    import danish_asr.lm as lm

    tokenizer_path = tmp_path / "fairseq2_cache" / "assets" / "abc123" / "omniasr_tokenizer.model"
    tokenizer_path.parent.mkdir(parents=True)
    tokenizer_path.write_text("tokenizer", encoding="utf-8")

    monkeypatch.setenv("FAIRSEQ2_CACHE_DIR", str(tmp_path / "fairseq2_cache"))

    assert lm._get_cached_tokenizer_path("omniASR_tokenizer_written_v2") == tokenizer_path


def test_get_cached_tokenizer_path_accepts_assets_dir_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_fairseq2_store: None
) -> None:
    import danish_asr.lm as lm

    assets_dir = tmp_path / "fairseq2_cache" / "assets"
    tokenizer_path = assets_dir / "abc123" / "omniasr_tokenizer.model"
    tokenizer_path.parent.mkdir(parents=True)
    tokenizer_path.write_text("tokenizer", encoding="utf-8")

    monkeypatch.setenv("FAIRSEQ2_CACHE_DIR", str(assets_dir))

    assert lm._get_cached_tokenizer_path("omniASR_tokenizer_written_v2") == tokenizer_path


def test_get_cached_tokenizer_path_ignores_blank_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_fairseq2_store: None
) -> None:
    import danish_asr.lm as lm

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FAIRSEQ2_CACHE_DIR", " ")

    assert lm._get_cached_tokenizer_path("omniASR_tokenizer_written_v2") is None
    assert not (tmp_path / "assets").exists()
