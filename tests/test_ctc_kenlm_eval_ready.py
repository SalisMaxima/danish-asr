from __future__ import annotations

import os
from pathlib import Path

from scripts.hpc.check_ctc_kenlm_eval_ready import (
    _checkpoint_exists,
    load_manifest,
    validate_manifest,
)
from scripts.hpc.collect_ctc_kenlm_results import _collect_root


def _write_manifest(path: Path, root: Path) -> None:
    checkpoint = root / "models" / "model.pt"
    kenlm = root / "lm.bin"
    parquet_dir = root / "parquet" / "corpus=coral_v3_read_aloud" / "split=test" / "language=dan_Latn"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"model")
    kenlm.write_bytes(b"lm")
    parquet_dir.mkdir(parents=True)
    (parquet_dir / "part-00000.parquet").write_bytes(b"parquet")
    path.write_text(
        f"""
artifacts:
  kenlm_binary: {kenlm}
  parquet_root: {root}/parquet
  hf_cache_dir: /work3/${{USER}}/hf_cache
output_roots:
  my_method: {root}/outputs/my
  coral_method: {root}/outputs/coral
  results: {root}/outputs/results
tokenizer:
  name: omniASR_tokenizer_written_v2
  model_path:
beam:
  width: 64
  alpha: 0.5
  beta: 1.5
coral:
  min_seconds: 0.5
  max_seconds: 10.0
  subsets: [read_aloud]
my_method:
  dataset_splits:
    - label: read_aloud
      split: test_coral_v3_read_aloud
models:
  - label: tiny
    arch: 300m_v2
    checkpoint_path: {checkpoint}
    batch_size: 1
""",
        encoding="utf-8",
    )


def test_preflight_manifest_expands_env_and_validates_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("USER", os.environ.get("USER", "s204696"))
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, tmp_path)

    manifest = load_manifest(manifest_path)
    errors = validate_manifest(
        manifest,
        method="my",
        require_imports=False,
        require_cuda=False,
        require_kenlm=True,
    )

    assert errors == []
    assert manifest["artifacts"]["hf_cache_dir"].endswith("/hf_cache")


def test_preflight_reports_missing_checkpoint(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("USER", os.environ.get("USER", "s204696"))
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, tmp_path)
    manifest = load_manifest(manifest_path)
    Path(manifest["models"][0]["checkpoint_path"]).unlink()

    errors = validate_manifest(
        manifest,
        method="my",
        require_imports=False,
        require_cuda=False,
        require_kenlm=True,
    )

    assert any("Checkpoint not found" in error for error in errors)


def test_collect_results_reads_plain_and_coral_scores(tmp_path: Path) -> None:
    my_run = tmp_path / "my" / "300m" / "combined" / "greedy"
    coral_run = tmp_path / "coral" / "300m" / "read_aloud" / "beam_lm"
    my_run.mkdir(parents=True)
    coral_run.mkdir(parents=True)
    (my_run / "scores.json").write_text('{"num_examples": 2, "wer": 10.0, "cer": 5.0}\n', encoding="utf-8")
    (my_run / "metadata.json").write_text('{"beam_width": 64, "alpha": 0.5, "beta": 1.5}\n', encoding="utf-8")
    (my_run / "SUCCESS").write_text("\n", encoding="utf-8")
    (coral_run / "scores.json").write_text(
        '{"scores": {"num_examples": 1, "cer_coral": 3.0, "wer_coral": 4.0},'
        ' "metadata": {"beam_width": 32, "alpha": 0.7, "beta": 1.2}}\n',
        encoding="utf-8",
    )

    rows = _collect_root(tmp_path / "my", "my_method") + _collect_root(tmp_path / "coral", "coral_method")

    assert rows[0]["wer"] == 10.0
    assert rows[0]["success"] is True
    assert rows[1]["cer_coral"] == 3.0
    assert rows[1]["beam_width"] == 32
    assert rows[1]["success"] is False  # coral run has no SUCCESS file


def test_checkpoint_exists_recognises_plain_file(tmp_path: Path) -> None:
    pt = tmp_path / "model.pt"
    pt.write_bytes(b"")
    assert _checkpoint_exists(pt) is True


def test_checkpoint_exists_recognises_sharded_fairseq2_format(tmp_path: Path) -> None:
    shard_file = tmp_path / "pp_00" / "tp_00" / "sdp_00.pt"
    shard_file.parent.mkdir(parents=True)
    shard_file.write_bytes(b"")
    assert _checkpoint_exists(tmp_path) is True


def test_checkpoint_exists_recognises_model_prefixed_sharded_format(tmp_path: Path) -> None:
    shard_file = tmp_path / "model" / "pp_00" / "tp_00" / "sdp_00.pt"
    shard_file.parent.mkdir(parents=True)
    shard_file.write_bytes(b"")
    assert _checkpoint_exists(tmp_path) is True


def test_checkpoint_exists_returns_false_for_empty_dir(tmp_path: Path) -> None:
    assert _checkpoint_exists(tmp_path) is False


def test_checkpoint_exists_returns_false_for_missing_path(tmp_path: Path) -> None:
    assert _checkpoint_exists(tmp_path / "nonexistent") is False


def test_collect_results_skips_corrupt_json(tmp_path: Path) -> None:
    good = tmp_path / "300m" / "split" / "greedy"
    bad = tmp_path / "300m" / "split" / "beam"
    good.mkdir(parents=True)
    bad.mkdir(parents=True)
    (good / "scores.json").write_text('{"num_examples": 5, "wer": 20.0}', encoding="utf-8")
    (bad / "scores.json").write_text("{broken json", encoding="utf-8")

    rows = _collect_root(tmp_path, "my_method")

    assert len(rows) == 1
    assert rows[0]["wer"] == 20.0


def test_preflight_reports_wrong_hf_cache_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("USER", "s204696")
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, tmp_path)
    manifest = load_manifest(manifest_path)
    manifest["artifacts"]["hf_cache_dir"] = "/home/s204696/wrong_cache"

    errors = validate_manifest(
        manifest,
        method="coral",
        require_imports=False,
        require_cuda=False,
        require_kenlm=False,
    )

    assert any("HF cache must stay under" in e for e in errors)
