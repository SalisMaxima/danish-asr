from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.hpc.prepare_parquet_subset_eval import prepare_config


def _write_config(path: Path, summary_path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "model": {"name": "omniASR_LLM_300M_v2"},
                "dataset": {
                    "mixture_parquet_storage_config": {
                        "dataset_summary_path": str(summary_path),
                        "beta_corpus": 0.5,
                        "beta_language": 0.5,
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_prepare_config_writes_one_corpus_parquet_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_root = tmp_path / "data" / "parquet" / "version=0"
    (data_root / "corpus=coral_v3_read_aloud").mkdir(parents=True)
    (data_root / "corpus=coral_v3_conversation").mkdir(parents=True)
    (data_root / "language_distribution_0.tsv").write_text(
        "corpus\tlanguage\tsplit\thours\n"
        "coral_v3_read_aloud\tdan_Latn\ttest\t1.0\n"
        "coral_v3_conversation\tdan_Latn\ttest\t2.0\n",
        encoding="utf-8",
    )

    source_config = tmp_path / "config.yaml"
    output_config = tmp_path / "prepared" / "config.yaml"
    subset_root_parent = tmp_path / "subsets"
    _write_config(source_config, Path("data/parquet/version=0/language_distribution_read_aloud.tsv"))

    prepare_config(
        source_config=source_config,
        output_config=output_config,
        subset_root_parent=subset_root_parent,
        subset_corpus="coral_v3_read_aloud",
    )

    prepared = yaml.safe_load(output_config.read_text(encoding="utf-8"))
    summary_path = Path(prepared["dataset"]["mixture_parquet_storage_config"]["dataset_summary_path"])
    subset_root = subset_root_parent / "coral_v3_read_aloud" / "version=0"

    assert summary_path == subset_root / "language_distribution_read_aloud.tsv"
    assert summary_path.read_text(encoding="utf-8").splitlines() == [
        "corpus\tlanguage\tsplit\thours",
        "coral_v3_read_aloud\tdan_Latn\ttest\t1.0",
    ]
    assert (subset_root / "corpus=coral_v3_read_aloud").is_symlink()
    assert not (subset_root / "corpus=coral_v3_conversation").exists()


def test_prepare_config_combined_preserves_configured_summary(tmp_path: Path) -> None:
    source_config = tmp_path / "config.yaml"
    output_config = tmp_path / "prepared.yaml"
    summary_path = tmp_path / "data" / "parquet" / "version=0" / "language_distribution_0.tsv"
    _write_config(source_config, summary_path)

    prepare_config(
        source_config=source_config,
        output_config=output_config,
        subset_root_parent=tmp_path / "subsets",
        subset_corpus=None,
    )

    prepared = yaml.safe_load(output_config.read_text(encoding="utf-8"))
    assert Path(prepared["dataset"]["mixture_parquet_storage_config"]["dataset_summary_path"]) == summary_path
