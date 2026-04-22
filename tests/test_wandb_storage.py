from datetime import UTC, datetime, timedelta

from danish_asr.wandb_storage import (
    ArtifactVersionRow,
    RunFileRow,
    _bytes_to_gib,
    _parse_created_at,
    build_artifact_report,
    build_run_file_report,
    collect_run_file_rows,
    filter_artifact_rows,
)


def test_build_artifact_report_sorts_and_aggregates() -> None:
    rows = [
        ArtifactVersionRow(
            project="proj-a",
            artifact_type="model",
            collection="ckpt-a",
            version="ckpt-a:v0",
            size=10,
            aliases=("latest",),
            created_at=None,
        ),
        ArtifactVersionRow(
            project="proj-a",
            artifact_type="model",
            collection="ckpt-a",
            version="ckpt-a:v1",
            size=30,
            aliases=(),
            created_at=None,
        ),
        ArtifactVersionRow(
            project="proj-b",
            artifact_type="dataset",
            collection="data-b",
            version="data-b:v0",
            size=20,
            aliases=("prod",),
            created_at=None,
        ),
    ]

    report = build_artifact_report(rows)

    assert [row.version for row in report["artifact_versions"]] == ["ckpt-a:v1", "data-b:v0", "ckpt-a:v0"]
    assert report["project_totals"] == [("proj-a", 40), ("proj-b", 20)]
    assert report["collection_totals"] == [
        (("proj-a", "model", "ckpt-a"), 40),
        (("proj-b", "dataset", "data-b"), 20),
    ]


def test_build_run_file_report_sorts_and_aggregates() -> None:
    rows = [
        RunFileRow(project="proj-a", run_id="1", run_name="run-1", file_name="a.bin", size=5),
        RunFileRow(project="proj-b", run_id="2", run_name="run-2", file_name="b.bin", size=9),
        RunFileRow(project="proj-a", run_id="3", run_name="run-3", file_name="c.bin", size=7),
    ]

    report = build_run_file_report(rows)

    assert [row.file_name for row in report["run_files"]] == ["b.bin", "c.bin", "a.bin"]
    assert report["project_totals"] == [("proj-a", 12), ("proj-b", 9)]


def test_bytes_to_gib_uses_binary_units() -> None:
    assert _bytes_to_gib(1024**3) == 1.0


def test_collect_run_file_rows_skips_runs_with_broken_file_paginator() -> None:
    class DummyFile:
        def __init__(self, name: str, size: int) -> None:
            self.name = name
            self.size = size

    class GoodRun:
        id = "good"
        name = "good-run"

        def files(self):
            return [DummyFile("keep.bin", 10)]

    class BrokenRun:
        id = "broken"
        name = "broken-run"

        def files(self):
            class BrokenIterable:
                def __iter__(self):
                    raise TypeError("'NoneType' object is not subscriptable")

            return BrokenIterable()

    class DummyApi:
        def runs(self, path: str):
            assert path == "entity/proj"
            return [GoodRun(), BrokenRun()]

    rows = collect_run_file_rows(DummyApi(), "entity", ["proj"])

    assert rows == [RunFileRow(project="proj", run_id="good", run_name="good-run", file_name="keep.bin", size=10)]


def test_parse_created_at_handles_z_suffix() -> None:
    parsed = _parse_created_at("2026-04-22T11:32:00Z")

    assert parsed == datetime(2026, 4, 22, 11, 32, 0, tzinfo=UTC)


def test_filter_artifact_rows_respects_cleanup_filters() -> None:
    old_date = (datetime.now(UTC) - timedelta(days=30)).isoformat().replace("+00:00", "Z")
    new_date = (datetime.now(UTC) - timedelta(days=2)).isoformat().replace("+00:00", "Z")
    rows = [
        ArtifactVersionRow(
            project="CT_Scan_MLOps",
            artifact_type="model",
            collection="ct_scan_classifier_model",
            version="ct_scan_classifier_model:v1",
            size=2 * 1024**3,
            aliases=(),
            created_at=old_date,
        ),
        ArtifactVersionRow(
            project="CT_Scan_MLOps",
            artifact_type="model",
            collection="ct_scan_classifier_model",
            version="ct_scan_classifier_model:v2",
            size=3 * 1024**3,
            aliases=("best",),
            created_at=old_date,
        ),
        ArtifactVersionRow(
            project="CT_Scan_MLOps",
            artifact_type="dataset",
            collection="ct_scan_classifier_model",
            version="ct_scan_classifier_model:v3",
            size=4 * 1024**3,
            aliases=(),
            created_at=old_date,
        ),
        ArtifactVersionRow(
            project="CT_Scan_MLOps",
            artifact_type="model",
            collection="other_collection",
            version="other_collection:v0",
            size=5 * 1024**3,
            aliases=(),
            created_at=old_date,
        ),
        ArtifactVersionRow(
            project="CT_Scan_MLOps",
            artifact_type="model",
            collection="ct_scan_classifier_model",
            version="ct_scan_classifier_model:v4",
            size=2 * 1024**3,
            aliases=(),
            created_at=new_date,
        ),
    ]

    filtered = filter_artifact_rows(
        rows,
        collection="ct_scan_classifier_model",
        artifact_type="model",
        unaliased_only=True,
        min_size_gib=1.5,
        older_than_days=14,
    )

    assert [row.version for row in filtered] == ["ct_scan_classifier_model:v1"]
