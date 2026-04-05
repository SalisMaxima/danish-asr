from pathlib import Path

from scripts.hpc.run_eval import _MetricParser, _select_eval_workspace


def test_metric_parser_extracts_multiline_validation_metrics() -> None:
    parser = _MetricParser()

    metrics, step = parser.parse_line("Validation Metrics (step 53000) - CTC")
    assert metrics == {}
    assert step is None

    metrics, step = parser.parse_line(
        "Loss: 103.296 | Unit Error Rate (UER): 23.98 | Word Error Rate (WER): 59.88 | Character Error Rate (CER): 42.11 |"
    )
    assert metrics == {}
    assert step is None

    metrics, step = parser.parse_line("")
    assert step == 53000
    assert metrics["val/loss"] == 103.296
    assert metrics["val/wer"] == 59.88
    assert metrics["val/cer"] == 42.11


def test_metric_parser_extracts_legacy_validation_metrics() -> None:
    parser = _MetricParser()

    metrics, step = parser.parse_line("| valid | step 53000 | wer 0.42 | cer 0.12 | loss 2.1 |")
    assert step == 53000
    assert metrics == {"val/wer": 0.42, "val/cer": 0.12, "val/loss": 2.1}


def test_select_eval_workspace_reuses_empty_base_dir(tmp_path: Path) -> None:
    workspace = _select_eval_workspace(tmp_path / "eval")
    assert workspace == tmp_path / "eval"
    assert workspace.exists()


def test_select_eval_workspace_creates_fresh_child_when_base_populated(tmp_path: Path) -> None:
    base = tmp_path / "eval"
    base.mkdir()
    (base / "old-artifact.txt").write_text("done")

    workspace = _select_eval_workspace(base)

    assert workspace.parent == base
    assert workspace.name.startswith("run_")
    assert workspace.exists()
    assert workspace != base
