from pathlib import Path

from scripts.hpc.run_eval import (
    _backup_score_file,
    _get_score_file,
    _MetricParser,
    _read_score_file,
    _restore_score_file,
    _select_eval_workspace,
)


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


def test_metric_parser_extracts_inline_evaluation_metrics() -> None:
    parser = _MetricParser()

    metrics, step = parser.parse_line(
        "Evaluation Metrics - CTC Loss: 42.7155 | Unit Error Rate (UER): 11.3601 | Word Error Rate (WER): 30.1429 | Data Time: 1s | Compute Time: 68s"
    )

    assert step is None
    assert metrics == {"eval/loss": 42.7155, "eval/wer": 30.1429}


def test_metric_parser_extracts_wrapped_evaluation_metrics() -> None:
    parser = _MetricParser()

    metrics, step = parser.parse_line("Evaluation Metrics - CTC")
    assert metrics == {}
    assert step is None

    metrics, step = parser.parse_line("Word Error Rate")
    assert metrics == {}
    assert step is None

    metrics, step = parser.parse_line("(WER): 30.1429 | Data Time: 1s |")
    assert metrics == {"eval/wer": 30.1429}
    assert step is None

    metrics, step = parser.parse_line("Character Error Rate")
    assert metrics == {}
    assert step is None

    metrics, step = parser.parse_line("(CER): 12.3456 | Compute Time: 68s")
    assert metrics == {"eval/cer": 12.3456}
    assert step is None

    metrics, step = parser.parse_line("")
    assert step is None
    assert metrics == {}


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


def test_get_score_file_returns_checkpoint_scores_path(tmp_path: Path) -> None:
    model_path = tmp_path / "ws_1.abc" / "checkpoints" / "step_53000"
    assert _get_score_file(model_path) == tmp_path / "ws_1.abc" / "checkpoints" / "scores" / "step_53000.txt"


def test_get_score_file_handles_model_subdirectory_path(tmp_path: Path) -> None:
    model_path = tmp_path / "ws_1.abc" / "checkpoints" / "step_53000" / "model"
    assert _get_score_file(model_path) == tmp_path / "ws_1.abc" / "checkpoints" / "scores" / "step_53000.txt"


def test_backup_score_file_uses_numbered_suffix_when_default_backup_exists(tmp_path: Path) -> None:
    score_file = tmp_path / "step_53000.txt"
    score_file.write_text("-59.88\n")
    (tmp_path / "step_53000.val.bak").write_text("-61.00\n")

    backup = _backup_score_file(score_file)

    assert backup == tmp_path / "step_53000.val.1.bak"
    assert backup.read_text() == "-59.88\n"
    assert not score_file.exists()


def test_restore_score_file_restores_validation_and_preserves_eval_score(tmp_path: Path) -> None:
    score_file = tmp_path / "step_53000.txt"
    score_file.write_text("-59.88\n")

    backup = _backup_score_file(score_file)
    assert backup is not None

    score_file.write_text("-30.14\n")
    eval_backup = _restore_score_file(score_file, backup)

    assert eval_backup == tmp_path / "step_53000.test.bak"
    assert eval_backup.read_text() == "-30.14\n"
    assert score_file.read_text() == "-59.88\n"
    assert not backup.exists()


def test_read_score_file_returns_absolute_value_for_valid_score(tmp_path: Path) -> None:
    score_file = tmp_path / "step_53000.txt"
    score_file.write_text("-59.88\n")

    assert _read_score_file(score_file) == 59.88


def test_read_score_file_returns_none_for_invalid_or_empty_content(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.txt"
    invalid.write_text("not-a-number\n")
    empty = tmp_path / "empty.txt"
    empty.write_text("")

    assert _read_score_file(invalid) is None
    assert _read_score_file(empty) is None
