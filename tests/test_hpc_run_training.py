"""Tests for HPC run_training metric parsing."""

from scripts.hpc.run_training import _MetricParser


def test_metric_parser_extracts_multiline_training_and_validation_metrics() -> None:
    parser = _MetricParser()

    metrics, step = parser.parse_line("Training Metrics (step 100) - CTC")
    assert metrics == {}
    assert step is None

    metrics, step = parser.parse_line("Loss: 5.432 | Gradient Norm: 1.23 |")
    assert metrics == {}
    assert step is None

    metrics, step = parser.parse_line("")
    assert step == 100
    assert metrics == {"train/loss": 5.432, "train/grad_norm": 1.23}

    metrics, step = parser.parse_line("Validation Metrics (step 500) - CTC")
    assert metrics == {}
    assert step is None

    metrics, step = parser.parse_line(
        "Loss: 103.296 | Unit Error Rate (UER): 23.98 | Word Error Rate (WER): 59.88 | Character Error Rate (CER): 42.11 |"
    )
    assert metrics == {}
    assert step is None

    metrics, step = parser.parse_line("")
    assert step == 500
    assert metrics == {"val/loss": 103.296, "val/uer": 23.98, "val/wer": 59.88, "val/cer": 42.11}


def test_metric_parser_extracts_legacy_single_line_metrics() -> None:
    parser = _MetricParser()

    metrics, step = parser.parse_line("| train | step 100 | loss 1.234 | gradient norm 0.9 |")
    assert step == 100
    assert metrics == {"train/loss": 1.234, "train/grad_norm": 0.9}

    metrics, step = parser.parse_line("| valid | step 500 | wer 0.42 | cer 0.12 | loss 2.1 |")
    assert step == 500
    assert metrics == {"val/wer": 0.42, "val/cer": 0.12, "val/loss": 2.1}
