"""Tests for HPC run_training metric parsing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.hpc.run_training import _DuplicateColWarningFilter, _MetricParser


# ---------------------------------------------------------------------------
# _DuplicateColWarningFilter
# ---------------------------------------------------------------------------


class TestDuplicateColWarningFilter:
    def test_first_warning_returns_replacement_and_suppresses_original(self) -> None:
        f = _DuplicateColWarningFilter()
        suppress, replacement = f.process("DataFrame columns are not unique, some columns will be omitted.")
        assert suppress is True
        assert replacement is not None
        assert "further occurrences suppressed" in replacement

    def test_subsequent_warnings_suppressed_without_replacement(self) -> None:
        f = _DuplicateColWarningFilter()
        f.process("DataFrame columns are not unique")  # first occurrence
        suppress, replacement = f.process("DataFrame columns are not unique")
        assert suppress is True
        assert replacement is None

    def test_context_lines_after_warning_are_suppressed(self) -> None:
        f = _DuplicateColWarningFilter()
        f.process("DataFrame columns are not unique")
        # Next 2 lines (whatever they contain) should be suppressed
        s1, r1 = f.process("  File mixture_parquet_storage.py, line 42")
        assert s1 is True
        assert r1 is None
        s2, r2 = f.process("    records = table.to_pandas()")
        assert s2 is True
        assert r2 is None

    def test_line_after_context_window_is_not_suppressed(self) -> None:
        f = _DuplicateColWarningFilter()
        f.process("DataFrame columns are not unique")
        f.process("context line 1")
        f.process("context line 2")
        # Third line after the warning — no longer inside the context window
        suppress, replacement = f.process("Normal log output after the warning")
        assert suppress is False
        assert replacement is None

    def test_unrelated_lines_pass_through(self) -> None:
        f = _DuplicateColWarningFilter()
        suppress, replacement = f.process("Training Metrics (step 100) - CTC")
        assert suppress is False
        assert replacement is None

    def test_unrelated_lines_still_pass_after_warning_and_context(self) -> None:
        f = _DuplicateColWarningFilter()
        f.process("DataFrame columns are not unique")
        f.process("ctx 1")
        f.process("ctx 2")
        # Unrelated line from same module should still be visible
        suppress, _ = f.process("CRITICAL: mixture_parquet_storage failed with OSError")
        assert suppress is False

    def test_warned_property_reflects_state(self) -> None:
        f = _DuplicateColWarningFilter()
        assert f.warned is False
        f.process("DataFrame columns are not unique")
        assert f.warned is True


# ---------------------------------------------------------------------------
# check_prerequisites — Parquet schema validation
# ---------------------------------------------------------------------------


@pytest.fixture()
def clean_fairseq2_dir(tmp_path: Path) -> Path:
    """Minimal FAIRSEQ2_DIR layout that satisfies the basic checks before
    the Parquet schema test: a stats TSV and one corpus directory."""
    fs2_dir = tmp_path / "parquet" / "version=0"
    corpus_dir = fs2_dir / "corpus=read_aloud" / "split=train" / "language=dan_Latn"
    corpus_dir.mkdir(parents=True)
    (fs2_dir / "language_distribution_0.tsv").write_text("corpus\tcount\n")
    return fs2_dir


class TestCheckPrerequisitesParquetSchema:
    """Tests for the Parquet schema validation inside check_prerequisites()."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_pyarrow(self) -> None:
        pytest.importorskip("pyarrow", reason="pyarrow not installed (omni group)")

    def _make_parquet(self, directory: Path, columns: dict) -> Path:
        """Write a minimal Parquet file with the given column name→list mapping."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table({k: pa.array(v) for k, v in columns.items()})
        path = directory / "part-0.parquet"
        pq.write_table(table, path)
        return path

    def test_legacy_schema_raises_system_exit(
        self, clean_fairseq2_dir: Path, tmp_path: Path
    ) -> None:
        """check_prerequisites() must sys.exit(1) when a Parquet file contains
        in-file partition columns (corpus / split / language)."""
        corpus_dir = clean_fairseq2_dir / "corpus=read_aloud" / "split=train" / "language=dan_Latn"
        self._make_parquet(
            corpus_dir,
            {
                "text": ["hello"],
                "audio_bytes": [b"data"],
                "audio_size": [1],
                "corpus": ["read_aloud"],  # legacy partition column
                "split": ["train"],  # legacy partition column
                "language": ["dan_Latn"],  # legacy partition column
            },
        )

        config = tmp_path / "config.yaml"
        config.write_text("{}")

        from scripts.hpc import run_training

        with patch.object(run_training, "FAIRSEQ2_DIR", clean_fairseq2_dir):
            with pytest.raises(SystemExit) as exc_info:
                run_training.check_prerequisites(config)
        assert exc_info.value.code == 1

    def test_correct_schema_does_not_exit(
        self, clean_fairseq2_dir: Path, tmp_path: Path
    ) -> None:
        """check_prerequisites() must NOT sys.exit from the Parquet check when
        files only contain the required columns (text, audio_bytes, audio_size)."""
        from unittest.mock import MagicMock

        corpus_dir = clean_fairseq2_dir / "corpus=read_aloud" / "split=train" / "language=dan_Latn"
        self._make_parquet(
            corpus_dir,
            {
                "text": ["hello"],
                "audio_bytes": [b"data"],
                "audio_size": [1],
            },
        )

        config = tmp_path / "config.yaml"
        config.write_text("{}")

        from scripts.hpc import run_training

        # Mock torch and the fairseq2 recipe so the function reaches its end
        # without hitting hardware-dependent checks that always fail in CI.
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_workflows = MagicMock()

        with patch.object(run_training, "FAIRSEQ2_DIR", clean_fairseq2_dir):
            with patch.dict(
                "sys.modules",
                {
                    "torch": mock_torch,
                    "workflows": mock_workflows,
                    "workflows.recipes": mock_workflows,
                    "workflows.recipes.wav2vec2": mock_workflows,
                    "workflows.recipes.wav2vec2.asr": mock_workflows,
                },
            ):
                # Should complete without raising SystemExit
                run_training.check_prerequisites(config)


# ---------------------------------------------------------------------------
# _MetricParser (pre-existing tests, kept for completeness)
# ---------------------------------------------------------------------------


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

