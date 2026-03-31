"""Tests for fairseq2 warning log filtering in HPC wrappers."""

from scripts.hpc.fairseq2_logging import should_log_fairseq2_line


def test_should_log_fairseq2_line_allows_normal_output() -> None:
    assert should_log_fairseq2_line("Evaluation completed successfully in 5.1 min") is True


def test_should_log_fairseq2_line_filters_duplicate_column_warning() -> None:
    assert (
        should_log_fairseq2_line(
            "/path/mixture_parquet_storage.py:438: UserWarning: DataFrame columns are not unique, some columns will be omitted."
        )
        is False
    )


def test_should_log_fairseq2_line_filters_actual_hpc_warning_line() -> None:
    assert (
        should_log_fairseq2_line(
            "/zhome/d8/4/155560/danish_asr/.venv/lib/python3.12/site-packages/omnilingual_asr/datasets/storage/mixture_parquet_storage.py:438: UserWarning: DataFrame columns are not unique, some columns will be omitted."
        )
        is False
    )


def test_should_log_fairseq2_line_keeps_to_pandas_source_line() -> None:
    # This line appears in tracebacks when to_pandas raises — must NOT be suppressed.
    assert (
        should_log_fairseq2_line("  records = table.to_pandas(memory_pool=memory_pool, self_destruct=True).to_dict(")
        is True
    )


def test_should_log_fairseq2_line_keeps_empty_or_whitespace_lines() -> None:
    assert should_log_fairseq2_line("") is True
    assert should_log_fairseq2_line("   ") is True
