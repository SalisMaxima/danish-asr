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


def test_should_log_fairseq2_line_filters_warning_continuation_line() -> None:
    assert (
        should_log_fairseq2_line("records = table.to_pandas(memory_pool=memory_pool, self_destruct=True).to_dict(")
        is False
    )
    assert (
        should_log_fairseq2_line("  records = table.to_pandas(memory_pool=memory_pool, self_destruct=True).to_dict(  ")
        is False
    )


def test_should_log_fairseq2_line_keeps_empty_or_whitespace_lines() -> None:
    assert should_log_fairseq2_line("") is True
    assert should_log_fairseq2_line("   ") is True
