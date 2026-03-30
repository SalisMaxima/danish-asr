"""Tests for fairseq2 warning log filtering in HPC wrappers."""

from scripts.hpc.fairseq2_logging import should_log_fairseq2_line


def test_run_eval_filters_duplicate_dataframe_warning_lines() -> None:
    assert (
        should_log_fairseq2_line(
            "/path/mixture_parquet_storage.py:438: UserWarning: DataFrame columns are not unique, some columns will be omitted."
        )
        is False
    )
    assert (
        should_log_fairseq2_line("records = table.to_pandas(memory_pool=memory_pool, self_destruct=True).to_dict(")
        is False
    )
    assert should_log_fairseq2_line("Evaluation completed successfully in 5.1 min") is True


def test_sweep_wrapper_filters_duplicate_dataframe_warning_lines() -> None:
    assert (
        should_log_fairseq2_line(
            "/path/mixture_parquet_storage.py:438: UserWarning: DataFrame columns are not unique, some columns will be omitted."
        )
        is False
    )
    assert (
        should_log_fairseq2_line("records = table.to_pandas(memory_pool=memory_pool, self_destruct=True).to_dict(")
        is False
    )
    assert should_log_fairseq2_line("Training Metrics (step 100) - CTC") is True
