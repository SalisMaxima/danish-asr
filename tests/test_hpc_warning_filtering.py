"""Tests for fairseq2 warning log filtering in HPC wrappers."""

from scripts.hpc.run_eval import _should_log_fairseq2_line as eval_should_log
from scripts.hpc.sweep_agent_wrapper import _should_log_fairseq2_line as sweep_should_log


def test_run_eval_filters_duplicate_dataframe_warning_lines() -> None:
    assert (
        eval_should_log(
            "/path/mixture_parquet_storage.py:438: UserWarning: DataFrame columns are not unique, some columns will be omitted."
        )
        is False
    )
    assert (
        eval_should_log("records = table.to_pandas(memory_pool=memory_pool, self_destruct=True).to_dict(") is False
    )
    assert eval_should_log("Evaluation completed successfully in 5.1 min") is True


def test_sweep_wrapper_filters_duplicate_dataframe_warning_lines() -> None:
    assert (
        sweep_should_log(
            "/path/mixture_parquet_storage.py:438: UserWarning: DataFrame columns are not unique, some columns will be omitted."
        )
        is False
    )
    assert (
        sweep_should_log("records = table.to_pandas(memory_pool=memory_pool, self_destruct=True).to_dict(") is False
    )
    assert sweep_should_log("Training Metrics (step 100) - CTC") is True
