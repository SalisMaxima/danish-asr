"""Shared fairseq2 log filtering helpers for HPC wrappers."""

from __future__ import annotations

_MUTED_WARNING_SUBSTRINGS = ("UserWarning: DataFrame columns are not unique, some columns will be omitted.",)


def should_log_fairseq2_line(line: str) -> bool:
    """Return whether a fairseq2 subprocess log line should be emitted.

    Mutes known-noisy duplicate-column warnings from omnilingual-asr Parquet loading.

    Args:
        line: Raw log line emitted by the fairseq2 subprocess.

    Returns:
        True if the line should be logged; False if it should be muted.
    """
    return not any(part in line for part in _MUTED_WARNING_SUBSTRINGS)
