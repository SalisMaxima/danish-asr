"""Shared fairseq2 log filtering helpers for HPC wrappers."""

from __future__ import annotations

_MUTED_WARNING_SUBSTRINGS = (
    "UserWarning: DataFrame columns are not unique, some columns will be omitted.",
)
_MUTED_WARNING_CONTINUATION_LINE = "records = table.to_pandas(memory_pool=memory_pool, self_destruct=True).to_dict("


def should_log_fairseq2_line(line: str) -> bool:
    """Return whether a fairseq2 subprocess log line should be emitted.

    Mutes known-noisy duplicate-column warnings from omnilingual-asr Parquet loading.
    """
    stripped = line.strip()
    if any(part in stripped for part in _MUTED_WARNING_SUBSTRINGS):
        return False
    return stripped != _MUTED_WARNING_CONTINUATION_LINE
