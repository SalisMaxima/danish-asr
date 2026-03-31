"""Shared fairseq2 log filtering helpers for HPC wrappers."""

from __future__ import annotations

_MUTED_WARNING_SUBSTRINGS = ("UserWarning: DataFrame columns are not unique, some columns will be omitted.",)
_MUTED_WARNING_CONTINUATION_LINE = "records = table.to_pandas(memory_pool=memory_pool, self_destruct=True).to_dict("


def should_log_fairseq2_line(line: str) -> bool:
    """Return whether a fairseq2 subprocess log line should be emitted.

    Mutes known-noisy duplicate-column warnings from omnilingual-asr Parquet loading.

    Args:
        line: Raw log line emitted by the fairseq2 subprocess.

    Returns:
        True if the line should be logged; False if it should be muted.
    """
    if any(part in line for part in _MUTED_WARNING_SUBSTRINGS):
        return False

    # The warning is emitted on two lines; the second line is source-code context.
    # Match it exactly (after trimming) to avoid muting unrelated lines that might
    # only contain this text as a substring.
    stripped = line.strip()
    return stripped != _MUTED_WARNING_CONTINUATION_LINE
