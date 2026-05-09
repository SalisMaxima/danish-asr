from __future__ import annotations

import pytest

from scripts.render_results_table import (
    _cell_class,
    _extract_table,
    _is_separator_row,
    _render_inline_markdown,
)

MD = "## My Section\n\n| A | B |\n|---|---|\n| `code` | **bold** |\n"


def test_extract_table_parses_header_and_rows() -> None:
    header, rows = _extract_table(MD, "My Section")
    assert header == ["A", "B"]
    assert rows == [["`code`", "**bold**"]]


def test_extract_table_stops_at_next_non_pipe_line() -> None:
    md = "## Sec\n\n| X |\n|---|\n| v1 |\n\nsome text\n| Y |\n|---|\n"
    _, rows = _extract_table(md, "Sec")
    assert rows == [["v1"]]


def test_extract_table_raises_on_missing_section() -> None:
    with pytest.raises(ValueError, match="Could not find section"):
        _extract_table(MD, "Not Present")


def test_extract_table_raises_on_section_with_no_table() -> None:
    with pytest.raises(ValueError, match="Could not find a markdown table"):
        _extract_table("## Empty Section\n\nsome text\n", "Empty Section")


def test_extract_table_raises_on_malformed_separator() -> None:
    with pytest.raises(ValueError, match="Malformed markdown table"):
        _extract_table("## Sec\n\n| A |\n| not-sep |\n", "Sec")


def test_render_inline_markdown_handles_bold_and_code() -> None:
    assert "<strong>x</strong>" in _render_inline_markdown("**x**")
    assert "<code>y</code>" in _render_inline_markdown("`y`")


def test_render_inline_markdown_escapes_html() -> None:
    result = _render_inline_markdown("<script>")
    assert "<script>" not in result
    assert "&lt;" in result


def test_is_separator_row_accepts_colon_variants() -> None:
    assert _is_separator_row("|:---|:---:|---:|")


def test_is_separator_row_rejects_non_separator() -> None:
    assert not _is_separator_row("| some | data |")


def test_cell_class_wer_columns() -> None:
    assert _cell_class("Test WER") == "metric-cell"
    assert _cell_class("Read-aloud WER") == "metric-cell"
    assert _cell_class("Conversation WER") == "metric-cell"


def test_cell_class_cer_columns() -> None:
    assert _cell_class("Test CER") == "metric-cell"
    assert _cell_class("Read-aloud CER") == "metric-cell"
    assert _cell_class("Conversation CER") == "metric-cell"


def test_cell_class_path_columns() -> None:
    assert _cell_class("Config") == "path-cell"
    assert _cell_class("Script") == "path-cell"


def test_cell_class_number_columns() -> None:
    assert _cell_class("Steps") == "number-cell"
    assert _cell_class("Params") == "number-cell"


def test_cell_class_unknown_returns_empty() -> None:
    assert _cell_class("Some Random Column") == ""
