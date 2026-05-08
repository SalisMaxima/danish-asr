"""Render a results table from docs/evaluation-results.md to HTML or PNG.

Examples:
    uv run python scripts/render_results_table.py \
        --section "Combined Test Results" \
        --output outputs/tables/combined-test-results.png

    uv run python scripts/render_results_table.py \
        --section "Split-Tagged Results" \
        --output outputs/tables/split-tagged-results.html
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "docs" / "evaluation-results.md"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "table"


def _split_markdown_row(line: str) -> list[str]:
    return [part.strip() for part in line.strip().strip("|").split("|")]


def _is_separator_row(line: str) -> bool:
    cells = _split_markdown_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _render_inline_markdown(value: str) -> str:
    escaped = html.escape(value)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    return re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)


def _extract_table(markdown: str, section: str) -> tuple[list[str], list[list[str]]]:
    section_re = re.compile(rf"^##\s+{re.escape(section)}\s*$", re.MULTILINE)
    match = section_re.search(markdown)
    if not match:
        raise ValueError(f"Could not find section: {section!r}")

    lines = markdown[match.end() :].splitlines()
    table_start = next((idx for idx, line in enumerate(lines) if line.startswith("|")), None)
    if table_start is None:
        raise ValueError(f"Could not find a markdown table in section: {section!r}")

    table_lines: list[str] = []
    for line in lines[table_start:]:
        if not line.startswith("|"):
            break
        table_lines.append(line)

    if len(table_lines) < 2 or not _is_separator_row(table_lines[1]):
        raise ValueError(f"Malformed markdown table in section: {section!r}")

    header = _split_markdown_row(table_lines[0])
    rows = [_split_markdown_row(line) for line in table_lines[2:]]
    return header, rows


def _cell_class(header: str) -> str:
    normalized = header.lower()
    if "config" in normalized or "script" in normalized:
        return "path-cell"
    if normalized in {"test wer", "read-aloud wer", "conversation wer"}:
        return "metric-cell"
    if normalized in {"steps", "params"}:
        return "number-cell"
    return ""


def _build_html(section: str, header: list[str], rows: list[list[str]]) -> str:
    column_count = len(header)
    body_rows = []
    for row in rows:
        cells = []
        for idx, value in enumerate(row):
            header_value = header[idx] if idx < len(header) else ""
            class_name = _cell_class(header_value)
            class_attr = f' class="{class_name}"' if class_name else ""
            cells.append(f"<td{class_attr}>{_render_inline_markdown(value)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    header_html = "".join(f"<th>{html.escape(value)}</th>" for value in header)
    subtitle = "Fairseq2 WER result provenance: training recipe, evaluation recipe, metric, and W&B run."
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(section)}</title>
  <style>
    :root {{
      --ink: #172033;
      --muted: #5b6578;
      --line: #d9dfeb;
      --panel: #ffffff;
      --header: #eef4ff;
      --header-ink: #10254f;
      --stripe: #f8fafd;
      --accent: #1d6fa5;
      --metric-bg: #e9f7ef;
      --metric-ink: #116339;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      padding: 32px;
      background: #f2f5f9;
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}

    .sheet {{
      display: inline-block;
      max-width: none;
      padding: 26px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 18px 45px rgba(21, 31, 51, 0.12);
    }}

    h1 {{
      margin: 0;
      color: var(--ink);
      font-size: 28px;
      font-weight: 760;
      letter-spacing: 0;
    }}

    .subtitle {{
      margin: 8px 0 22px;
      color: var(--muted);
      font-size: 14px;
    }}

    table {{
      width: max-content;
      min-width: 1180px;
      max-width: {max(1280, column_count * 210)}px;
      border-collapse: separate;
      border-spacing: 0;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 8px;
      font-size: 13px;
      line-height: 1.35;
    }}

    thead th {{
      position: sticky;
      top: 0;
      padding: 11px 12px;
      background: var(--header);
      color: var(--header-ink);
      border-bottom: 1px solid #c8d4ea;
      border-right: 1px solid #d8e1f2;
      text-align: left;
      font-size: 12px;
      font-weight: 750;
      text-transform: uppercase;
      white-space: nowrap;
    }}

    thead th:last-child,
    tbody td:last-child {{
      border-right: 0;
    }}

    tbody tr:nth-child(even) {{
      background: var(--stripe);
    }}

    tbody td {{
      max-width: 245px;
      padding: 10px 12px;
      border-bottom: 1px solid #e4e9f2;
      border-right: 1px solid #ebeff6;
      vertical-align: top;
      overflow-wrap: anywhere;
    }}

    tbody tr:last-child td {{
      border-bottom: 0;
    }}

    code {{
      color: #24324b;
      font-family: "SFMono-Regular", "Cascadia Code", "Liberation Mono", Menlo, monospace;
      font-size: 0.92em;
      background: #eef1f6;
      border-radius: 4px;
      padding: 1px 4px;
    }}

    .path-cell {{
      min-width: 210px;
      max-width: 310px;
      color: #2c3850;
      font-family: "SFMono-Regular", "Cascadia Code", "Liberation Mono", Menlo, monospace;
      font-size: 12px;
    }}

    .metric-cell {{
      min-width: 98px;
      color: var(--metric-ink);
      background: var(--metric-bg);
      font-weight: 800;
      text-align: right;
      white-space: nowrap;
    }}

    .number-cell {{
      text-align: right;
      white-space: nowrap;
    }}

    strong {{
      color: var(--metric-ink);
      font-weight: 850;
    }}

    .footer {{
      margin-top: 14px;
      color: var(--muted);
      font-size: 11px;
    }}
  </style>
</head>
<body>
  <main class="sheet">
    <h1>{html.escape(section)}</h1>
    <p class="subtitle">{subtitle}</p>
    <table>
      <thead><tr>{header_html}</tr></thead>
      <tbody>
        {"".join(body_rows)}
      </tbody>
    </table>
    <div class="footer">Generated from docs/evaluation-results.md</div>
  </main>
</body>
</html>
"""


def _find_chrome() -> str:
    for candidate in ("google-chrome", "chromium", "chromium-browser"):
        path = shutil.which(candidate)
        if path:
            return path
    raise RuntimeError("Could not find google-chrome, chromium, or chromium-browser for PNG export.")


def _render_png(html_path: Path, output_path: Path, width: int, height: int, scale: float) -> None:
    chrome = _find_chrome()
    command = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        f"--force-device-scale-factor={scale}",
        f"--window-size={width},{height}",
        f"--screenshot={output_path}",
        html_path.as_uri(),
    ]
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Markdown file containing the table")
    parser.add_argument("--section", default="Combined Test Results", help="Level-2 section containing the table")
    parser.add_argument("--output", type=Path, required=True, help="Output .html or .png path")
    parser.add_argument("--width", type=int, default=2260, help="PNG viewport width")
    parser.add_argument("--height", type=int, default=900, help="PNG viewport height")
    parser.add_argument("--scale", type=float, default=1.0, help="PNG device scale factor")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    markdown = args.input.read_text(encoding="utf-8")
    header, rows = _extract_table(markdown, args.section)
    rendered = _build_html(args.section, header, rows)

    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".html":
        output_path.write_text(rendered, encoding="utf-8")
        print(f"Wrote {output_path}")
        return

    if output_path.suffix.lower() != ".png":
        raise ValueError("Output path must end in .html or .png")

    with tempfile.TemporaryDirectory(prefix="results-table-") as tmp_dir:
        html_path = Path(tmp_dir) / f"{_slugify(args.section)}.html"
        html_path.write_text(rendered, encoding="utf-8")
        _render_png(html_path, output_path, width=args.width, height=args.height, scale=args.scale)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
