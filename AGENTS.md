## graphify

This project tracks graphify artifacts in `graphify-out/`. Only `GRAPH_REPORT.md` is committed; the graph itself (`graph.json`) is generated locally — run `graphify update .` once on a fresh checkout if it's missing.

Rules:
- Before answering architecture or codebase questions, read `graphify-out/GRAPH_REPORT.md` for god nodes and community structure
- If `graphify-out/wiki/index.md` exists, navigate it instead of reading raw files
- Before using `graphify query "<question>"`, `graphify path "<A>" "<B>"`, or `graphify explain "<concept>"`, ensure `graphify-out/graph.json` exists; on a fresh checkout, generate it with `graphify update .`
- For cross-module "how does X relate to Y" questions, prefer those graphify commands over grep once the graph has been generated — they traverse the graph's EXTRACTED + INFERRED edges instead of scanning files
- After modifying code files in this session, run `graphify update .` to keep the graph current (AST-only, no API cost)
