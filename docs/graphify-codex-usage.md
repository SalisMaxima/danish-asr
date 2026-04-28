# Graphify and Codex Usage

This repository has a graphify knowledge graph in `graphify-out/`.

The graph is meant to help Codex answer codebase and architecture questions without rereading the whole repository from scratch. It stores extracted code structure, document concepts, inferred relationships, community labels, and an audit report.

## Main Outputs

Only `GRAPH_REPORT.md` is committed to the repo. The other artifacts are generated locally by `graphify update .` and are gitignored:

- `graphify-out/GRAPH_REPORT.md` (committed) — readable report with god nodes, communities, surprising connections, and suggested questions.
- `graphify-out/graph.json` (local, gitignored) — raw graph data backing the `graphify query/path/explain` commands.
- `graphify-out/graph.html` (local, gitignored) — interactive graph that can be opened in a browser.

On a fresh checkout, run `graphify update .` once to generate `graph.json` and `graph.html` before using the query commands. The temporary `.graphify_*`, `manifest.json`, and `cost.json` files in `graphify-out/` are working artifacts from graph construction; they are useful for debugging a run but are not committed.

## How Codex Should Use It

Future Codex sessions should read `AGENTS.md` first. It tells Codex to check the graph before answering architecture or codebase questions.

For broad architecture questions, Codex should start with:

```bash
graphify query "how does CTC LM decode connect to HPC evaluation?"
```

For a direct relationship between two concepts:

```bash
graphify path "CoRalDataset" "ASRLitModel"
```

For a single concept and its neighbors:

```bash
graphify explain "PreprocessedCoRalDataset"
```

These commands traverse `graphify-out/graph.json`, including extracted and inferred edges, so they are often a better first pass than raw grep for cross-module questions.

## Keeping the Graph Fresh

After code changes, update the graph with:

```bash
graphify update .
```

That update is AST-only for code changes and should not require LLM/API cost. If documentation, papers, or images changed, run a fuller graphify update so the semantic layer is refreshed too.

## Local Codex Integration

This repo includes:

- `AGENTS.md`, which tells Codex how to use graphify.
- `.codex/hooks.json`, which reminds Codex to read `graphify-out/GRAPH_REPORT.md` before shell/tool use, and to run `graphify update .` if the local `graph.json` is missing.

After adding or changing these files, reload the Codex session in VS Code so the local instructions are picked up.
