# Graphify and Codex Usage

This repository has a graphify knowledge graph in `graphify-out/`.

The graph is meant to help Codex answer codebase and architecture questions without rereading the whole repository from scratch. It stores extracted code structure, document concepts, inferred relationships, community labels, and an audit report.

## Main Outputs

- `graphify-out/graph.json` - raw graph data for queries and traversal.
- `graphify-out/GRAPH_REPORT.md` - readable report with god nodes, communities, surprising connections, and suggested questions.
- `graphify-out/graph.html` - interactive graph that can be opened in a browser.

The temporary `.graphify_*` files in `graphify-out/` are working artifacts from graph construction. They are useful for debugging a run, but the three files above are the important persistent outputs.

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
- `.codex/hooks.json`, which reminds Codex that `graphify-out/graph.json` exists before shell/tool use.

After adding or changing these files, reload the Codex session in VS Code so the local instructions are picked up.
