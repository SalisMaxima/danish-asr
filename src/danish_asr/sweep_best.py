"""Utilities for working with W&B sweeps.

Provides a CLI to identify the best run in a sweep.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
import wandb
from omegaconf import OmegaConf


def _get_summary_metric(run: Any, metric: str) -> float | None:
    summary = getattr(run, "summary", None)
    if not summary:
        return None
    value = summary.get(metric)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_default_entity_project() -> tuple[str | None, str | None]:
    project_root = Path(__file__).resolve().parent.parent.parent
    cfg_path = project_root / "configs" / "config.yaml"
    if not cfg_path.exists():
        return None, None
    cfg = OmegaConf.load(cfg_path)
    wandb_cfg = cfg.get("wandb", {})
    return wandb_cfg.get("entity"), wandb_cfg.get("project")


def _normalize_sweep_id(sweep_id: str, entity: str | None, project: str | None) -> str:
    if sweep_id.count("/") >= 2:
        return sweep_id
    default_entity, default_project = _get_default_entity_project()
    entity = entity or default_entity
    project = project or default_project
    if not entity or not project:
        raise typer.BadParameter("Provide a full sweep id ENTITY/PROJECT/SWEEP_ID")
    return f"{entity}/{project}/{sweep_id}"


def _is_better_value(value: float, best_value: float | None, goal: str) -> bool:
    """Return True if *value* improves on *best_value* for the given goal."""
    if best_value is None:
        return True
    if goal == "minimize":
        return value < best_value
    return value > best_value


def _find_best_run(runs, metric: str, goal: str) -> tuple[Any, float] | None:
    """Return (run, metric_value) for the best finished run, or None."""
    best_run = None
    best_value: float | None = None
    for run in runs:
        if getattr(run, "state", "") != "finished":
            continue
        value = _get_summary_metric(run, metric)
        if value is None:
            continue
        if _is_better_value(value, best_value, goal):
            best_run, best_value = run, value
    if best_run is None or best_value is None:
        return None
    return best_run, best_value


def _build_result(
    normalized: str, metric: str, goal: str, best_run: Any, best_value: float, include_config: bool
) -> dict[str, Any]:
    """Build the JSON-serialisable result dict."""
    result: dict[str, Any] = {
        "sweep_id": normalized,
        "metric": metric,
        "goal": goal,
        "best": {"run_id": best_run.id, "run_name": best_run.name, "url": best_run.url, "value": best_value},
    }
    if include_config:
        cfg = dict(getattr(best_run, "config", {}) or {})
        result["best"]["config"] = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return result


def best(
    sweep_id: str = typer.Argument(..., help="Sweep id: ENTITY/PROJECT/SWEEP_ID"),
    entity: str = typer.Option(None, help="W&B entity"),
    project: str = typer.Option(None, help="W&B project"),
    metric: str = typer.Option("val_acc", help="Metric to optimize"),
    goal: str = typer.Option("maximize", help="maximize | minimize"),
    include_config: bool = typer.Option(True, help="Print run config"),
) -> None:
    """Print the best run for a given sweep."""
    normalized = _normalize_sweep_id(sweep_id, entity=entity, project=project)
    api = wandb.Api()
    try:
        sweep = api.sweep(normalized)
    except Exception as exc:
        raise typer.BadParameter(f"Could not find sweep '{normalized}'.") from exc

    found = _find_best_run(sweep.runs, metric, goal)
    if found is None:
        raise typer.BadParameter("No finished runs with the requested metric found.")

    best_run, best_value = found
    result = _build_result(normalized, metric, goal, best_run, best_value, include_config)
    print(json.dumps(result, indent=2))


def main() -> None:
    typer.run(best)


if __name__ == "__main__":
    main()
