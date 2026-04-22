"""Audit W&B storage usage by artifact versions, collections, and run files."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import wandb
from omegaconf import OmegaConf


@dataclass(frozen=True)
class ArtifactVersionRow:
    project: str
    artifact_type: str
    collection: str
    version: str
    size: int
    aliases: tuple[str, ...]
    created_at: str | None


@dataclass(frozen=True)
class RunFileRow:
    project: str
    run_id: str
    run_name: str
    file_name: str
    size: int


def _get_default_entity_project() -> tuple[str | None, str | None]:
    project_root = Path(__file__).resolve().parent.parent.parent
    cfg_path = project_root / "configs" / "config.yaml"
    if not cfg_path.exists():
        return None, None
    cfg = OmegaConf.load(cfg_path)
    wandb_cfg = cfg.get("wandb", {})
    return wandb_cfg.get("entity"), wandb_cfg.get("project")


def _resolve_entity(entity: str | None) -> str:
    default_entity, _ = _get_default_entity_project()
    resolved = entity or os.environ.get("WANDB_ENTITY") or default_entity
    if not resolved:
        raise ValueError("Provide --entity or set WANDB_ENTITY.")
    return resolved


def _resolve_projects(api: wandb.Api, entity: str, project: str | None, all_projects: bool) -> list[str]:
    _, default_project = _get_default_entity_project()
    if all_projects:
        if project:
            raise ValueError("Use either --project or --all-projects, not both.")
        return [proj.name for proj in api.projects(entity=entity)]

    resolved_project = project or os.environ.get("WANDB_PROJECT") or default_project
    if not resolved_project:
        raise ValueError("Provide --project or pass --all-projects.")
    return [resolved_project]


def _safe_size(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_aliases(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(str(item) for item in value)


def _bytes_to_gib(num_bytes: int) -> float:
    return num_bytes / (1024**3)


def _emit_warning(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr)


def _parse_created_at(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def collect_artifact_rows(api: wandb.Api, entity: str, projects: list[str]) -> list[ArtifactVersionRow]:
    rows: list[ArtifactVersionRow] = []

    for project in projects:
        project_path = f"{entity}/{project}"
        try:
            artifact_types = list(api.artifact_types(project=project_path))
        except Exception as exc:
            _emit_warning(f"failed to list artifact types for {project_path}: {exc}")
            continue

        for artifact_type in artifact_types:
            type_name = getattr(artifact_type, "name", None)
            if not type_name:
                continue

            try:
                collections = list(api.artifact_collections(project_name=project, type_name=type_name))
            except Exception as exc:
                _emit_warning(f"failed to list collections for {project_path} type {type_name}: {exc}")
                continue

            for collection in collections:
                collection_name = getattr(collection, "name", None)
                if not collection_name:
                    continue

                collection_path = f"{entity}/{project}/{collection_name}"
                try:
                    artifacts = list(api.artifacts(type_name=type_name, name=collection_path))
                except Exception as exc:
                    _emit_warning(f"failed to list artifacts for {collection_path}: {exc}")
                    continue

                for artifact in artifacts:
                    rows.append(
                        ArtifactVersionRow(
                            project=project,
                            artifact_type=type_name,
                            collection=collection_name,
                            version=str(getattr(artifact, "name", collection_name)),
                            size=_safe_size(getattr(artifact, "size", 0)),
                            aliases=_safe_aliases(getattr(artifact, "aliases", ()) or ()),
                            created_at=getattr(artifact, "created_at", None),
                        )
                    )

    return rows


def collect_run_file_rows(api: wandb.Api, entity: str, projects: list[str]) -> list[RunFileRow]:
    rows: list[RunFileRow] = []

    for project in projects:
        project_path = f"{entity}/{project}"
        try:
            runs = api.runs(project_path)
        except Exception as exc:
            _emit_warning(f"failed to list runs for {project_path}: {exc}")
            continue

        for run in runs:
            try:
                files = list(run.files())
            except Exception as exc:
                _emit_warning(f"failed to list files for run {project_path}/{run.id}: {exc}")
                continue

            for file_obj in files:
                size = _safe_size(getattr(file_obj, "size", 0))
                if size <= 0:
                    continue
                rows.append(
                    RunFileRow(
                        project=project,
                        run_id=str(run.id),
                        run_name=str(getattr(run, "name", run.id)),
                        file_name=str(getattr(file_obj, "name", "<unknown>")),
                        size=size,
                    )
                )

    return rows


def filter_artifact_rows(
    rows: list[ArtifactVersionRow],
    *,
    collection: str | None,
    artifact_type: str | None,
    unaliased_only: bool,
    min_size_gib: float,
    older_than_days: int | None,
) -> list[ArtifactVersionRow]:
    min_size_bytes = int(min_size_gib * (1024**3))
    cutoff: datetime | None = None
    if older_than_days is not None:
        cutoff = datetime.now(UTC) - timedelta(days=older_than_days)

    filtered: list[ArtifactVersionRow] = []
    for row in rows:
        if collection and row.collection != collection:
            continue
        if artifact_type and row.artifact_type != artifact_type:
            continue
        if unaliased_only and row.aliases:
            continue
        if row.size < min_size_bytes:
            continue
        if cutoff is not None:
            created_at = _parse_created_at(row.created_at)
            if created_at is None or created_at > cutoff:
                continue
        filtered.append(row)

    return sorted(filtered, key=lambda row: row.size, reverse=True)


def delete_artifact_rows(
    api: wandb.Api,
    entity: str,
    rows: list[ArtifactVersionRow],
    *,
    allow_delete_aliased: bool,
) -> tuple[int, int]:
    deleted_count = 0
    deleted_bytes = 0

    for row in rows:
        if row.aliases and not allow_delete_aliased:
            _emit_warning(f"skipping aliased artifact {row.version} aliases={list(row.aliases)}")
            continue

        full_name = f"{entity}/{row.project}/{row.version}"
        artifact = api.artifact(full_name, type=row.artifact_type)
        artifact.delete(delete_aliases=allow_delete_aliased)
        deleted_count += 1
        deleted_bytes += row.size
        print(f"Deleted {full_name} ({_bytes_to_gib(row.size):.2f} GiB)")

    return deleted_count, deleted_bytes


def build_artifact_report(rows: list[ArtifactVersionRow]) -> dict[str, Any]:
    project_totals: dict[str, int] = defaultdict(int)
    collection_totals: dict[tuple[str, str, str], int] = defaultdict(int)

    for row in rows:
        project_totals[row.project] += row.size
        collection_totals[(row.project, row.artifact_type, row.collection)] += row.size

    sorted_versions = sorted(rows, key=lambda row: row.size, reverse=True)
    sorted_projects = sorted(project_totals.items(), key=lambda item: item[1], reverse=True)
    sorted_collections = sorted(collection_totals.items(), key=lambda item: item[1], reverse=True)

    return {
        "artifact_versions": sorted_versions,
        "project_totals": sorted_projects,
        "collection_totals": sorted_collections,
    }


def build_run_file_report(rows: list[RunFileRow]) -> dict[str, Any]:
    project_totals: dict[str, int] = defaultdict(int)

    for row in rows:
        project_totals[row.project] += row.size

    sorted_files = sorted(rows, key=lambda row: row.size, reverse=True)
    sorted_projects = sorted(project_totals.items(), key=lambda item: item[1], reverse=True)

    return {
        "run_files": sorted_files,
        "project_totals": sorted_projects,
    }


def _json_ready_artifact_report(report: dict[str, Any], top_n: int) -> dict[str, Any]:
    return {
        "artifact_versions": [asdict(row) for row in report["artifact_versions"][:top_n]],
        "project_totals": [
            {"project": project, "size_bytes": size, "size_gib": _bytes_to_gib(size)}
            for project, size in report["project_totals"][:top_n]
        ],
        "collection_totals": [
            {
                "project": project,
                "artifact_type": artifact_type,
                "collection": collection,
                "size_bytes": size,
                "size_gib": _bytes_to_gib(size),
            }
            for (project, artifact_type, collection), size in report["collection_totals"][:top_n]
        ],
    }


def _json_ready_run_file_report(report: dict[str, Any], top_n: int) -> dict[str, Any]:
    return {
        "run_files": [asdict(row) for row in report["run_files"][:top_n]],
        "project_totals": [
            {"project": project, "size_bytes": size, "size_gib": _bytes_to_gib(size)}
            for project, size in report["project_totals"][:top_n]
        ],
    }


def _print_artifact_report(report: dict[str, Any], top_n: int) -> None:
    print("\nTop artifact versions\n")
    for row in report["artifact_versions"][:top_n]:
        print(
            f"{_bytes_to_gib(row.size):8.2f} GiB  "
            f"{row.project:<20}  "
            f"{row.artifact_type:<10}  "
            f"{row.version}  aliases={list(row.aliases)}"
        )

    print("\nTop collections by summed version size\n")
    for (project, artifact_type, collection), size in report["collection_totals"][:top_n]:
        print(f"{_bytes_to_gib(size):8.2f} GiB  {project:<20}  {artifact_type:<10}  {collection}")

    print("\nTop projects by summed artifact version size\n")
    for project, size in report["project_totals"][:top_n]:
        print(f"{_bytes_to_gib(size):8.2f} GiB  {project}")


def _print_run_file_report(report: dict[str, Any], top_n: int) -> None:
    print("\nTop run files\n")
    for row in report["run_files"][:top_n]:
        print(f"{_bytes_to_gib(row.size):8.2f} GiB  {row.project:<20}  {row.run_id:<12}  {row.file_name}")

    print("\nTop projects by summed run-file size\n")
    for project, size in report["project_totals"][:top_n]:
        print(f"{_bytes_to_gib(size):8.2f} GiB  {project}")


def _print_candidate_report(rows: list[ArtifactVersionRow], top_n: int) -> None:
    total_bytes = sum(row.size for row in rows)
    print("\nCleanup candidates\n")
    print(f"Candidates: {len(rows)}")
    print(f"Candidate summed size: {_bytes_to_gib(total_bytes):.2f} GiB")
    for row in rows[:top_n]:
        print(
            f"{_bytes_to_gib(row.size):8.2f} GiB  "
            f"{row.project:<20}  "
            f"{row.artifact_type:<10}  "
            f"{row.version}  aliases={list(row.aliases)}  created_at={row.created_at}"
        )


def audit(
    entity: str | None,
    project: str | None,
    all_projects: bool,
    include_run_files: bool,
    top_n: int,
    json_output: Path | None,
    collection: str | None,
    artifact_type_filter: str | None,
    unaliased_only: bool,
    min_size_gib: float,
    older_than_days: int | None,
    delete: bool,
    delete_aliased: bool,
    max_delete: int | None,
) -> None:
    """Audit W&B cloud storage usage for artifacts and optionally run files."""
    resolved_entity = _resolve_entity(entity)
    api = wandb.Api()
    projects = _resolve_projects(api, resolved_entity, project, all_projects)

    print(f"Auditing W&B storage for entity={resolved_entity} projects={projects}")

    artifact_rows = collect_artifact_rows(api, resolved_entity, projects)
    artifact_report = build_artifact_report(artifact_rows)
    _print_artifact_report(artifact_report, top_n=top_n)

    output_payload: dict[str, Any] = {
        "entity": resolved_entity,
        "projects": projects,
        "note": (
            "Artifact version size is useful for triage but can overstate reclaimable cloud storage "
            "because W&B deduplicates unchanged files across versions."
        ),
        "artifacts": _json_ready_artifact_report(artifact_report, top_n=top_n),
    }

    candidate_rows = filter_artifact_rows(
        artifact_rows,
        collection=collection,
        artifact_type=artifact_type_filter,
        unaliased_only=unaliased_only,
        min_size_gib=min_size_gib,
        older_than_days=older_than_days,
    )
    if any(
        [
            collection is not None,
            artifact_type_filter is not None,
            unaliased_only,
            min_size_gib > 0,
            older_than_days is not None,
        ]
    ):
        _print_candidate_report(candidate_rows, top_n=top_n)
        output_payload["cleanup_candidates"] = [asdict(row) for row in candidate_rows[:top_n]]

    if include_run_files:
        run_file_rows = collect_run_file_rows(api, resolved_entity, projects)
        run_file_report = build_run_file_report(run_file_rows)
        _print_run_file_report(run_file_report, top_n=top_n)
        output_payload["run_files"] = _json_ready_run_file_report(run_file_report, top_n=top_n)

    if delete:
        rows_to_delete = candidate_rows if max_delete is None else candidate_rows[:max_delete]
        if not rows_to_delete:
            print("\nNo cleanup candidates matched the current filters, so nothing was deleted.")
        else:
            deleted_count, deleted_bytes = delete_artifact_rows(
                api,
                resolved_entity,
                rows_to_delete,
                allow_delete_aliased=delete_aliased,
            )
            output_payload["deleted"] = {
                "count": deleted_count,
                "size_bytes": deleted_bytes,
                "size_gib": _bytes_to_gib(deleted_bytes),
            }
            print(f"\nDeleted {deleted_count} artifacts totaling {_bytes_to_gib(deleted_bytes):.2f} GiB")

    if json_output is not None:
        json_output.write_text(json.dumps(output_payload, indent=2))
        print(f"\nWrote JSON report to {json_output}")

    print("\nNote: deleting artifacts is soft-delete first; storage usage may not drop immediately after cleanup.")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect W&B storage usage for a project or an entire entity.")
    parser.add_argument("--entity", default=None, help="W&B entity or team name.")
    parser.add_argument("--project", default=None, help="Single W&B project name.")
    parser.add_argument(
        "--all-projects", action="store_true", help="Audit every project in the entity instead of one project."
    )
    parser.add_argument(
        "--include-run-files", action="store_true", help="Also inspect uploaded run files, not just artifacts."
    )
    parser.add_argument("--top-n", type=int, default=20, help="Number of rows to print in each section.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional path to write the report as JSON.")
    parser.add_argument("--collection", default=None, help="Optional artifact collection to focus on.")
    parser.add_argument("--artifact-type-filter", default=None, help="Optional artifact type filter, e.g. model.")
    parser.add_argument(
        "--unaliased-only", action="store_true", help="Only keep artifact versions that have no aliases."
    )
    parser.add_argument("--min-size-gib", type=float, default=0.0, help="Only keep artifacts at or above this size.")
    parser.add_argument(
        "--older-than-days", type=int, default=None, help="Only keep artifacts created at least this many days ago."
    )
    parser.add_argument(
        "--delete", action="store_true", help="Delete the filtered artifact candidates. Dry-run is the default."
    )
    parser.add_argument(
        "--delete-aliased",
        action="store_true",
        help="Allow deletion of artifacts that still have aliases. Use with care.",
    )
    parser.add_argument(
        "--max-delete",
        type=int,
        default=None,
        help="Optional cap on how many filtered candidates to delete, starting from the largest.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.top_n < 1:
        parser.error("--top-n must be at least 1")
    if args.min_size_gib < 0:
        parser.error("--min-size-gib must be non-negative")
    if args.older_than_days is not None and args.older_than_days < 0:
        parser.error("--older-than-days must be non-negative")
    if args.delete and not (args.unaliased_only or args.delete_aliased):
        parser.error("--delete requires --unaliased-only or --delete-aliased for safety")
    if args.max_delete is not None and args.max_delete < 1:
        parser.error("--max-delete must be at least 1")
    try:
        audit(
            entity=args.entity,
            project=args.project,
            all_projects=args.all_projects,
            include_run_files=args.include_run_files,
            top_n=args.top_n,
            json_output=args.json_output,
            collection=args.collection,
            artifact_type_filter=args.artifact_type_filter,
            unaliased_only=args.unaliased_only,
            min_size_gib=args.min_size_gib,
            older_than_days=args.older_than_days,
            delete=args.delete,
            delete_aliased=args.delete_aliased,
            max_delete=args.max_delete,
        )
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
