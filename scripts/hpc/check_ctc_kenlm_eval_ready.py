"""Preflight and manifest helpers for CTC + beam + KenLM HPC evaluations."""

from __future__ import annotations

import argparse
import importlib
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "eval" / "ctc_kenlm_finetuned_hpc.yaml"
LANGUAGE = "dan_Latn"


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    return value


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Load and environment-expand the shared CTC + KenLM eval manifest."""
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    if not isinstance(manifest, dict):
        msg = f"Manifest must be a YAML mapping: {manifest_path}"
        raise ValueError(msg)
    return _expand(manifest)


def _checkpoint_exists(path: str | Path) -> bool:
    checkpoint = Path(path)
    if checkpoint.is_file():
        return True
    if not checkpoint.is_dir():
        return False
    return (checkpoint / "pp_00" / "tp_00" / "sdp_00.pt").is_file() or (
        checkpoint / "model" / "pp_00" / "tp_00" / "sdp_00.pt"
    ).is_file()


# Mirrors lm.parse_valid_split; kept private to avoid importing the full lm module
# in this preflight script which is designed to run with minimal dependencies.
def _parse_valid_split(valid_split: str) -> tuple[str, str | None]:
    if valid_split in {"train", "dev", "test"}:
        return valid_split, None
    split, _, corpus = valid_split.partition("_")
    if split not in {"train", "dev", "test"} or not corpus:
        msg = f"Unsupported fairseq2 split: {valid_split}"
        raise ValueError(msg)
    return split, corpus


def _split_exists(dataset_root: str | Path, valid_split: str) -> bool:
    root = Path(dataset_root)
    split, corpus = _parse_valid_split(valid_split)
    if corpus is not None:
        return any((root / f"corpus={corpus}" / f"split={split}" / f"language={LANGUAGE}").glob("*.parquet"))
    return any(root.glob(f"corpus=*/split={split}/language={LANGUAGE}/*.parquet"))


def _import_modules(module_names: Iterable[str]) -> list[str]:
    errors: list[str] = []
    for module_name in module_names:
        try:
            importlib.import_module(module_name)
        except Exception as ex:  # pragma: no cover - exact dependency failures are environment-specific
            errors.append(f"Cannot import {module_name}: {ex}")
    return errors


def _selected_models(manifest: dict[str, Any], labels: set[str] | None) -> list[dict[str, Any]]:
    models = list(manifest.get("models", []))
    if labels is None:
        return models
    selected = [model for model in models if str(model["label"]) in labels]
    missing = labels - {str(model["label"]) for model in selected}
    if missing:
        msg = f"Unknown model label(s): {', '.join(sorted(missing))}"
        raise ValueError(msg)
    return selected


def _nearest_existing_parent(path: Path) -> Path | None:
    candidate = path if path.exists() else path.parent
    while not candidate.exists():
        if candidate == candidate.parent:
            return None
        candidate = candidate.parent
    return candidate


def _writable_path_errors(paths: Iterable[str | Path]) -> list[str]:
    errors: list[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        existing_parent = _nearest_existing_parent(path)
        if existing_parent is None:
            errors.append(f"No existing parent directory for output path: {path}")
            continue
        if not os.access(existing_parent, os.W_OK | os.X_OK):
            errors.append(f"Output path parent is not writable/searchable: {existing_parent} (for {path})")
    return errors


def _selected_output_roots(manifest: dict[str, Any], method: str) -> list[str]:
    output_roots = manifest["output_roots"]
    roots: list[str] = []
    if method in {"all", "my"}:
        roots.append(str(output_roots["my_method"]))
    if method in {"all", "coral"}:
        roots.append(str(output_roots["coral_method"]))
    roots.append(str(output_roots["results"]))
    return roots


def _format_exists(path: Path) -> str:
    return "exists" if path.exists() else "missing"


def _fairseq2_cache_roots() -> list[Path]:
    roots: list[Path] = []
    fairseq2_cache_dir = os.environ.get("FAIRSEQ2_CACHE_DIR")
    if fairseq2_cache_dir is not None and fairseq2_cache_dir.strip():
        cache_path = Path(fairseq2_cache_dir).expanduser()
        if cache_path.name == "assets":
            roots.extend([cache_path, cache_path.parent])
        else:
            roots.extend([cache_path, cache_path / "assets"])
    roots.extend([Path.home() / ".cache" / "fairseq2", Path.home() / ".cache" / "fairseq2" / "assets"])

    deduped: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        deduped.append(root)
    return deduped


def _resolve_tokenizer_model_path(tokenizer: dict[str, Any]) -> tuple[Path | None, str | None]:
    explicit_path = tokenizer.get("model_path")
    if explicit_path:
        path = Path(str(explicit_path)).expanduser()
        if path.is_file():
            return path, None
        return None, f"Tokenizer model_path not found: {path}"

    tokenizer_name = str(tokenizer["name"])
    try:
        from danish_asr.lm import _get_cached_tokenizer_path

        path = _get_cached_tokenizer_path(tokenizer_name)
    except Exception as ex:  # pragma: no cover - exact fairseq2 failures are environment-specific
        return None, f"Cannot resolve cached tokenizer for {tokenizer_name}: {ex}"

    if path is not None:
        return path, None

    roots = ", ".join(f"{root} ({_format_exists(root)})" for root in _fairseq2_cache_roots())
    return (
        None,
        f"Cached tokenizer not found for {tokenizer_name}; checked FAIRSEQ2 cache roots: {roots}. "
        "Pre-download the asset or set TOKENIZER_MODEL_PATH/manifest tokenizer.model_path explicitly.",
    )


def build_diagnostics(
    manifest: dict[str, Any],
    *,
    method: str,
    model_labels: set[str] | None = None,
    require_tokenizer: bool = True,
) -> list[str]:
    """Return human-readable diagnostics for paths and cache layout."""
    artifacts = manifest["artifacts"]
    tokenizer = manifest["tokenizer"]
    models = _selected_models(manifest, model_labels)

    lines = [
        f"method: {method}",
        f"kenlm_binary: {artifacts['kenlm_binary']} ({_format_exists(Path(artifacts['kenlm_binary']))})",
        f"parquet_root: {artifacts['parquet_root']} ({_format_exists(Path(artifacts['parquet_root']))})",
        f"hf_cache_dir: {artifacts['hf_cache_dir']} ({_format_exists(Path(artifacts['hf_cache_dir']))})",
        "output_roots: " + ", ".join(_selected_output_roots(manifest, method)),
        "fairseq2_cache_roots: " + ", ".join(f"{root} ({_format_exists(root)})" for root in _fairseq2_cache_roots()),
        "models: "
        + ", ".join(
            f"{model['label']}={model['checkpoint_path']} ({_format_exists(Path(model['checkpoint_path']))})"
            for model in models
        ),
    ]

    if require_tokenizer:
        tokenizer_path, tokenizer_error = _resolve_tokenizer_model_path(tokenizer)
        if tokenizer_path is not None:
            lines.append(f"tokenizer_model: {tokenizer_path} (exists)")
        elif tokenizer_error is not None:
            lines.append(f"tokenizer_model: unresolved ({tokenizer_error})")

    return lines


def _print_quota() -> None:
    quota_bin = shutil.which("getquota_work3.sh")
    if quota_bin is None:
        return
    print("=== work3 quota ===")
    result = subprocess.run([quota_bin], check=False)
    if result.returncode != 0:
        print(
            f"WARNING: getquota_work3.sh exited with code {result.returncode}; quota information may be incomplete.",
            file=sys.stderr,
        )


def validate_manifest(
    manifest: dict[str, Any],
    *,
    method: str,
    model_labels: set[str] | None = None,
    require_imports: bool = True,
    require_cuda: bool = True,
    require_kenlm: bool = True,
    require_tokenizer: bool | None = None,
    require_output_roots: bool = True,
) -> list[str]:
    """Return preflight errors for the requested eval method."""
    errors: list[str] = []
    artifacts = manifest["artifacts"]
    tokenizer = manifest["tokenizer"]
    models = _selected_models(manifest, model_labels)
    if require_tokenizer is None:
        require_tokenizer = require_imports

    if require_imports:
        imports = ["fairseq2", "omnilingual_asr", "pyctcdecode", "kenlm", "pyarrow", "torch"]
        if method in {"all", "coral"}:
            imports.append("datasets")
        errors.extend(_import_modules(imports))

    if require_cuda:
        try:
            import torch

            if not torch.cuda.is_available():
                errors.append("torch.cuda.is_available() is false; run CUDA-required preflight on a GPU node.")
        except Exception as ex:  # pragma: no cover - environment-specific
            errors.append(f"Cannot check CUDA availability: {ex}")

    if require_kenlm and not Path(artifacts["kenlm_binary"]).is_file():
        errors.append(f"KenLM binary not found: {artifacts['kenlm_binary']}")

    if require_tokenizer:
        tokenizer_path, tokenizer_error = _resolve_tokenizer_model_path(tokenizer)
        if tokenizer_path is None and tokenizer_error is not None:
            errors.append(tokenizer_error)

    for model in models:
        if not _checkpoint_exists(model["checkpoint_path"]):
            errors.append(f"Checkpoint not found for {model['label']}: {model['checkpoint_path']}")

    if require_output_roots:
        errors.extend(_writable_path_errors(_selected_output_roots(manifest, method)))

    if method in {"all", "my"}:
        dataset_root = artifacts["parquet_root"]
        for split in manifest["my_method"]["dataset_splits"]:
            if not _split_exists(dataset_root, split["split"]):
                errors.append(f"Parquet split not found for {split['label']}: {dataset_root} :: {split['split']}")

    if method in {"all", "coral"}:
        hf_cache_dir = Path(artifacts["hf_cache_dir"])
        expected_prefix = f"/work3/{os.environ.get('USER', '')}/hf_cache"
        if not str(hf_cache_dir).startswith(expected_prefix):
            errors.append(f"HF cache must stay under {expected_prefix}: {hf_cache_dir}")

    return errors


def _emit_models(manifest: dict[str, Any]) -> None:
    for model in manifest["models"]:
        print(
            "\t".join(
                [
                    str(model["label"]),
                    str(model["arch"]),
                    str(model["checkpoint_path"]),
                    str(model["batch_size"]),
                ]
            )
        )


def _emit_my_splits(manifest: dict[str, Any]) -> None:
    for split in manifest["my_method"]["dataset_splits"]:
        print(f"{split['label']}\t{split['split']}")


def _emit_coral_subsets(manifest: dict[str, Any]) -> None:
    for subset in manifest["coral"]["subsets"]:
        print(str(subset))


def _emit_shell(manifest: dict[str, Any]) -> None:
    artifacts = manifest["artifacts"]
    output_roots = manifest["output_roots"]
    tokenizer = manifest["tokenizer"]
    beam = manifest["beam"]
    coral = manifest["coral"]
    values = {
        "CTC_KENLM_BINARY": artifacts["kenlm_binary"],
        "CTC_PARQUET_ROOT": artifacts["parquet_root"],
        "CTC_HF_CACHE_DIR": artifacts["hf_cache_dir"],
        "CTC_OUTPUT_ROOT_MY": output_roots["my_method"],
        "CTC_OUTPUT_ROOT_CORAL": output_roots["coral_method"],
        "CTC_RESULTS_ROOT": output_roots["results"],
        "CTC_TOKENIZER_NAME": tokenizer["name"],
        "CTC_TOKENIZER_MODEL_PATH": tokenizer.get("model_path") or "",
        "CTC_BEAM_WIDTH": beam["width"],
        "CTC_ALPHA": beam["alpha"],
        "CTC_BETA": beam["beta"],
        "CTC_MIN_SECONDS": coral["min_seconds"],
        "CTC_MAX_SECONDS": coral["max_seconds"],
    }
    for key, value in values.items():
        print(f"{key}={shlex.quote(str(value))}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--method", choices=("all", "my", "coral"), default="all")
    parser.add_argument("--models", default="", help="Comma-separated model labels to validate.")
    parser.add_argument("--skip-imports", action="store_true")
    parser.add_argument("--skip-cuda", action="store_true")
    parser.add_argument("--skip-tokenizer", action="store_true")
    parser.add_argument("--skip-output-roots", action="store_true")
    parser.add_argument("--allow-missing-kenlm", action="store_true")
    parser.add_argument("--print-quota", action="store_true")
    parser.add_argument("--quiet-diagnostics", action="store_true")
    parser.add_argument("--emit", choices=("models", "my-splits", "coral-subsets", "shell"), default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_manifest(args.manifest)

    if args.emit == "models":
        _emit_models(manifest)
        return 0
    if args.emit == "my-splits":
        _emit_my_splits(manifest)
        return 0
    if args.emit == "coral-subsets":
        _emit_coral_subsets(manifest)
        return 0
    if args.emit == "shell":
        _emit_shell(manifest)
        return 0

    if args.print_quota:
        _print_quota()

    model_labels = {label for label in args.models.split(",") if label} or None
    if not args.quiet_diagnostics:
        print("=== CTC + KenLM eval preflight diagnostics ===")
        for line in build_diagnostics(
            manifest,
            method=args.method,
            model_labels=model_labels,
            require_tokenizer=not args.skip_tokenizer and not args.skip_imports,
        ):
            print(line)
        print("=============================================")

    errors = validate_manifest(
        manifest,
        method=args.method,
        model_labels=model_labels,
        require_imports=not args.skip_imports,
        require_cuda=not args.skip_cuda,
        require_kenlm=not args.allow_missing_kenlm,
        require_tokenizer=not args.skip_tokenizer and not args.skip_imports,
        require_output_roots=not args.skip_output_roots,
    )
    if errors:
        print("CTC + KenLM eval preflight failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(f"CTC + KenLM eval preflight OK ({args.method})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
