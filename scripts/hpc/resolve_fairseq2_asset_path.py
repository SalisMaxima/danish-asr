"""Resolve an already-cached fairseq2 asset to a local path without downloading.

This is intended for zero-shot eval scripts that need to turn a pretrained
asset card such as ``omniASR_LLM_1B_v2`` into an explicit ``model.path``.
It deliberately refuses to download; run the existing asset pull helper first
when the cache is empty.
"""

from __future__ import annotations

import argparse
import os
import sys
from hashlib import sha1
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from danish_asr.utils import configure_project_cache_environment, get_project_fairseq2_cache_dir


def _starts_with_scheme(uri: str) -> bool:
    return bool(urlparse(uri).scheme)


def _normalise_uri(uri: str) -> tuple[str, dict[str, str]]:
    """Mirror fairseq2's cache-key URI normalisation."""
    candidate = uri
    if not _starts_with_scheme(candidate):
        candidate = Path(candidate).as_uri()

    parsed = urlparse(candidate)
    params: dict[str, str] = {}
    if parsed.params:
        for raw_param in parsed.params.split(";"):
            parts = raw_param.split("=")
            if len(parts) == 2:
                params[unquote(parts[0]).strip().lower()] = unquote(parts[1]).strip()

    return parsed._replace(params="").geturl(), params


def cache_dir_for_uri(uri: str, cache_dir: str | Path) -> Path:
    """Return the fairseq2 content-addressed cache directory for ``uri``."""
    normalised_uri, _params = _normalise_uri(uri)
    return Path(cache_dir).expanduser() / sha1(normalised_uri.encode(), usedforsecurity=False).hexdigest()[:24]


def resolve_cached_asset_uri(uri: str, cache_dir: str | Path) -> Path | None:
    """Resolve a cached fairseq2 asset URI without triggering a download."""
    normalised_uri, params = _normalise_uri(uri)
    if normalised_uri.startswith("file://"):
        parsed_uri = urlparse(normalised_uri)
        uri_path = f"//{parsed_uri.netloc}{parsed_uri.path}" if parsed_uri.netloc else parsed_uri.path
        return Path(url2pathname(uri_path))

    asset_dir = cache_dir_for_uri(normalised_uri, cache_dir)
    if not asset_dir.exists():
        return None

    asset_pathname = params.get("path")
    if asset_pathname:
        asset_path = (asset_dir / asset_pathname).resolve()
        try:
            asset_path.relative_to(asset_dir.resolve())
        except ValueError:
            msg = "path parameter of the URI points outside the asset cache directory"
            raise ValueError(msg) from None
        return asset_path if asset_path.exists() else None

    files = [path for path in asset_dir.iterdir() if path.is_file()]
    if len(files) == 1:
        return files[0]
    return asset_dir


def resolve_cached_asset_card(
    asset_name: str,
    *,
    field: str,
    cache_dir: str | Path | None = None,
) -> Path | None:
    """Resolve a cached fairseq2 asset card field such as ``checkpoint``."""
    from fairseq2.assets import get_asset_store

    card = get_asset_store().retrieve_card(asset_name)
    uri = card.field(field).as_uri()
    resolved_cache_dir = Path(cache_dir or os.environ.get("FAIRSEQ2_CACHE_DIR") or get_project_fairseq2_cache_dir())
    return resolve_cached_asset_uri(str(uri), resolved_cache_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True, help="fairseq2 asset card name, e.g. omniASR_LLM_1B_v2")
    parser.add_argument("--field", default="checkpoint", help="asset card URI field to resolve")
    parser.add_argument("--cache-dir", default=None, help="Override FAIRSEQ2_CACHE_DIR")
    parser.add_argument("--require-existing", action="store_true", help="Exit non-zero if the cached asset is missing")
    parser.add_argument("--print-shell", default="", help="Print KEY=quoted_path instead of the raw path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_project_cache_environment()
    args = parse_args(argv)
    try:
        path = resolve_cached_asset_card(args.asset, field=args.field, cache_dir=args.cache_dir)
    except Exception as ex:
        print(f"ERROR: Could not resolve {args.asset}.{args.field}: {ex}", file=sys.stderr)
        return 1

    if path is None:
        message = (
            f"Cached fairseq2 asset not found for {args.asset}.{args.field}. "
            "Pre-download it first, then rerun this resolver."
        )
        if args.require_existing:
            print(f"ERROR: {message}", file=sys.stderr)
            return 1
        print(message, file=sys.stderr)
        return 0

    if args.print_shell:
        import shlex

        print(f"{args.print_shell}={shlex.quote(str(path))}")
    else:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
