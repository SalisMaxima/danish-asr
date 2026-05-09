"""Build a KenLM 3-gram model from a prepared text corpus."""

from __future__ import annotations

import argparse
import subprocess
from contextlib import ExitStack
from pathlib import Path

from loguru import logger

from danish_asr.lm import load_yaml_config
from danish_asr.utils import resolve_project_path


def run_kenlm_command(command: list[str], *, stdin_path: Path | None = None, stdout_path: Path | None = None) -> None:
    """Run a KenLM binary with optional redirected stdin/stdout."""
    logger.info("Running: {}", " ".join(command))

    with ExitStack() as stack:
        stdin_handle = stack.enter_context(stdin_path.open("rb")) if stdin_path is not None else None
        stdout_handle = stack.enter_context(stdout_path.open("wb")) if stdout_path is not None else None
        subprocess.run(
            command,
            check=True,
            stdin=stdin_handle,
            stdout=stdout_handle,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lm/alexandra_proxy_v1.yaml")
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--text-path", default=None)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--lmplz-bin", default="lmplz")
    parser.add_argument("--build-binary-bin", default="build_binary")
    parser.add_argument("--memory", default=None, help="KenLM lmplz memory budget, e.g. 24G or 80%.")
    parser.add_argument("--temp-dir", type=Path, default=None, help="Directory for KenLM temporary files.")
    parser.add_argument(
        "--skip-symbols",
        action="store_true",
        help="Pass KenLM --skip_symbols to treat <s>, </s>, and <unk> in corpus text as whitespace.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    default_text_path = resolve_project_path(config["output"]["corpus_text_path"])
    text_path = resolve_project_path(args.text_path) if args.text_path else default_text_path

    default_prefix = resolve_project_path(f"artifacts/lm/{config['name']}_{args.order}gram")
    output_prefix = resolve_project_path(args.output_prefix) if args.output_prefix else default_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    arpa_path = output_prefix.with_suffix(".arpa")
    binary_path = output_prefix.with_suffix(".bin")

    lmplz_command = [args.lmplz_bin, "-o", str(args.order)]
    if args.memory:
        lmplz_command.extend(["-S", args.memory])
    if args.temp_dir:
        temp_dir = resolve_project_path(args.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        lmplz_command.extend(["-T", str(temp_dir)])
    if args.skip_symbols:
        lmplz_command.append("--skip_symbols")

    if not text_path.is_file():
        msg = f"LM corpus text file not found: {text_path}"
        raise FileNotFoundError(msg)
    if text_path.stat().st_size == 0:
        msg = f"LM corpus text file is empty: {text_path}. Run build_danish_lm_corpus.py first."
        raise ValueError(msg)

    run_kenlm_command(lmplz_command, stdin_path=text_path, stdout_path=arpa_path)
    run_kenlm_command([args.build_binary_bin, str(arpa_path), str(binary_path)])

    logger.info("Built KenLM artifacts: {} and {}", arpa_path, binary_path)


if __name__ == "__main__":
    main()
