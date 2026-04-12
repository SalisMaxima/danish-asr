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
    parser.add_argument("--config", default="configs/lm/coral_train_only_v1.yaml")
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--text-path", default=None)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--lmplz-bin", default="lmplz")
    parser.add_argument("--build-binary-bin", default="build_binary")
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

    run_kenlm_command([args.lmplz_bin, "-o", str(args.order)], stdin_path=text_path, stdout_path=arpa_path)
    run_kenlm_command([args.build_binary_bin, str(arpa_path), str(binary_path)])

    logger.info("Built KenLM artifacts: {} and {}", arpa_path, binary_path)


if __name__ == "__main__":
    main()
