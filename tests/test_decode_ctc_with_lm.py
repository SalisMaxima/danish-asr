from __future__ import annotations

import pytest

from scripts.decode_ctc_with_lm import parse_args


def test_parse_args_rejects_kenlm_binary_with_greedy() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--kenlm-binary", "lm.bin", "--decoder", "greedy", "--output-dir", "out"])


def test_parse_args_rejects_beam_width_with_greedy() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--beam-width", "64", "--decoder", "greedy", "--output-dir", "out"])


def test_parse_args_rejects_alpha_with_greedy() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--alpha", "0.5", "--decoder", "greedy", "--output-dir", "out"])


def test_parse_args_accepts_beam_with_kenlm() -> None:
    args = parse_args(
        [
            "--decoder",
            "beam",
            "--kenlm-binary",
            "lm.bin",
            "--output-dir",
            "out",
            "--dataset-root",
            ".",
            "--dataset-split",
            "test",
            "--model-arch",
            "300m_v2",
            "--checkpoint-path",
            "/tmp/ckpt",
        ]
    )
    assert args.kenlm_binary == "lm.bin"
    assert args.decoder == "beam"


def test_parse_args_defaults_to_greedy() -> None:
    args = parse_args(["--output-dir", "out"])
    assert args.decoder == "greedy"


def test_parse_args_beam_without_kenlm_is_allowed() -> None:
    args = parse_args(
        [
            "--decoder",
            "beam",
            "--beam-width",
            "32",
            "--output-dir",
            "out",
            "--dataset-root",
            ".",
            "--dataset-split",
            "test",
            "--model-arch",
            "300m_v2",
            "--checkpoint-path",
            "/tmp/ckpt",
        ]
    )
    assert args.decoder == "beam"
    assert args.beam_width == 32
    assert args.kenlm_binary is None
