from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from danish_asr.coral_benchmark import (
    CoRalBenchmarkExample,
    bounded_error_rate,
    load_coral_v3_test_subset,
    normalize_coral_benchmark_text,
    normalized_prediction_reference_pairs,
    score_by_group,
    score_coral_style,
    write_benchmark_outputs,
)


def _sample(
    *,
    seconds: float,
    text: str = "Hej verden",
    validated: str = "accepted",
    age: int = 42,
    gender: str = "female",
    dialect: str = "fynsk",
    country_birth: str | None = "DK",
) -> dict:
    sampling_rate = 16_000
    return {
        "audio": {"array": np.zeros(int(seconds * sampling_rate), dtype=np.float32), "sampling_rate": sampling_rate},
        "text": text,
        "validated": validated,
        "age": age,
        "gender": gender,
        "dialect": dialect,
        "country_birth": country_birth,
    }


def _example(text: str = "Hej verden", **metadata) -> CoRalBenchmarkExample:
    return CoRalBenchmarkExample(
        audio=np.zeros(16_000, dtype=np.float32),
        sampling_rate=16_000,
        text=text,
        subset="read_aloud",
        row_index=0,
        duration_s=1.0,
        metadata={"age": 42, "gender": "female", "dialect": "fynsk", "country_birth": "DK"} | metadata,
    )


def test_normalize_coral_benchmark_text_matches_coral_conventions() -> None:
    text = "AAH øhm, 12 kg + 5% gå-hjem"

    assert normalize_coral_benchmark_text(text) == "åh tolv kilo plus fem procent gå minus hjem"


def test_normalize_coral_benchmark_text_keeps_danish_allowlist_and_strips_punctuation() -> None:
    text = " ÆØÅ ÉÜ! café? aa "

    assert normalize_coral_benchmark_text(text) == "æøå éü café å"


def test_bounded_error_rate_includes_insertions_in_denominator() -> None:
    assert bounded_error_rate(["abcx"], ["abc"], unit="char") == 0.25
    assert bounded_error_rate(["hej smukke verden"], ["hej verden"], unit="word") == 1 / 3


def test_score_coral_style_reports_percent_metrics() -> None:
    scores = score_coral_style(["abcx"], ["abc"])

    assert scores["num_examples"] == 1
    assert scores["cer_coral"] == 25.0
    assert scores["wer_coral"] == 100.0


def test_load_coral_v3_test_subset_filters_like_coral_eval() -> None:
    rows = [
        _sample(seconds=0.5),
        _sample(seconds=1.0, text=""),
        _sample(seconds=1.0, validated="rejected"),
        _sample(seconds=10.0),
        _sample(seconds=2.0, text="Behold me"),
    ]

    def fake_loader(**kwargs):
        assert kwargs["path"] == "CoRal-project/coral-v3"
        assert kwargs["name"] == "read_aloud"
        assert kwargs["split"] == "test"
        return rows

    examples, stats = load_coral_v3_test_subset("read_aloud", dataset_loader=fake_loader)

    assert [example.text for example in examples] == ["Behold me"]
    assert stats.seen == 5
    assert stats.kept == 1
    assert stats.too_short == 1
    assert stats.empty_text == 1
    assert stats.rejected == 1
    assert stats.too_long == 1


def test_normalized_prediction_reference_pairs_uses_same_transform_for_both() -> None:
    predictions, references = normalized_prediction_reference_pairs(["AA 12%"], [_example("å tolv procent")])

    assert predictions == ["å tolv procent"]
    assert references == ["å tolv procent"]


def test_score_by_group_writes_overall_and_demographic_rows() -> None:
    examples = [
        _example("hej verden", gender="female", dialect="fynsk", age=42),
        _example("hej verden", gender="male", dialect="sønderjysk", age=64),
    ]
    records = score_by_group(["hej verden", "hej"], ["hej verden", "hej verden"], examples)

    assert any(
        record["age_group"] is None and record["gender"] is None and record["dialect"] is None for record in records
    )
    assert any(record["gender"] == "male" for record in records)
    assert any(record["dialect"] == "Sønderjysk" for record in records)


def test_write_benchmark_outputs_creates_expected_artifacts(tmp_path: Path) -> None:
    example = _example("Hej verden")
    scores = score_coral_style(["hej verden"], ["hej verden"])
    by_group = score_by_group(["hej verden"], ["hej verden"], [example])

    write_benchmark_outputs(
        tmp_path,
        predictions=["hej verden"],
        references=["hej verden"],
        raw_predictions=["Hej verden"],
        examples=[example],
        scores=scores,
        by_group=by_group,
        metadata={"official_metric": "cer_coral"},
    )

    assert (tmp_path / "predictions.txt").read_text(encoding="utf-8") == "hej verden\n"
    assert (tmp_path / "references.txt").read_text(encoding="utf-8") == "hej verden\n"
    assert (tmp_path / "by_group.csv").exists()
    assert json.loads((tmp_path / "scores.json").read_text(encoding="utf-8"))["scores"]["cer_coral"] == 0.0
    assert json.loads((tmp_path / "records.jsonl").read_text(encoding="utf-8").splitlines()[0])["raw_prediction"] == (
        "Hej verden"
    )


def test_benchmark_cli_smoke_writes_artifacts(tmp_path: Path, monkeypatch) -> None:
    import scripts.hpc.benchmark_coral_style as cli

    examples = [_example("Hej verden")]

    monkeypatch.setattr(
        cli,
        "make_inference_pipeline",
        lambda **kwargs: (object(), Path("tokenizer.model")),
    )
    monkeypatch.setattr(cli, "get_device", lambda: SimpleNamespace(type="cpu"))
    monkeypatch.setattr(cli, "resolve_dtype", lambda dtype_name, device: "float32")
    monkeypatch.setattr(
        cli, "load_coral_v3_test_subset", lambda *args, **kwargs: (examples, SimpleNamespace(__dict__={}))
    )
    monkeypatch.setattr(
        cli,
        "_decode_batch",
        lambda *, examples, pipeline, decoder_kind, beam_decoder, beam_width, removable_tokens: ["Hej verden"],
    )

    cli.main(
        [
            "--checkpoint-path",
            str(tmp_path / "model"),
            "--model-arch",
            "300m_v2",
            "--subset",
            "read_aloud",
            "--max-samples",
            "5",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert (tmp_path / "out" / "predictions.txt").exists()
    score_payload = json.loads((tmp_path / "out" / "scores.json").read_text(encoding="utf-8"))
    assert score_payload["scores"]["cer_coral"] == 0.0
    assert score_payload["metadata"]["decoder"] == "greedy"
    assert score_payload["metadata"]["report_label"] == "greedy"


def test_benchmark_cli_parse_args_supports_decoder_options() -> None:
    import scripts.hpc.benchmark_coral_style as cli

    args = cli.parse_args(
        [
            "--checkpoint-path",
            "model",
            "--model-arch",
            "300m_v2",
            "--subset",
            "read_aloud",
            "--decoder",
            "beam",
            "--kenlm-binary",
            "lm.bin",
            "--beam-width",
            "32",
            "--alpha",
            "0.7",
            "--beta",
            "1.2",
            "--report-label",
            "CTC LM-enabled",
            "--output-dir",
            "out",
        ]
    )

    assert args.decoder == "beam"
    assert args.kenlm_binary == "lm.bin"
    assert args.beam_width == 32
    assert args.alpha == 0.7
    assert args.beta == 1.2
    assert args.report_label == "CTC LM-enabled"


def test_benchmark_cli_beam_without_lm_uses_no_kenlm_decoder(tmp_path: Path, monkeypatch) -> None:
    import scripts.hpc.benchmark_coral_style as cli

    examples = [_example("Hej verden")]
    decoder_calls: list[dict[str, object]] = []
    decode_calls: list[dict[str, object]] = []

    monkeypatch.setattr(cli, "make_inference_pipeline", lambda **kwargs: (object(), Path("tokenizer.model")))
    monkeypatch.setattr(cli, "get_device", lambda: SimpleNamespace(type="cpu"))
    monkeypatch.setattr(cli, "resolve_dtype", lambda dtype_name, device: "float32")
    monkeypatch.setattr(cli, "load_coral_v3_test_subset", lambda *args, **kwargs: (examples, SimpleNamespace(__dict__={})))
    monkeypatch.setattr(cli, "build_pyctcdecode_labels", lambda path: (["", "h", "e", "j"], {"<pad>"}))

    def fake_decoder_factory(labels, *, kenlm_model_path, alpha, beta):
        decoder_calls.append(
            {"labels": labels, "kenlm_model_path": kenlm_model_path, "alpha": alpha, "beta": beta}
        )
        return "beam-decoder"

    def fake_decode_batch(*, examples, pipeline, decoder_kind, beam_decoder, beam_width, removable_tokens):
        decode_calls.append(
            {
                "decoder_kind": decoder_kind,
                "beam_decoder": beam_decoder,
                "beam_width": beam_width,
                "removable_tokens": removable_tokens,
            }
        )
        return ["Hej verden"]

    monkeypatch.setattr(cli, "make_decoder_factory", fake_decoder_factory)
    monkeypatch.setattr(cli, "_decode_batch", fake_decode_batch)

    cli.main(
        [
            "--checkpoint-path",
            str(tmp_path / "model"),
            "--model-arch",
            "300m_v2",
            "--subset",
            "read_aloud",
            "--decoder",
            "beam",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert decoder_calls == [{"labels": ["", "h", "e", "j"], "kenlm_model_path": None, "alpha": 0.6, "beta": 0.5}]
    assert decode_calls == [
        {
            "decoder_kind": "beam",
            "beam_decoder": "beam-decoder",
            "beam_width": 64,
            "removable_tokens": {"<pad>"},
        }
    ]
    score_payload = json.loads((tmp_path / "out" / "scores.json").read_text(encoding="utf-8"))
    assert score_payload["metadata"]["report_label"] == "beam"


def test_benchmark_cli_beam_with_lm_writes_alexandra_label(tmp_path: Path, monkeypatch) -> None:
    import scripts.hpc.benchmark_coral_style as cli

    examples = [_example("Hej verden")]
    decoder_calls: list[dict[str, object]] = []

    monkeypatch.setattr(cli, "make_inference_pipeline", lambda **kwargs: (object(), Path("tokenizer.model")))
    monkeypatch.setattr(cli, "get_device", lambda: SimpleNamespace(type="cpu"))
    monkeypatch.setattr(cli, "resolve_dtype", lambda dtype_name, device: "float32")
    monkeypatch.setattr(cli, "load_coral_v3_test_subset", lambda *args, **kwargs: (examples, SimpleNamespace(__dict__={})))
    monkeypatch.setattr(cli, "build_pyctcdecode_labels", lambda path: (["", "h", "e", "j"], set()))
    monkeypatch.setattr(
        cli,
        "_decode_batch",
        lambda *, examples, pipeline, decoder_kind, beam_decoder, beam_width, removable_tokens: ["Hej verden"],
    )

    def fake_decoder_factory(labels, *, kenlm_model_path, alpha, beta):
        decoder_calls.append(
            {"labels": labels, "kenlm_model_path": kenlm_model_path, "alpha": alpha, "beta": beta}
        )
        return "beam-lm-decoder"

    monkeypatch.setattr(cli, "make_decoder_factory", fake_decoder_factory)

    cli.main(
        [
            "--checkpoint-path",
            str(tmp_path / "model"),
            "--model-arch",
            "300m_v2",
            "--subset",
            "conversation",
            "--decoder",
            "beam",
            "--kenlm-binary",
            "kenlm.bin",
            "--report-label",
            "CTC LM-enabled",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert decoder_calls == [
        {"labels": ["", "h", "e", "j"], "kenlm_model_path": "kenlm.bin", "alpha": 0.6, "beta": 0.5}
    ]
    score_payload = json.loads((tmp_path / "out" / "scores.json").read_text(encoding="utf-8"))
    assert score_payload["metadata"]["decoder"] == "beam"
    assert score_payload["metadata"]["kenlm_binary"] == "kenlm.bin"
    assert score_payload["metadata"]["report_label"] == "CTC LM-enabled"
