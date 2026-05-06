"""Evaluate an OmniASR checkpoint with Alexandra/CoRal-style CER/WER."""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import torch
from loguru import logger

from danish_asr.coral_benchmark import (
    TARGET_SAMPLE_RATE,
    CoRalBenchmarkExample,
    load_coral_v3_test_subset,
    normalized_prediction_reference_pairs,
    score_by_group,
    score_coral_style,
    write_benchmark_outputs,
)
from danish_asr.lm import (
    build_pyctcdecode_labels,
    decode_ctc_logits,
    make_decoder_factory,
    make_inference_pipeline,
    resolve_dtype,
)
from danish_asr.utils import get_device, resolve_project_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--model-arch", required=True, choices=("300m_v2", "1b_v2", "3b_v2"))
    parser.add_argument("--subset", required=True, choices=("read_aloud", "conversation"))
    parser.add_argument("--tokenizer-name", default="omniASR_tokenizer_written_v2")
    parser.add_argument("--tokenizer-model-path", default=None)
    parser.add_argument("--decoder", choices=("greedy", "beam"), default="greedy")
    parser.add_argument("--kenlm-binary", default=None)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--report-label", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def _audio_to_tensor(example: CoRalBenchmarkExample) -> torch.Tensor:
    audio = example.audio
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)

    waveform = torch.tensor(audio, dtype=torch.float32)
    if example.sampling_rate != TARGET_SAMPLE_RATE:
        import torchaudio

        waveform = torchaudio.functional.resample(waveform, example.sampling_rate, TARGET_SAMPLE_RATE)
    return waveform


def _decode_batch(
    *,
    examples: list[CoRalBenchmarkExample],
    pipeline: Any,
    decoder_kind: str,
    beam_decoder: Any,
    beam_width: int,
    removable_tokens: set[str],
) -> list[str]:
    from fairseq2.nn.batch_layout import BatchLayout

    audio_tensors = [_audio_to_tensor(example) for example in examples]
    batch = pipeline._create_batch_simple([(audio_tensor, None) for audio_tensor in audio_tensors])
    batch_layout = BatchLayout(
        batch.source_seqs.shape,
        seq_lens=batch.source_seq_lens,
        device=batch.source_seqs.device,
    )
    logits, output_layout = pipeline.model(batch.source_seqs, batch_layout)

    predictions: list[str] = []
    for index in range(logits.shape[0]):
        seq_len = int(output_layout.seq_lens[index])
        predictions.append(
            decode_ctc_logits(
                logits[index],
                seq_len=seq_len,
                token_decoder=pipeline.token_decoder,
                decoder_kind=decoder_kind,
                beam_decoder=beam_decoder,
                beam_width=beam_width,
                removable_tokens=removable_tokens,
            )
        )
    return predictions


def _default_report_label(args: argparse.Namespace) -> str:
    if args.report_label is not None:
        return args.report_label
    if args.decoder == "greedy":
        return "greedy"
    if args.kenlm_binary:
        return "beam + KenLM"
    return "beam"


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    dtype = resolve_dtype(args.dtype, device)
    pipeline, tokenizer_model_path = make_inference_pipeline(
        checkpoint_path=args.checkpoint_path,
        model_arch=args.model_arch,
        tokenizer_name=args.tokenizer_name,
        tokenizer_model_path=args.tokenizer_model_path,
        device=device,
        dtype=dtype,
    )
    model = getattr(pipeline, "model", None)
    if model is not None and hasattr(model, "eval"):
        model.eval()

    beam_decoder = None
    removable_tokens: set[str] = set()
    if args.decoder == "beam":
        labels, removable_tokens = build_pyctcdecode_labels(tokenizer_model_path)
        beam_decoder = make_decoder_factory(
            labels,
            kenlm_model_path=args.kenlm_binary,
            alpha=args.alpha,
            beta=args.beta,
        )

    examples, filter_stats = load_coral_v3_test_subset(
        args.subset,
        max_samples=args.max_samples,
        cache_dir=args.cache_dir,
    )
    logger.info("Loaded {} filtered CoRal-v3 {} examples", len(examples), args.subset)
    logger.info("Filter stats: {}", json.dumps(filter_stats.__dict__, ensure_ascii=False))

    raw_predictions: list[str] = []
    with torch.inference_mode():
        for start in range(0, len(examples), args.batch_size):
            batch_examples = examples[start : start + args.batch_size]
            raw_predictions.extend(
                _decode_batch(
                    examples=batch_examples,
                    pipeline=pipeline,
                    decoder_kind=args.decoder,
                    beam_decoder=beam_decoder,
                    beam_width=args.beam_width,
                    removable_tokens=removable_tokens,
                )
            )
            logger.info("Decoded {}/{} examples", len(raw_predictions), len(examples))

    predictions, references = normalized_prediction_reference_pairs(raw_predictions, examples)
    scores = score_coral_style(predictions, references)
    by_group = score_by_group(predictions, references, examples)
    metadata = {
        "checkpoint_path": str(args.checkpoint_path),
        "model_arch": args.model_arch,
        "subset": args.subset,
        "decoder": args.decoder,
        "report_label": _default_report_label(args),
        "kenlm_binary": args.kenlm_binary,
        "beam_width": args.beam_width,
        "alpha": args.alpha,
        "beta": args.beta,
        "tokenizer_name": args.tokenizer_name,
        "tokenizer_model_path": str(tokenizer_model_path),
        "batch_size": args.batch_size,
        "dtype": str(dtype),
        "max_samples": args.max_samples,
        "filter_stats": filter_stats.__dict__,
        "official_metric": "cer_coral",
        "metric_units": "percent",
    }

    write_benchmark_outputs(
        output_dir,
        predictions=predictions,
        references=references,
        raw_predictions=raw_predictions,
        examples=examples,
        scores=scores,
        by_group=by_group,
        metadata=metadata,
    )

    logger.info("CoRal-style CER: {:.2f}%", scores["cer_coral"])
    logger.info("CoRal-style WER: {:.2f}%", scores["wer_coral"])
    return {"scores": scores, "metadata": metadata}


def main(argv: list[str] | None = None) -> None:
    run_benchmark(parse_args(argv))


if __name__ == "__main__":
    main()
