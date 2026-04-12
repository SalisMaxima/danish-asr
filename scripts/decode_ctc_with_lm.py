"""Run standalone CTC decoding for OmniASR checkpoints with greedy or pyctcdecode beam search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fairseq2.nn.batch_layout import BatchLayout
from loguru import logger

from danish_asr.lm import (
    DecodeResult,
    build_pyctcdecode_labels,
    collate_decode_records,
    decode_logits_with_argmax,
    infer_split_from_eval_config,
    iter_fairseq2_rows,
    make_decoder_factory,
    make_inference_pipeline,
    parse_valid_split,
    resolve_dtype,
    score_predictions,
    strip_special_tokens,
    write_text_lines,
)
from danish_asr.utils import configure_project_cache_environment, get_device, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-config", default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--model-arch", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--dataset-split", default=None)
    parser.add_argument("--tokenizer-name", default="omniASR_tokenizer_written_v2")
    parser.add_argument("--tokenizer-model-path", "--vocab-path", dest="tokenizer_model_path", default=None)
    parser.add_argument("--decoder", choices=("greedy", "beam"), default="greedy")
    parser.add_argument("--kenlm-binary", default=None)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _resolve_inputs(args: argparse.Namespace) -> dict[str, str]:
    if args.eval_config:
        inferred = infer_split_from_eval_config(args.eval_config)
        return {
            "checkpoint_path": str(inferred["checkpoint_path"]),
            "model_arch": inferred["model_arch"],
            "dataset_root": str(inferred["dataset_root"]),
            "dataset_split": inferred["valid_split"],
            "tokenizer_name": inferred["tokenizer_name"],
        }

    required = {
        "checkpoint_path": args.checkpoint_path,
        "model_arch": args.model_arch,
        "dataset_root": args.dataset_root,
        "dataset_split": args.dataset_split,
        "tokenizer_name": args.tokenizer_name,
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        msg = "When --eval-config is not provided, the following arguments are required: " + ", ".join(
            f"--{name.replace('_', '-')}" for name in missing
        )
        raise ValueError(msg)

    return {key: str(value) for key, value in required.items()}


def _decode_batch(
    *,
    audio_payloads: list[bytes],
    references: list[str],
    corpora: list[str],
    files: list[str],
    row_indices: list[int],
    pipeline,
    decoder_kind: str,
    beam_decoder,
    beam_width: int,
    removable_tokens: set[str],
) -> list[DecodeResult]:
    audio_tensors = list(pipeline._build_audio_wavform_pipeline(audio_payloads).and_return())
    batch = pipeline._create_batch_simple([(audio_tensor, None) for audio_tensor in audio_tensors])
    batch_layout = BatchLayout(
        batch.source_seqs.shape,
        seq_lens=batch.source_seq_lens,
        device=batch.source_seqs.device,
    )
    logits, output_layout = pipeline.model(batch.source_seqs, batch_layout)

    token_decoder = pipeline.token_decoder
    decoded: list[str] = []

    for index in range(logits.shape[0]):
        seq_len = int(output_layout.seq_lens[index])
        logit_slice = logits[index, :seq_len]

        if decoder_kind == "greedy":
            hypothesis = decode_logits_with_argmax(logit_slice, seq_len=seq_len, token_decoder=token_decoder)
        else:
            if beam_decoder is None:
                raise ValueError("Beam decoder must be initialized when --decoder beam is selected.")

            hypothesis = beam_decoder.decode(logit_slice.float().cpu().numpy(), beam_width=beam_width)
            hypothesis = strip_special_tokens(hypothesis, removable_tokens)

        decoded.append(hypothesis)

    return [
        DecodeResult(
            prediction=prediction,
            reference=reference,
            corpus=corpus,
            file=file_path,
            row_index=row_index,
        )
        for prediction, reference, corpus, file_path, row_index in zip(
            decoded, references, corpora, files, row_indices, strict=True
        )
    ]


def main() -> None:
    configure_project_cache_environment()
    args = parse_args()
    resolved = _resolve_inputs(args)

    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    dtype = resolve_dtype(args.dtype, device)

    pipeline, tokenizer_model_path = make_inference_pipeline(
        checkpoint_path=resolved["checkpoint_path"],
        model_arch=resolved["model_arch"],
        tokenizer_name=resolved["tokenizer_name"],
        tokenizer_model_path=args.tokenizer_model_path,
        device=device,
        dtype=dtype,
    )

    split_name, corpus_name = parse_valid_split(resolved["dataset_split"])
    corpora = (corpus_name,) if corpus_name else None

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

    batch_payloads: list[bytes] = []
    batch_refs: list[str] = []
    batch_corpora: list[str] = []
    batch_files: list[str] = []
    batch_rows: list[int] = []
    results: list[DecodeResult] = []

    for example_index, row in enumerate(
        iter_fairseq2_rows(
            Path(resolved["dataset_root"]),
            split=split_name,
            corpora=corpora,
            columns=("text", "audio_bytes"),
        )
    ):
        batch_payloads.append(row["audio_bytes"])
        batch_refs.append(str(row["text"]))
        batch_corpora.append(str(row["corpus"]))
        batch_files.append(str(row["file"]))
        batch_rows.append(int(row["row_index"]))

        should_flush = len(batch_payloads) >= args.batch_size
        reached_limit = args.max_samples is not None and example_index + 1 >= args.max_samples
        if should_flush or reached_limit:
            results.extend(
                _decode_batch(
                    audio_payloads=batch_payloads,
                    references=batch_refs,
                    corpora=batch_corpora,
                    files=batch_files,
                    row_indices=batch_rows,
                    pipeline=pipeline,
                    decoder_kind=args.decoder,
                    beam_decoder=beam_decoder,
                    beam_width=args.beam_width,
                    removable_tokens=removable_tokens,
                )
            )
            batch_payloads = []
            batch_refs = []
            batch_corpora = []
            batch_files = []
            batch_rows = []

        if reached_limit:
            break

    if batch_payloads:
        results.extend(
            _decode_batch(
                audio_payloads=batch_payloads,
                references=batch_refs,
                corpora=batch_corpora,
                files=batch_files,
                row_indices=batch_rows,
                pipeline=pipeline,
                decoder_kind=args.decoder,
                beam_decoder=beam_decoder,
                beam_width=args.beam_width,
                removable_tokens=removable_tokens,
            )
        )

    predictions, references = collate_decode_records(results)
    predictions_path = output_dir / "predictions.txt"
    references_path = output_dir / "references.txt"
    records_path = output_dir / "records.jsonl"
    metadata_path = output_dir / "metadata.json"
    score_path = output_dir / "scores.json"

    write_text_lines(predictions_path, predictions)
    write_text_lines(references_path, references)
    records_path.write_text(
        "".join(json.dumps(result.__dict__, ensure_ascii=False) + "\n" for result in results),
        encoding="utf-8",
    )

    metadata = {
        "checkpoint_path": resolved["checkpoint_path"],
        "model_arch": resolved["model_arch"],
        "dataset_root": resolved["dataset_root"],
        "dataset_split": resolved["dataset_split"],
        "tokenizer_name": resolved["tokenizer_name"],
        "tokenizer_model_path": str(tokenizer_model_path),
        "decoder": args.decoder,
        "kenlm_binary": args.kenlm_binary,
        "beam_width": args.beam_width,
        "alpha": args.alpha,
        "beta": args.beta,
        "batch_size": args.batch_size,
        "dtype": str(dtype),
        "num_examples": len(results),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    score_summary = score_predictions(predictions, references)
    score_path.write_text(json.dumps(score_summary, indent=2) + "\n", encoding="utf-8")

    logger.info("Decoded {} examples -> {}", len(results), output_dir)
    logger.info("WER: {:.2f}%", score_summary["wer"])


if __name__ == "__main__":
    main()
