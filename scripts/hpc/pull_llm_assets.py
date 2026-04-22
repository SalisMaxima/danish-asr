"""Pre-download omniASR LLM V2 checkpoint + tokenizer to FAIRSEQ2_CACHE_DIR.

Run on the DTU HPC login node (or locally) before submitting a training job so
the compute node does not spend GPU-allocated time downloading multi-GiB
checkpoints. Fairseq2's asset cache is content-addressed, so re-running is safe.

Usage:
    uv run python scripts/hpc/pull_llm_assets.py --size 300m
    uv run python scripts/hpc/pull_llm_assets.py --size 1b

The FAIRSEQ2_CACHE_DIR env var (set by scripts/hpc/env.sh) controls where
assets land. On DTU HPC this resolves to /work3/$USER/fairseq2_cache/.
"""

import argparse
import os

CARDS: dict[str, str] = {
    "300m": "omniASR_LLM_300M_v2",
    "1b": "omniASR_LLM_1B_v2",
}

SIZES: dict[str, str] = {
    "300m": "6.1 GiB",
    "1b": "8.5 GiB",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", choices=sorted(CARDS), required=True, help="Model size to pre-download.")
    args = ap.parse_args()

    card = CARDS[args.size]
    cache = os.environ.get("FAIRSEQ2_CACHE_DIR", "~/.cache/fairseq2/assets")
    print(f"Downloading {card} (~{SIZES[args.size]}) → {cache}")

    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    ASRInferencePipeline(model_card=card)
    print(f"Done: {card} cached at {cache}")


if __name__ == "__main__":
    main()
