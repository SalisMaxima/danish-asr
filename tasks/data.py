"""Data management tasks for Danish ASR."""

import os
from pathlib import Path

from invoke import Context, task
from loguru import logger

_PROJECT_ROOT = Path(__file__).parent.parent
WINDOWS = os.name == "nt"
PROJECT_NAME = "danish_asr"
VALID_SUBSETS = {"read_aloud", "conversation"}
VALID_CONVERT_SUBSETS = {"read_aloud", "conversation", "all"}
HF_CACHE_DIR = str(_PROJECT_ROOT / ".cache" / "huggingface")


def _hf_env_prefix() -> str:
    """Return shell prefix that sets HF_HOME to the project cache dir."""
    return f"HF_HOME={HF_CACHE_DIR} "


@task
def download(ctx: Context, subset: str = "read_aloud") -> None:
    """Download CoRal Danish speech dataset via HuggingFace.

    Args:
        subset: Dataset subset (read_aloud or conversation)
    """
    if subset not in VALID_SUBSETS:
        raise ValueError(f"Invalid subset {subset!r}. Must be one of: {VALID_SUBSETS}")
    logger.info(f"Downloading CoRal dataset (subset={subset})...")
    ctx.run(
        _hf_env_prefix() + f'uv run python -c "'
        f"from datasets import load_dataset; "
        f"ds = load_dataset('CoRal-project/coral-v3', '{subset}', "
        f"cache_dir='{HF_CACHE_DIR}'); "
        f"print('Downloaded:', len(ds), 'splits'); "
        f"[print(' ', k + ':', len(v), 'samples') for k, v in ds.items()]"
        f'"',
        echo=True,
        pty=not WINDOWS,
    )


@task
def stats(ctx: Context, subset: str = "read_aloud") -> None:
    """Show dataset statistics (splits, duration, sample counts).

    Args:
        subset: Dataset subset
    """
    if subset not in VALID_SUBSETS:
        raise ValueError(f"Invalid subset {subset!r}. Must be one of: {VALID_SUBSETS}")
    ctx.run(
        _hf_env_prefix() + f'uv run python -c "'
        f"from datasets import load_dataset; "
        f"ds = load_dataset('CoRal-project/coral-v3', '{subset}', "
        f"cache_dir='{HF_CACHE_DIR}'); "
        f"print('CoRal Dataset Statistics'); "
        f"print('=' * 60); "
        f"for split, data in ds.items(): "
        f"    print(' ', split + ':', len(data), 'samples'); "
        f"    ex = data[0] if len(data) > 0 else None; "
        f'    sr = ex["audio"]["sampling_rate"] if ex else 0; '
        f'    dur = len(ex["audio"]["array"]) / sr if ex else 0; '
        f"    print(f'    Sample rate: {{sr}} Hz, example duration: {{dur:.1f}}s') if ex else None; "
        f'"',
        echo=True,
        pty=not WINDOWS,
    )


@task
def validate(ctx: Context) -> None:
    """Validate audio data integrity."""
    logger.info("Validating audio data...")
    ctx.run(
        _hf_env_prefix() + 'uv run python -c "'
        "import sys; "
        "from datasets import load_dataset; "
        f"ds = load_dataset('CoRal-project/coral-v3', 'read_aloud', "
        f"split='train[:10]', cache_dir='{HF_CACHE_DIR}'); "
        "errors = 0; "
        "for i, item in enumerate(ds): "
        "    audio = item['audio']; "
        "    if audio['sampling_rate'] <= 0: errors += 1; "
        "    if len(audio['array']) == 0: errors += 1; "
        "print('Validated 10 samples,', errors, 'errors found'); "
        "sys.exit(1 if errors > 0 else 0); "
        '"',
        echo=True,
        pty=not WINDOWS,
    )


@task
def preprocess(
    ctx: Context,
    subset: str = "all",
    target: str = "all",
    max_samples: int | None = None,
) -> None:
    """Unified preprocessing: resample + FLAC-encode CoRal-v3 once.

    Args:
        subset: Which subset (read_aloud, conversation, or all)
        target: Output format (fairseq2, universal, or all)
        max_samples: Max samples per split (for testing)
    """
    valid_targets = {"fairseq2", "universal", "all"}
    if target not in valid_targets:
        raise ValueError(f"Invalid target {target!r}. Must be one of: {valid_targets}")
    if subset not in VALID_CONVERT_SUBSETS:
        raise ValueError(f"Invalid subset {subset!r}. Must be one of: {VALID_CONVERT_SUBSETS}")

    cmd = (
        _hf_env_prefix() + f"uv run python -m danish_asr.preprocessing"
        f" --subset {subset}"
        f" --target {target}"
        f" --cache-dir '{HF_CACHE_DIR}'"
    )
    if max_samples is not None:
        cmd += f" --max-samples {max_samples}"
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task(name="check-auth")
def check_auth(ctx: Context) -> None:
    """Verify HuggingFace authentication."""
    try:
        ctx.run(
            _hf_env_prefix() + 'uv run python -c "'
            "from huggingface_hub import whoami; "
            "info = whoami(); "
            "print('Authenticated as:', info['name']); "
            '"',
            echo=True,
            pty=not WINDOWS,
        )
    except Exception:
        logger.error("HuggingFace authentication failed. Run 'huggingface-cli login' or set HF_TOKEN env var.")
        raise


@task(name="download-all")
def download_all(ctx: Context) -> None:
    """Download both CoRal subsets (read_aloud + conversation)."""
    failed = []
    for subset in ("read_aloud", "conversation"):
        try:
            logger.info(f"Downloading {subset}...")
            download(ctx, subset=subset)
        except Exception as e:
            logger.error(f"Failed to download {subset}: {e}")
            failed.append(subset)
    if failed:
        raise RuntimeError(f"Failed to download: {', '.join(failed)}")


@task(name="convert-parquet")
def convert_parquet(
    ctx: Context,
    subset: str = "all",
    output_dir: str = "data/parquet/version=0",
    rows_per_file: int = 5000,
    max_samples: int | None = None,
) -> None:
    """Convert CoRal-v3 to omnilingual ASR Parquet format.

    Args:
        subset: Which subset to convert (read_aloud, conversation, or all)
        output_dir: Output directory for Parquet files
        rows_per_file: Number of samples per Parquet part file
        max_samples: Max samples per split (for testing)
    """
    if subset not in VALID_CONVERT_SUBSETS:
        raise ValueError(f"Invalid subset {subset!r}. Must be one of: {VALID_CONVERT_SUBSETS}")
    safe_output_dir = str(Path(output_dir))
    cmd = (
        _hf_env_prefix() + f"uv run python -m danish_asr.preprocessing"
        f" --subset {subset}"
        f" --target fairseq2"
        f" --fairseq2-dir '{safe_output_dir}'"
        f" --rows-per-file {rows_per_file}"
        f" --cache-dir '{HF_CACHE_DIR}'"
    )
    if max_samples is not None:
        cmd += f" --max-samples {max_samples}"
    ctx.run(cmd, echo=True, pty=not WINDOWS)
