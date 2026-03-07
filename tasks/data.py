"""Data management tasks for Danish ASR."""

import os

from invoke import Context, task
from loguru import logger

WINDOWS = os.name == "nt"
PROJECT_NAME = "danish_asr"
VALID_SUBSETS = {"read_aloud", "conversation"}


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
        f'uv run python -c "'
        f"from datasets import load_dataset; "
        f"ds = load_dataset('CoRal-project/coral-v3', '{subset}', trust_remote_code=True); "
        f"print(f'Downloaded: {{{{len(ds)}}}} splits'); "
        f"[print(f'  {{{{k}}}}: {{{{len(v)}}}} samples') for k, v in ds.items()]"
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
        f'uv run python -c "'
        f"from datasets import load_dataset; "
        f"ds = load_dataset('CoRal-project/coral-v3', '{subset}', trust_remote_code=True); "
        f"print('CoRal Dataset Statistics'); "
        f"print('=' * 60); "
        f"for split, data in ds.items(): "
        f"    print(f'  {{{{split}}}}: {{{{len(data)}}}} samples'); "
        f"    if len(data) > 0: "
        f"        example = data[0]; "
        f'        sr = example["audio"]["sampling_rate"]; '
        f'        dur = len(example["audio"]["array"]) / sr; '
        f"        print(f'    Sample rate: {{{{sr}}}} Hz'); "
        f"        print(f'    Example duration: {{{{dur:.1f}}}}s'); "
        f'"',
        echo=True,
        pty=not WINDOWS,
    )


@task
def validate(ctx: Context) -> None:
    """Validate audio data integrity."""
    logger.info("Validating audio data...")
    ctx.run(
        'uv run python -c "'
        "from datasets import load_dataset; "
        "ds = load_dataset('CoRal-project/coral-v3', 'read_aloud', trust_remote_code=True, split='train[:10]'); "
        "errors = 0; "
        "for i, item in enumerate(ds): "
        "    audio = item['audio']; "
        "    if audio['sampling_rate'] <= 0: errors += 1; "
        "    if len(audio['array']) == 0: errors += 1; "
        "print(f'Validated 10 samples, {{errors}} errors found'); "
        '"',
        echo=True,
        pty=not WINDOWS,
    )


@task
def preprocess(ctx: Context, target: str = "whisper") -> None:
    """Preprocess audio data.

    Args:
        target: Processing target ('whisper' for on-the-fly, 'omniasr' for Parquet conversion)
    """
    if target == "whisper":
        logger.info("Whisper preprocessing is handled on-the-fly by CoRalDataset.")
        logger.info("No separate preprocessing step needed.")
    elif target == "omniasr":
        logger.info("Run 'invoke data.convert-parquet' for omnilingual ASR Parquet conversion.")
    else:
        raise ValueError(f"Invalid target {target!r}. Must be 'whisper' or 'omniasr'.")


@task(name="check-auth")
def check_auth(ctx: Context) -> None:
    """Verify HuggingFace authentication."""
    ctx.run(
        'uv run python -c "'
        "from huggingface_hub import whoami; "
        "info = whoami(); "
        "print(f'Authenticated as: {info[\"name\"]}'); "
        '"',
        echo=True,
        pty=not WINDOWS,
    )


@task(name="download-all")
def download_all(ctx: Context) -> None:
    """Download both CoRal subsets (read_aloud + conversation)."""
    for subset in ("read_aloud", "conversation"):
        logger.info(f"Downloading {subset}...")
        download(ctx, subset=subset)


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
    cmd = (
        f"uv run python scripts/convert_coral_to_parquet.py"
        f" --subset {subset}"
        f" --output-dir {output_dir}"
        f" --rows-per-file {rows_per_file}"
    )
    if max_samples is not None:
        cmd += f" --max-samples {max_samples}"
    ctx.run(cmd, echo=True, pty=not WINDOWS)
