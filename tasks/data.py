"""Data management tasks for Danish ASR."""

import os

from invoke import Context, task
from loguru import logger

WINDOWS = os.name == "nt"
PROJECT_NAME = "danish_asr"


@task
def download(ctx: Context, subset: str = "read_aloud") -> None:
    """Download CoRal Danish speech dataset via HuggingFace.

    Args:
        subset: Dataset subset (read_aloud or conversational)
    """
    logger.info(f"Downloading CoRal dataset (subset={subset})...")
    ctx.run(
        f'uv run python -c "'
        f"from datasets import load_dataset; "
        f"ds = load_dataset('CoRal-project/coral-v2', '{subset}', trust_remote_code=True); "
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
    ctx.run(
        f'uv run python -c "'
        f"from datasets import load_dataset; "
        f"ds = load_dataset('CoRal-project/coral-v2', '{subset}', trust_remote_code=True); "
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
        "ds = load_dataset('CoRal-project/coral-v2', 'read_aloud', trust_remote_code=True, split='train[:10]'); "
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
def preprocess(ctx: Context) -> None:
    """Preprocess audio data (resample, normalize)."""
    logger.info("Audio preprocessing is handled on-the-fly by CoRalDataset.")
    logger.info("No separate preprocessing step needed.")
