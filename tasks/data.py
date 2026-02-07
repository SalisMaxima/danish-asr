"""Data management tasks."""

import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "danish_asr"


@task
def download(ctx: Context) -> None:
    """Download dataset."""
    ctx.run(f"uv run python -m {PROJECT_NAME}.data download", echo=True, pty=not WINDOWS)


@task
def preprocess(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"uv run python -m {PROJECT_NAME}.data preprocess", echo=True, pty=not WINDOWS)


@task
def stats(ctx: Context) -> None:
    """Show dataset statistics."""
    ctx.run(f"uv run python -m {PROJECT_NAME}.stats", echo=True, pty=not WINDOWS)


@task
def validate(ctx: Context) -> None:
    """Validate data integrity and structure."""
    ctx.run(f"uv run python -m {PROJECT_NAME}.validate", echo=True, pty=not WINDOWS)
