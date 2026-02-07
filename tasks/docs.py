"""Documentation building and serving tasks."""

import os

from invoke import Context, task

WINDOWS = os.name == "nt"


@task
def build(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve(ctx: Context) -> None:
    """Serve documentation locally."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
