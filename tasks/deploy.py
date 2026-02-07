"""Deployment and serving tasks."""

import os
from pathlib import Path

from invoke import Context, task
from loguru import logger

WINDOWS = os.name == "nt"
PROJECT_NAME = "danish_asr"


@task
def promote_model(ctx: Context, model_path: str = "", project: str = "", entity: str = "") -> None:
    """Promote model to production in W&B Model Registry."""
    if not model_path:
        logger.error("ERROR: --model-path is required")
        return
    if not Path(model_path).exists():
        logger.error(f"ERROR: Model not found: {model_path}")
        return
    cmd = f"uv run python -m {PROJECT_NAME}.promote_model --model-path {model_path}"
    if project:
        cmd += f" --wandb-project {project}"
    if entity:
        cmd += f" --wandb-entity {entity}"
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def export_onnx(ctx: Context, run_dir: str = "", output: str = "", opset: int = 17) -> None:
    """Export trained model to ONNX format for production deployment."""
    if not run_dir:
        logger.error("ERROR: --run-dir is required")
        return
    if not Path(run_dir).exists():
        logger.error(f"ERROR: Run directory not found: {run_dir}")
        return
    cmd = f"uv run python -m {PROJECT_NAME}.onnx_export --run-dir {run_dir}"
    if output:
        cmd += f" --output {output}"
    if opset != 17:
        cmd += f" --opset {opset}"
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def api(ctx: Context, reload: bool = True, port: int = 8000) -> None:
    """Run FastAPI development server."""
    reload_flag = " --reload" if reload else ""
    ctx.run(
        f"uv run uvicorn {PROJECT_NAME}.api:app --host 0.0.0.0 --port {port}{reload_flag}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def frontend(ctx: Context, port: int = 8501, api_url: str = "http://localhost:8000") -> None:
    """Run Streamlit frontend application."""
    os.environ["API_URL"] = api_url
    ctx.run(
        f"uv run streamlit run src/{PROJECT_NAME}/frontend/pages/home.py --server.port {port}",
        echo=True,
        pty=not WINDOWS,
    )
