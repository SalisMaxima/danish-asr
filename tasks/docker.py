"""Docker container and image management tasks."""

import os
from pathlib import Path

from invoke import Context, task
from loguru import logger

WINDOWS = os.name == "nt"


def check_docker_available(ctx: Context) -> bool:
    """Check if Docker is installed and running."""
    try:
        result = ctx.run("docker info", hide=True, warn=True, pty=not WINDOWS)
        return result.ok
    except FileNotFoundError:
        logger.error("Docker binary not found — is Docker installed and on PATH?")
        return False
    except Exception as exc:
        logger.error(f"Docker availability check failed: {exc}")
        return False


@task
def build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images (CPU versions)."""
    if not check_docker_available(ctx):
        logger.error("Docker is not running or not accessible. Check: docker info")
        raise SystemExit(1)
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not WINDOWS
    )


@task
def build_cuda(ctx: Context, progress: str = "plain") -> None:
    """Build CUDA-enabled training docker image."""
    if not check_docker_available(ctx):
        logger.error("Docker is not running or not accessible. Check: docker info")
        raise SystemExit(1)
    ctx.run(
        f"docker build -t train-cuda:latest . -f dockerfiles/train_cuda.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def train(ctx: Context, entity: str = "", cuda: bool = True, args: str = "") -> None:
    """Run Wav2Vec2/Whisper baseline training in Docker (not the omniasr fairseq2 pipeline)."""
    if not check_docker_available(ctx):
        logger.error("Docker is not running or not accessible. Check: docker info")
        raise SystemExit(1)
    cwd = Path.cwd()
    image = "train-cuda:latest" if cuda else "train:latest"
    gpu_flag = "--gpus all" if cuda else ""
    train_args = f"wandb.entity={entity} {args}".strip() if entity else args
    wandb_api_key = os.environ.get("WANDB_API_KEY", "")
    if not wandb_api_key:
        logger.warning("WANDB_API_KEY is not set — training metrics will not be uploaded to W&B.")
    ctx.run(
        f"docker run --rm {gpu_flag} --shm-size=2g "
        f"-v {cwd}/data:/app/data -v {cwd}/models:/app/models "
        f"-e WANDB_API_KEY={wandb_api_key} {image} {train_args}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def api(ctx: Context, port: int = 8000) -> None:
    """Run API in Docker container."""
    if not check_docker_available(ctx):
        logger.error("Docker is not running or not accessible. Check: docker info")
        raise SystemExit(1)
    cwd = Path.cwd()
    ctx.run(f"docker run -p {port}:8000 -v {cwd}/models:/app/models api:latest", echo=True, pty=not WINDOWS)


@task
def clean(ctx: Context, all: bool = False) -> None:
    """Clean up Docker images and containers."""
    if not check_docker_available(ctx):
        logger.error("Docker is not running or not accessible. Check: docker info")
        raise SystemExit(1)
    print("Removing stopped containers...")
    ctx.run("docker container prune -f", echo=True, pty=not WINDOWS)
    if all:
        print("Removing all unused images...")
        ctx.run("docker image prune -a -f", echo=True, pty=not WINDOWS)
    else:
        print("Removing dangling images...")
        ctx.run("docker image prune -f", echo=True, pty=not WINDOWS)
    print("Removing unused volumes...")
    ctx.run("docker volume prune -f", echo=True, pty=not WINDOWS)
    print("\n✓ Docker cleanup complete!")
