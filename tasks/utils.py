"""Utility and maintenance tasks."""

import os

from invoke import Context, task

WINDOWS = os.name == "nt"
HPC_USER = os.environ.get("HPC_USER", "s204696")
HPC_LOGIN = f"{HPC_USER}@login.hpc.dtu.dk"
HPC_TRANSFER = f"{HPC_USER}@transfer.gbar.dtu.dk"


@task
def clean_pyc(ctx: Context) -> None:
    """Remove Python bytecode files."""
    print("Removing Python bytecode files...")
    ctx.run("find . -type f -name '*.py[co]' -delete", warn=True, echo=True, pty=not WINDOWS)
    ctx.run("find . -type d -name '__pycache__' -delete", warn=True, echo=True, pty=not WINDOWS)
    print("✓ Python bytecode cleaned")


@task
def clean_build(ctx: Context) -> None:
    """Remove build artifacts."""
    print("Removing build artifacts...")
    ctx.run("rm -rf build/ dist/ *.egg-info .eggs/", warn=True, echo=True, pty=not WINDOWS)
    print("✓ Build artifacts cleaned")


@task
def clean_test(ctx: Context) -> None:
    """Remove test and coverage artifacts."""
    print("Removing test artifacts...")
    ctx.run("rm -rf .pytest_cache/ .coverage htmlcov/ .tox/", warn=True, echo=True, pty=not WINDOWS)
    print("✓ Test artifacts cleaned")


@task
def clean_outputs(ctx: Context) -> None:
    """Remove training outputs and logs."""
    print("Removing training outputs...")
    ctx.run("rm -rf outputs/logs/ wandb/", warn=True, echo=True, pty=not WINDOWS)
    print("✓ Training outputs cleaned")


@task(pre=[clean_pyc, clean_build, clean_test])
def clean_all(ctx: Context) -> None:
    """Clean all build, test, and Python artifacts."""
    print("\n✓ All artifacts cleaned!")


@task
def env_info(ctx: Context) -> None:
    """Show environment information."""
    print("Environment Information")
    print("=" * 60)
    print("\nPython:")
    ctx.run("uv run python --version", echo=True, pty=not WINDOWS)
    print("\nuv:")
    ctx.run("uv --version", echo=True, pty=not WINDOWS)
    print("\nGit:")
    ctx.run("git --version", echo=True, pty=not WINDOWS)
    print("\nDocker:")
    ctx.run("docker --version", warn=True, echo=True, pty=not WINDOWS)
    print("\nGPU:")
    result = ctx.run(
        "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader", warn=True, hide=True, pty=not WINDOWS
    )
    if result and result.ok:
        print(f"  {result.stdout.strip()}")
    else:
        print("  No NVIDIA GPU detected")
    print("\n" + "=" * 60)


@task
def check_gpu(ctx: Context) -> None:
    """Check GPU availability and CUDA support."""
    print("Checking GPU availability...\n")
    print("1. NVIDIA GPU:")
    ctx.run("nvidia-smi", warn=True, echo=True, pty=not WINDOWS)
    print("\n2. PyTorch CUDA:")
    ctx.run(
        'uv run python -c \'import torch; print(f"CUDA Available: {torch.cuda.is_available()}"); '
        'print(f"CUDA Version: {torch.version.cuda if torch.cuda.is_available() else None}"); '
        'print(f"Device Count: {torch.cuda.device_count()}")\'',
        echo=True,
        pty=not WINDOWS,
    )


@task
def port_check(ctx: Context, port: int = 8000) -> None:
    """Check if a port is in use."""
    print(f"Checking port {port}...")
    result = ctx.run(f"lsof -i :{port}", warn=True, echo=True, pty=not WINDOWS)
    if not result or not result.ok:
        print(f"✓ Port {port} is free")


@task
def kill_port(ctx: Context, port: int = 8000) -> None:
    """Kill process using a specific port."""
    print(f"Killing process on port {port}...")
    ctx.run(f"lsof -ti :{port} | xargs kill -9", warn=True, echo=True, pty=not WINDOWS)
    print(f"✓ Process on port {port} killed")


@task
def hpc_sftp(ctx: Context) -> None:
    """Open SFTP session to DTU HPC transfer node (requires VPN or DTU network)."""
    ctx.run(f"sftp {HPC_TRANSFER}", pty=True)


@task
def hpc_ssh(ctx: Context) -> None:
    """SSH into DTU HPC login node (requires VPN or DTU network)."""
    ctx.run(f"ssh {HPC_LOGIN}", pty=True)


@task
def dtu_vpn(ctx: Context) -> None:
    """Connect to DTU VPN via openconnect (requires sudo + MFA)."""
    print("Connecting to DTU VPN...")
    print(f"  Username: {HPC_USER}@dtu.dk  |  MFA: Microsoft Authenticator\n")
    ctx.run(
        'sudo openconnect vpn.dtu.dk --useragent "AnyConnect Linux_64 5.1.10.238" --version-string "5.1.10.238"',
        pty=True,
    )
