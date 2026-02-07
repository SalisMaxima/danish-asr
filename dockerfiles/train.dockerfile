FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY uv.lock pyproject.toml README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-install-project
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY src/ src/
COPY configs/ configs/
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen

RUN mkdir -p outputs/checkpoints outputs/logs outputs/reports outputs/profiling

ENTRYPOINT ["uv", "run", "python", "-m", "danish_asr.train"]
CMD []
