FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS python-base

ENV DEBIAN_FRONTEND=noninteractive
RUN --mount=type=cache,target=/var/cache/apt apt-get update && \
    apt-get install --no-install-recommends -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install --no-install-recommends -y python3.12 python3.12-venv
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

FROM python-base AS builder
RUN --mount=type=cache,target=/var/cache/apt apt-get update && \
    apt-get install --no-install-recommends -y python3.12-dev build-essential gcc
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY uv.lock pyproject.toml README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-install-project
RUN --mount=type=cache,target=/root/.cache/uv uv pip install torch --index-url https://download.pytorch.org/whl/cu124
COPY src/ src/
COPY configs/ configs/
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen

FROM python-base AS runtime
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
RUN mkdir -p data outputs/checkpoints outputs/logs
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/configs /app/configs
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["uv", "run", "--frozen", "python", "-u", "-m", "danish_asr.train"]
CMD []
