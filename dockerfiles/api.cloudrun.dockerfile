FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app
COPY uv.lock pyproject.toml README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-install-project
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --index-url https://download.pytorch.org/whl/cpu torch
COPY src/ src/
COPY configs/ configs/
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --no-deps .

FROM python:3.12-slim-bookworm AS runtime
WORKDIR /app
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/configs /app/configs
RUN mkdir -p /app/data /app/outputs/checkpoints
ENV PATH="/app/.venv/bin:$PATH"
ENV CONFIG_PATH="/app/configs/config.yaml"
ENV MODEL_PATH="/app/outputs/checkpoints/model.pt"

EXPOSE 8080
CMD ["bash", "-c", "uvicorn danish_asr.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
