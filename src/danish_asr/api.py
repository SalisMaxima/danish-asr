"""FastAPI prediction API."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import psutil
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from prometheus_client import Counter, Gauge, make_asgi_app
from prometheus_fastapi_instrumentator import Instrumentator

from danish_asr.feedback_store import create_feedback_store

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "outputs/checkpoints/model.pt"))
LOAD_MODEL = os.environ.get("LOAD_MODEL", "1") == "1"

model: torch.nn.Module | None = None
feedback_store = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
PROC = psutil.Process()
system_cpu_percent = Gauge("system_cpu_percent", "System CPU utilization percent")
system_memory_percent = Gauge("system_memory_percent", "System memory utilization percent")
prediction_counter = Counter("prediction_total", "Total predictions")
prediction_confidence = Gauge("prediction_confidence", "Prediction confidence")


async def _metrics_loop(stop_event: asyncio.Event, interval_s: float = 5.0) -> None:
    psutil.cpu_percent(interval=None)
    while not stop_event.is_set():
        system_cpu_percent.set(psutil.cpu_percent(interval=None))
        system_memory_percent.set(psutil.virtual_memory().percent)
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=interval_s)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global model, feedback_store

    # Initialize feedback store
    try:
        feedback_store = create_feedback_store()
        logger.info("Feedback store initialized: %s", type(feedback_store).__name__)
    except Exception as exc:
        logger.warning("Failed to initialize feedback store: %s", exc)

    # TODO: Load your model here
    # from danish_asr.model import build_model
    # model = build_model(cfg)
    # model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    # model.to(DEVICE)
    # model.eval()

    stop_event = asyncio.Event()
    task = asyncio.create_task(_metrics_loop(stop_event))
    yield
    stop_event.set()
    task.cancel()
    with suppress(Exception):
        await task
    model = None


error_counter = Counter("prediction_error", "Number of prediction errors")

app = FastAPI(
    title="Danish ASR API",
    description="ML prediction service for Danish ASR",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/metrics", make_asgi_app())
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


@app.get("/health")
def health() -> dict:
    """Check API health status."""
    return {
        "status": "healthy",
        "ok": True,
        "device": str(DEVICE),
        "model_loaded": model is not None,
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.post("/predict")
async def predict(file: Annotated[UploadFile, File(...)]) -> dict:
    """Run prediction on uploaded file."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    content = await file.read()
    # TODO: Implement your prediction logic here using `content`
    prediction_counter.inc()

    return {
        "prediction": {"class": "unknown", "confidence": 0.0},
        "metadata": {"device": str(DEVICE), "timestamp": datetime.now(UTC).isoformat(), "input_size": len(content)},
    }


@app.post("/feedback")
async def feedback(
    predicted_class: Annotated[str, Form(...)],
    is_correct: Annotated[bool, Form(...)] = True,
    correct_class: Annotated[str | None, Form()] = None,
) -> dict:
    """Save user feedback for model improvement."""
    if feedback_store is None:
        raise HTTPException(status_code=503, detail="Feedback store not available")

    feedback_id = feedback_store.save_feedback(
        image_path="",
        predicted_class=predicted_class,
        predicted_confidence=0.0,
        is_correct=is_correct,
        correct_class=correct_class,
    )
    return {"feedback_id": feedback_id, "timestamp": datetime.now(UTC).isoformat()}


@app.get("/feedback/stats")
def feedback_stats() -> dict:
    """Get feedback statistics."""
    if feedback_store is None:
        raise HTTPException(status_code=503, detail="Feedback store not available")
    return feedback_store.get_stats()
