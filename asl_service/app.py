from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .assets import ensure_wlasl_assets
from .config import CHECKPOINT_PATH, MAX_UPLOAD_BYTES, NUM_CLASSES
from .inference import load_labels, predict_video

local_origins = ["http://localhost:8000", "http://127.0.0.1:8000"]
configured_origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv("DEXTER_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]
max_concurrent_inferences = max(
    1, int(os.getenv("WLASL_MAX_CONCURRENT_INFERENCES", "1"))
)
inference_queue_timeout = max(
    1.0, float(os.getenv("WLASL_INFERENCE_QUEUE_TIMEOUT_SECONDS", "30"))
)
inference_semaphore = asyncio.Semaphore(max_concurrent_inferences)

app = FastAPI(title="Dexter WLASL300 inference", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[*local_origins, *configured_origins],
    allow_origin_regex=os.getenv(
        "DEXTER_ALLOWED_ORIGIN_REGEX",
        r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    ),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
async def service_info() -> dict:
    return {
        "service": "Dexter WLASL300 inference",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health")
@app.get("/api/asl/health")
async def health() -> dict:
    labels_ready = False
    try:
        await ensure_wlasl_assets()
        labels_ready = len(load_labels()) == NUM_CLASSES
    except (OSError, RuntimeError, ValueError):
        pass
    return {
        "status": "ready" if CHECKPOINT_PATH.exists() and labels_ready else "missing_model",
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpointFound": CHECKPOINT_PATH.exists(),
        "labelsFound": labels_ready,
        "classes": NUM_CLASSES,
    }


@app.post("/predict")
@app.post("/api/asl/predict")
async def predict(video: UploadFile = File(...)) -> dict:
    suffix = Path(video.filename or "clip.webm").suffix or ".webm"
    temporary_path = None
    try:
        await ensure_wlasl_assets()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
            temporary_path = temporary.name
            uploaded_bytes = 0
            while chunk := await video.read(1024 * 1024):
                uploaded_bytes += len(chunk)
                if uploaded_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Video exceeds the {MAX_UPLOAD_BYTES}-byte upload limit.",
                    )
                temporary.write(chunk)
        try:
            await asyncio.wait_for(
                inference_semaphore.acquire(), timeout=inference_queue_timeout
            )
        except asyncio.TimeoutError as error:
            raise HTTPException(
                status_code=429,
                detail="ASL inference is busy. Please retry shortly.",
            ) from error
        try:
            return await asyncio.to_thread(predict_video, temporary_path)
        finally:
            inference_semaphore.release()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        await video.close()
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)
