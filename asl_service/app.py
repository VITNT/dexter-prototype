from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import CHECKPOINT_PATH, NUM_CLASSES
from .inference import load_labels, predict_video

app = FastAPI(title="Dexter WLASL300 inference", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    labels_ready = False
    try:
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
async def predict(video: UploadFile = File(...)) -> dict:
    suffix = Path(video.filename or "clip.webm").suffix or ".webm"
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
            temporary_path = temporary.name
            while chunk := await video.read(1024 * 1024):
                temporary.write(chunk)
        return predict_video(temporary_path)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        await video.close()
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)

