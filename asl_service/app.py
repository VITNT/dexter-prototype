from __future__ import annotations

import json

from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import (
    CHECKPOINT_PATH,
    FEATURE_COUNT,
    FRAME_COUNT,
    MODEL_NAME,
    NUM_CLASSES,
)
from .inference import load_labels, load_model, predict_landmarks

app = FastAPI(title="Dexter PhonSSM WLASL300 inference", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    labels_ready = False
    model_ready = False
    model_error = None
    try:
        labels_ready = len(load_labels()) == NUM_CLASSES
    except (OSError, RuntimeError, ValueError):
        pass
    if CHECKPOINT_PATH.exists():
        try:
            load_model()
            model_ready = True
        except (OSError, RuntimeError, ValueError) as error:
            model_error = str(error)
    return {
        "status": "ready" if model_ready and labels_ready else "missing_model",
        "model": MODEL_NAME,
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpointFound": CHECKPOINT_PATH.exists(),
        "checkpointValid": model_ready,
        "labelsFound": labels_ready,
        "classes": NUM_CLASSES,
        "inputShape": [FRAME_COUNT, FEATURE_COUNT],
        "error": model_error,
    }


@app.post("/predict")
async def predict(landmarks: str = Form(...)) -> dict:
    try:
        frames = json.loads(landmarks)
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=422, detail="landmarks must be valid JSON.") from error
    try:
        return predict_landmarks(frames)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (RuntimeError, TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
