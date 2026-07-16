from __future__ import annotations

from functools import lru_cache

import torch

from .config import (
    CHECKPOINT_PATH,
    CLASS_LIST_PATH,
    FRAME_COUNT,
    INPUT_MODE,
    LANDMARK_COUNT,
    MIN_MOTION_SCORE,
    MIN_TOP1_CONFIDENCE,
    MIN_TOP1_MARGIN,
    MODEL_NAME,
    NUM_CLASSES,
    TEMPERATURE,
)
from .phonssm import PhonSSM, PhonSSMConfig
from .phonssm_preprocessing import prepare_landmarks


def load_labels() -> list[str]:
    labels: dict[int, str] = {}
    for line in CLASS_LIST_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, gloss = line.split(maxsplit=1)
        class_index = int(class_id)
        labels[class_index] = gloss
    if sorted(labels) != list(range(NUM_CLASSES)):
        raise RuntimeError("The PhonSSM WLASL300 label file must contain IDs 0-299.")
    return [labels[index] for index in range(NUM_CLASSES)]


def _model_state(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise RuntimeError("The PhonSSM checkpoint must contain a state dictionary.")
    if "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise RuntimeError("The PhonSSM checkpoint contains no model weights.")
    return {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
        if isinstance(key, str)
    }


@lru_cache(maxsize=1)
def load_model() -> tuple[PhonSSM, torch.device]:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"PhonSSM WLASL300 checkpoint not found at {CHECKPOINT_PATH}. "
            "Follow asl_service/models/README.md."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = PhonSSMConfig(
        input_mode=INPUT_MODE,
        num_landmarks=LANDMARK_COUNT,
        num_frames=FRAME_COUNT,
        num_signs=NUM_CLASSES,
        temperature=TEMPERATURE,
    )
    model = PhonSSM(config)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    try:
        model.load_state_dict(_model_state(checkpoint), strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "The installed checkpoint is not the 300-class pose+hands PhonSSM model."
        ) from error
    model.to(device).eval()
    return model, device


def predict_landmarks(frames: list, top_k: int = 3) -> dict:
    labels = load_labels()
    model, device = load_model()
    landmarks, motion_score = prepare_landmarks(frames)

    with torch.inference_mode():
        outputs = model(landmarks.to(device))
        probabilities = torch.softmax(outputs["logits"][0], dim=0)
        scores, indices = probabilities.topk(min(max(2, top_k), NUM_CLASSES))

    predictions = [
        {"word": labels[index], "confidence": float(score)}
        for score, index in zip(scores.cpu().tolist(), indices.cpu().tolist())
    ]
    margin = predictions[0]["confidence"] - predictions[1]["confidence"]
    if motion_score < MIN_MOTION_SCORE:
        rejection_reason = "no_motion"
    elif predictions[0]["confidence"] < MIN_TOP1_CONFIDENCE:
        rejection_reason = "low_confidence"
    elif margin < MIN_TOP1_MARGIN:
        rejection_reason = "ambiguous"
    else:
        rejection_reason = None

    return {
        "model": MODEL_NAME,
        "detected": rejection_reason is None,
        "word": predictions[0]["word"],
        "confidence": predictions[0]["confidence"],
        "margin": margin,
        "motionScore": motion_score,
        "rejectionReason": rejection_reason,
        "predictions": predictions,
    }
