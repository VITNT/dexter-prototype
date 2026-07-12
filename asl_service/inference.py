from __future__ import annotations

import sys
from functools import lru_cache

import torch

from .config import CHECKPOINT_PATH, CLASS_LIST_PATH, NUM_CLASSES, OFFICIAL_I3D_ROOT
from .preprocessing import prepare_video


def load_labels() -> list[str]:
    labels: dict[int, str] = {}
    for line in CLASS_LIST_PATH.read_text(encoding="utf-8").splitlines():
        class_id, gloss = line.split(maxsplit=1)
        class_index = int(class_id)
        if class_index < NUM_CLASSES:
            labels[class_index] = gloss
    if sorted(labels) != list(range(NUM_CLASSES)):
        raise RuntimeError("The official WLASL class list does not contain IDs 0-299.")
    return [labels[index] for index in range(NUM_CLASSES)]


@lru_cache(maxsize=1)
def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"WLASL300 checkpoint not found at {CHECKPOINT_PATH}. "
            "Follow asl_service/models/README.md."
        )
    if not OFFICIAL_I3D_ROOT.exists():
        raise FileNotFoundError("The official WLASL I3D source is not installed.")

    sys.path.insert(0, str(OFFICIAL_I3D_ROOT))
    from pytorch_i3d import InceptionI3d

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InceptionI3d(400, in_channels=3)
    model.replace_logits(NUM_CLASSES)
    state = torch.load(CHECKPOINT_PATH, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    return model, device


def predict_video(video_path: str, top_k: int = 3) -> dict:
    labels = load_labels()
    model, device = load_model()
    clip = prepare_video(video_path).to(device)

    with torch.inference_mode():
        per_frame_logits = model(clip)
        logits = per_frame_logits.max(dim=2).values[0]
        probabilities = torch.softmax(logits, dim=0)
        scores, indices = probabilities.topk(min(top_k, NUM_CLASSES))

    predictions = [
        {"word": labels[index], "confidence": float(score)}
        for score, index in zip(scores.cpu().tolist(), indices.cpu().tolist())
    ]
    return {
        "word": predictions[0]["word"],
        "confidence": predictions[0]["confidence"],
        "predictions": predictions,
    }

