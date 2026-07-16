from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from .config import INPUT_FEATURE_COUNT, INPUT_LANDMARK_COUNT, MODEL_LANDMARK_COUNT


# AI4Bharat/OpenHands PoseSelect preset ``mediapipe_holistic_minimal_27``.
SELECTED_LANDMARK_INDICES = np.asarray(
    [0, 2, 5, 11, 12, 13, 14, 33, 37, 38, 41, 42, 45, 46, 49, 50, 53, 54, 58, 59, 62, 63, 66, 67, 70, 71, 74],
    dtype=np.int64,
)
LEFT_SHOULDER_INDEX = 3
RIGHT_SHOULDER_INDEX = 4


def normalize_selected_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Match OpenHands clip-level ``CenterAndScaleNormalize`` exactly."""
    if landmarks.ndim != 3 or landmarks.shape[1:] != (MODEL_LANDMARK_COUNT, 2):
        raise ValueError(
            f"Expected selected landmarks shaped [frames, {MODEL_LANDMARK_COUNT}, 2]."
        )
    left_shoulders = landmarks[:, LEFT_SHOULDER_INDEX]
    right_shoulders = landmarks[:, RIGHT_SHOULDER_INDEX]
    center = np.mean((left_shoulders + right_shoulders) / 2, axis=0)
    mean_distance = float(
        np.mean(np.linalg.norm(left_shoulders - right_shoulders, axis=-1))
    )
    if not np.isfinite(mean_distance) or mean_distance <= 1e-12:
        return landmarks.copy()
    return ((landmarks - center) / mean_distance).astype(np.float32)


def landmark_motion_score(landmarks: np.ndarray) -> float:
    """Return robust mean movement for hand landmarks in a holistic window."""
    if len(landmarks) < 2:
        return 0.0
    hands = landmarks[:, 33:75]
    transitions: list[float] = []
    for previous, current in zip(hands, hands[1:]):
        visible = np.any(previous != 0, axis=1) & np.any(current != 0, axis=1)
        if np.any(visible):
            movement = np.linalg.norm(current[visible, :2] - previous[visible, :2], axis=1)
            transitions.append(float(movement.mean()))
        else:
            transitions.append(0.0)
    scores = np.asarray(transitions, dtype=np.float32)
    active_count = max(1, (len(scores) + 1) // 2)
    return float(np.mean(np.partition(scores, -active_count)[-active_count:]))


def prepare_landmarks(
    frames: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[torch.Tensor, float]:
    """Convert browser holistic landmarks to OpenHands ``[N,C,T,V]`` input."""
    array = np.asarray(frames, dtype=np.float32)
    if array.ndim == 2 and array.shape[1] == INPUT_FEATURE_COUNT:
        array = array.reshape(len(array), INPUT_LANDMARK_COUNT, 3)
    elif array.ndim != 3 or array.shape[1:] != (INPUT_LANDMARK_COUNT, 3):
        raise ValueError(
            f"Landmarks must be [frames, {INPUT_FEATURE_COUNT}] or "
            f"[frames, {INPUT_LANDMARK_COUNT}, 3]."
        )
    if len(array) == 0:
        raise ValueError("The landmark sequence contains no frames.")
    if not np.isfinite(array).all():
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

    motion_score = landmark_motion_score(array)
    selected_xy = array[:, SELECTED_LANDMARK_INDICES, :2]
    normalized = normalize_selected_landmarks(selected_xy)
    model_input = np.transpose(normalized, (2, 0, 1)).copy()
    return torch.from_numpy(model_input).unsqueeze(0), motion_score
