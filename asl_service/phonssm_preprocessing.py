from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from .config import FEATURE_COUNT, FRAME_COUNT, LANDMARK_COUNT


def sample_landmark_indices(
    source_count: int,
    target_count: int = FRAME_COUNT,
) -> np.ndarray:
    """Match the reference WLASL temporal preprocessing contract."""
    if source_count < 1:
        raise ValueError("The landmark sequence contains no frames.")
    if target_count < 1:
        raise ValueError("target_count must be positive.")
    if source_count <= target_count:
        return np.minimum(np.arange(target_count), source_count - 1).astype(np.int64)
    return np.linspace(0, source_count - 1, target_count).astype(np.int64)


def normalize_pose_hands(landmarks: np.ndarray) -> np.ndarray:
    """Center each frame at the shoulders and scale by shoulder width.

    This intentionally follows ``training/preprocess_wlasl_full.py`` from the
    PhonSSM repository so inference sees the same values as its WLASL training
    data, including its max-distance fallback when shoulders are unavailable.
    """
    if landmarks.ndim != 3 or landmarks.shape[1:] != (LANDMARK_COUNT, 3):
        raise ValueError(f"Expected landmarks shaped [frames, {LANDMARK_COUNT}, 3].")

    normalized: list[np.ndarray] = []
    for frame in landmarks:
        if np.all(frame == 0):
            normalized.append(np.zeros_like(frame))
            continue

        left_shoulder = frame[11]
        right_shoulder = frame[12]
        center = (left_shoulder + right_shoulder) / 2
        centered = frame - center

        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        if shoulder_width < 1e-6:
            distances = np.linalg.norm(centered, axis=1)
            maximum = float(distances.max())
            scale = maximum if maximum > 1e-6 else 1.0
        else:
            scale = float(shoulder_width)
        normalized.append(centered / scale)

    return np.asarray(normalized, dtype=np.float32)


def landmark_motion_score(landmarks: np.ndarray) -> float:
    """Return robust mean per-frame movement for visible hand landmarks."""
    if len(landmarks) < 2:
        return 0.0

    hands = landmarks[:, 33:75]
    transition_scores: list[float] = []
    for previous, current in zip(hands, hands[1:]):
        previous_present = np.any(previous != 0, axis=1)
        current_present = np.any(current != 0, axis=1)
        visible = previous_present & current_present
        if np.any(visible):
            displacement = np.linalg.norm(current[visible] - previous[visible], axis=1)
            transition_scores.append(float(displacement.mean()))
        else:
            transition_scores.append(0.0)

    scores = np.asarray(transition_scores, dtype=np.float32)
    active_count = max(1, (len(scores) + 1) // 2)
    return float(np.mean(np.partition(scores, -active_count)[-active_count:]))


def prepare_landmarks(
    frames: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[torch.Tensor, float]:
    """Validate, normalize, and sample browser landmarks for PhonSSM."""
    array = np.asarray(frames, dtype=np.float32)
    if array.ndim == 2 and array.shape[1] == FEATURE_COUNT:
        array = array.reshape(len(array), LANDMARK_COUNT, 3)
    elif array.ndim != 3 or array.shape[1:] != (LANDMARK_COUNT, 3):
        raise ValueError(
            f"Landmarks must be [frames, {FEATURE_COUNT}] or "
            f"[frames, {LANDMARK_COUNT}, 3]."
        )
    if len(array) == 0:
        raise ValueError("The landmark sequence contains no frames.")
    if not np.isfinite(array).all():
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

    motion_score = landmark_motion_score(array)
    normalized = normalize_pose_hands(array)
    indices = sample_landmark_indices(len(normalized))
    sampled = normalized[indices].reshape(FRAME_COUNT, FEATURE_COUNT)
    return torch.from_numpy(sampled).unsqueeze(0), motion_score
