from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch

from .config import (
    INPUT_FEATURE_COUNT,
    INPUT_LANDMARK_COUNT,
    MODEL_FRAME_COUNT,
    MODEL_LANDMARK_COUNT,
)


# OpenPose BODY_25 points retained by the original WLASL TGCN pipeline are
# [nose, neck, R shoulder/elbow/wrist, L shoulder/elbow/wrist, mid hip,
#  R eye, L eye, R ear, L ear]. The remaining 42 nodes are both hands.
MEDIAPIPE_BODY_INDICES = (0, None, 12, 14, 16, 11, 13, 15, None, 5, 2, 8, 7)
LEFT_HAND_SLICE = slice(33, 54)
RIGHT_HAND_SLICE = slice(54, 75)
TEST_CLIP_COUNT = 4


def landmark_motion_score(landmarks: np.ndarray) -> float:
    """Return robust mean movement for visible hand landmarks."""
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


def mediapipe_to_openpose_55(landmarks: np.ndarray) -> np.ndarray:
    """Map MediaPipe pose + hands to the 55-node OpenPose WLASL ordering."""
    if landmarks.ndim != 3 or landmarks.shape[1:] != (INPUT_LANDMARK_COUNT, 3):
        raise ValueError(f"Expected landmarks shaped [frames, {INPUT_LANDMARK_COUNT}, 3].")

    pose_xy = landmarks[:, :33, :2]
    body_points = []
    for position, index in enumerate(MEDIAPIPE_BODY_INDICES):
        if index is not None:
            body_points.append(pose_xy[:, index])
        elif position == 1:
            body_points.append((pose_xy[:, 11] + pose_xy[:, 12]) / 2)
        else:
            body_points.append((pose_xy[:, 23] + pose_xy[:, 24]) / 2)
    body = np.stack(body_points, axis=1)
    return np.concatenate(
        (body, landmarks[:, LEFT_HAND_SLICE, :2], landmarks[:, RIGHT_HAND_SLICE, :2]),
        axis=1,
    )


def _clip_start_indices(frame_count: int) -> list[int]:
    """Match WLASL's four-copy fixed-length sequential test sampling."""
    if frame_count <= MODEL_FRAME_COUNT:
        return [0]
    if frame_count > MODEL_FRAME_COUNT * TEST_CLIP_COUNT:
        first = (frame_count - MODEL_FRAME_COUNT * TEST_CLIP_COUNT) // 2
        return [first + index * MODEL_FRAME_COUNT for index in range(TEST_CLIP_COUNT)]
    stride = math.floor((frame_count - MODEL_FRAME_COUNT) / (TEST_CLIP_COUNT - 1))
    return [index * stride for index in range(TEST_CLIP_COUNT)]


def _fixed_length_clip(landmarks: np.ndarray, start: int) -> np.ndarray:
    clip = landmarks[start : start + MODEL_FRAME_COUNT]
    if len(clip) < MODEL_FRAME_COUNT:
        padding = np.repeat(clip[-1:], MODEL_FRAME_COUNT - len(clip), axis=0)
        clip = np.concatenate((clip, padding), axis=0)
    return clip


def prepare_landmarks(
    frames: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[torch.Tensor, float]:
    """Convert browser holistic frames to TGCN ``[views, 55, 100]`` input."""
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
    openpose = mediapipe_to_openpose_55(array)
    # WLASL decoded 256x256 coordinates and applied 2 * (coordinate / 256 - .5).
    # MediaPipe already reports normalized coordinates, so this is equivalent.
    normalized = openpose * 2.0 - 1.0
    views = []
    for start in _clip_start_indices(len(normalized)):
        clip = _fixed_length_clip(normalized, start)
        # Per node: [x(frame 0), y(frame 0), ..., x(frame 49), y(frame 49)].
        views.append(clip.transpose(1, 0, 2).reshape(MODEL_LANDMARK_COUNT, -1))
    return torch.from_numpy(np.stack(views).astype(np.float32)), motion_score
