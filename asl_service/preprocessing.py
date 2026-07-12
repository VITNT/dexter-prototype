from __future__ import annotations

import cv2
import numpy as np
import torch

from .config import FRAME_COUNT, IMAGE_SIZE


def sample_frame_indices(source_count: int, target_count: int = FRAME_COUNT) -> np.ndarray:
    if source_count < 1:
        raise ValueError("The video contains no readable frames.")
    if source_count == 1:
        return np.zeros(target_count, dtype=np.int64)
    return np.rint(np.linspace(0, source_count - 1, target_count)).astype(np.int64)


def prepare_video(video_path: str) -> torch.Tensor:
    capture = cv2.VideoCapture(video_path)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(_resize_and_crop(frame))
    capture.release()

    indices = sample_frame_indices(len(frames))
    clip = np.stack([frames[index] for index in indices]).astype(np.float32)
    clip = (clip / 255.0) * 2.0 - 1.0

    # The official I3D implementation consumes BGR frames decoded by OpenCV
    # and transposes T,H,W,C into C,T,H,W.
    return torch.from_numpy(clip.transpose(3, 0, 1, 2)).unsqueeze(0)


def _resize_and_crop(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = 256.0 / min(height, width)
    resized = cv2.resize(
        frame,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_LINEAR,
    )
    height, width = resized.shape[:2]
    top = max(0, (height - IMAGE_SIZE) // 2)
    left = max(0, (width - IMAGE_SIZE) // 2)
    crop = resized[top : top + IMAGE_SIZE, left : left + IMAGE_SIZE]
    if crop.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        crop = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE))
    return crop

