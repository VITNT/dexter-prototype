from __future__ import annotations

import cv2
import numpy as np
import torch

from .config import IMAGE_SIZE
from .preprocessing import _resize_and_crop


def prepare_continuous_video(
    video_path: str,
    target_fps: float = 15.0,
    max_seconds: float = 10.0,
) -> tuple[torch.Tensor, int]:
    capture = cv2.VideoCapture(video_path)
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(source_fps) or source_fps <= 0:
        source_fps = 30.0
    stride = max(1, round(source_fps / target_fps))
    max_source_frames = max(1, round(source_fps * max_seconds))
    frames: list[np.ndarray] = []
    source_index = 0

    while source_index < max_source_frames:
        ok, frame = capture.read()
        if not ok:
            break
        if source_index % stride == 0:
            frames.append(_resize_and_crop(frame))
        source_index += 1
    capture.release()

    if not frames:
        raise ValueError("The video contains no readable frames.")
    clip = np.stack(frames).astype(np.float32)
    clip = (clip / 255.0) * 2.0 - 1.0
    tensor = torch.from_numpy(clip.transpose(3, 0, 1, 2)).unsqueeze(0)
    duration_ms = round(len(frames) * 1000 / target_fps)
    assert tensor.shape[3:] == (IMAGE_SIZE, IMAGE_SIZE)
    return tensor, duration_ms
