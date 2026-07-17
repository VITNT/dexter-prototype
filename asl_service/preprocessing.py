from __future__ import annotations

import cv2
import numpy as np
import torch

# Retained only for the historical I3D preprocessing tests. The live service
# now imports ``tgcn_preprocessing`` instead of this module.
FRAME_COUNT = 64
IMAGE_SIZE = 224
RESIZE_SIZE = 256


def sample_frame_indices(source_count: int, target_count: int = FRAME_COUNT) -> np.ndarray:
    """Select one contiguous WLASL clip and pad short clips at the edge.

    This intentionally does not uniformly resample time. I3D was trained with
    contiguous 64-frame windows; spreading repeated frames through a short
    webcam recording changes the motion pattern seen by the network.
    """
    if source_count < 1:
        raise ValueError("The video contains no readable frames.")
    if source_count <= target_count:
        return np.minimum(np.arange(target_count), source_count - 1).astype(np.int64)
    start = (source_count - target_count) // 2
    return np.arange(start, start + target_count, dtype=np.int64)


def select_active_frame_indices(
    frame_energy: np.ndarray,
    source_count: int,
    target_count: int = FRAME_COUNT,
) -> np.ndarray:
    """Choose the contiguous target window containing the most movement."""
    if source_count <= target_count:
        return sample_frame_indices(source_count, target_count)
    if frame_energy.shape != (source_count - 1,):
        raise ValueError("frame_energy must contain one value per frame transition.")

    # energy[i] is the transition from frame i to i+1. A window beginning at
    # s contains transitions s .. s+target_count-2.
    transition_count = target_count - 1
    scores = np.convolve(frame_energy, np.ones(transition_count), mode="valid")
    start = int(np.argmax(scores))
    return np.arange(start, start + target_count, dtype=np.int64)


def prepare_video(video_path: str) -> tuple[torch.Tensor, float]:
    capture = cv2.VideoCapture(video_path)
    frames: list[np.ndarray] = []
    motion_frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(_resize_and_crop(frame))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_frames.append(cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA))
    capture.release()

    if not frames:
        raise ValueError("The video contains no readable frames.")

    if len(motion_frames) > 1:
        frame_energy = np.asarray([
            np.mean(np.abs(current.astype(np.float32) - previous.astype(np.float32))) / 255.0
            for previous, current in zip(motion_frames, motion_frames[1:])
        ], dtype=np.float32)
        active_count = max(1, (len(frame_energy) + 1) // 2)
        motion_score = float(np.mean(np.partition(frame_energy, -active_count)[-active_count:]))
    else:
        frame_energy = np.empty(0, dtype=np.float32)
        motion_score = 0.0

    indices = (
        select_active_frame_indices(frame_energy, len(frames))
        if len(frames) > FRAME_COUNT
        else sample_frame_indices(len(frames))
    )
    clip = np.stack([frames[index] for index in indices]).astype(np.float32)
    clip = (clip / 255.0) * 2.0 - 1.0

    # The released checkpoint was trained from OpenCV-decoded BGR video and
    # consumes C,T,H,W tensors.
    tensor = torch.from_numpy(clip.transpose(3, 0, 1, 2)).unsqueeze(0)
    return tensor, motion_score


def _resize_and_crop(frame: np.ndarray) -> np.ndarray:
    # WLASL's released videos/checkpoint use square 256x256 frames and a
    # 224x224 center crop. Preserve that training contract for webcam video;
    # an aspect-preserving square crop removes the signer's lateral hand space.
    resized = cv2.resize(frame, (RESIZE_SIZE, RESIZE_SIZE), interpolation=cv2.INTER_LINEAR)
    offset = (RESIZE_SIZE - IMAGE_SIZE) // 2
    return resized[offset : offset + IMAGE_SIZE, offset : offset + IMAGE_SIZE]
