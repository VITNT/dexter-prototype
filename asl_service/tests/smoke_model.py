"""Slow smoke test for the installed checkpoint; it does not test accuracy."""

import tempfile
from pathlib import Path

import cv2
import numpy as np

from asl_service.inference import predict_video


with tempfile.TemporaryDirectory() as directory:
    video_path = Path(directory) / "synthetic.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        25.0,
        (256, 256),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create the smoke-test video.")
    for frame_index in range(45):
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.circle(frame, (40 + frame_index * 3, 128), 24, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()

    result = predict_video(str(video_path))
    assert len(result["predictions"]) == 3
    assert result["word"]
    print(result)
