"""Slow smoke test for the installed TGCN checkpoint; not an accuracy test."""

import numpy as np

from asl_service.inference import predict_landmarks


frames = np.zeros((45, 75, 3), dtype=np.float32)
frames[:, 11] = [0.35, 0.4, 0]
frames[:, 12] = [0.65, 0.4, 0]
for frame_index in range(len(frames)):
    frames[frame_index, 33:54, 0] = 0.35 + frame_index * 0.003
    frames[frame_index, 33:54, 1] = 0.55

result = predict_landmarks(frames.tolist())
assert result["model"] == "TGCN WLASL2000"
assert len(result["predictions"]) == 3
assert result["word"]
print(result)
