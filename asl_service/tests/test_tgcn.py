import hashlib
import unittest

import numpy as np
import torch

from asl_service.config import (
    CHECKPOINT_PATH,
    INPUT_FEATURE_COUNT,
    MODEL_FEATURE_COUNT,
    MODEL_FRAME_COUNT,
    MODEL_LANDMARK_COUNT,
    NUM_CLASSES,
)
from asl_service.inference import _model_state, load_labels, load_model
from asl_service.tgcn_model import TGCN
from asl_service.tgcn_preprocessing import (
    _clip_start_indices,
    landmark_motion_score,
    mediapipe_to_openpose_55,
    prepare_landmarks,
)


EXPECTED_CHECKPOINT_SIZE = 26_491_626
EXPECTED_CHECKPOINT_SHA256 = (
    "23a994294be6156bcebb4d333a891dc8594a83f461b58b84068472729f5ba929"
)


class TgcnContractTests(unittest.TestCase):
    def test_mediapipe_points_map_to_openpose_55_order(self):
        frames = np.zeros((1, 75, 3), dtype=np.float32)
        frames[0, :, 0] = np.arange(75)

        mapped = mediapipe_to_openpose_55(frames)

        self.assertEqual(mapped.shape, (1, MODEL_LANDMARK_COUNT, 2))
        np.testing.assert_allclose(
            mapped[0, :13, 0],
            [0, 11.5, 12, 14, 16, 11, 13, 15, 23.5, 5, 2, 8, 7],
        )
        np.testing.assert_allclose(mapped[0, 13:, 0], np.arange(33, 75))

    def test_short_sequence_is_normalized_and_edge_padded(self):
        frames = np.zeros((2, 75, 3), dtype=np.float32)
        frames[0, 0, :2] = [0.25, 0.75]
        frames[1, 0, :2] = [0.5, 1.0]

        tensor, motion_score = prepare_landmarks(frames)

        self.assertEqual(tuple(tensor.shape), (1, MODEL_LANDMARK_COUNT, MODEL_FEATURE_COUNT))
        self.assertEqual(tensor.dtype, torch.float32)
        np.testing.assert_allclose(tensor[0, 0, :6], [-0.5, 0.5, 0.0, 1.0, 0.0, 1.0])
        self.assertEqual(motion_score, 0)

    def test_long_sequence_creates_original_four_test_views(self):
        frames = np.zeros((75, INPUT_FEATURE_COUNT), dtype=np.float32)

        tensor, _ = prepare_landmarks(frames)

        self.assertEqual(tuple(tensor.shape), (4, MODEL_LANDMARK_COUNT, MODEL_FEATURE_COUNT))
        self.assertEqual(_clip_start_indices(75), [0, 8, 16, 24])
        self.assertEqual(_clip_start_indices(250), [25, 75, 125, 175])

    def test_motion_score_uses_visible_hands(self):
        frames = np.zeros((4, 75, 3), dtype=np.float32)
        for index in range(4):
            frames[index, 33] = [0.2 + index * 0.05, 0.5, 0]
        self.assertGreater(landmark_motion_score(frames), 0.04)

    def test_rejects_wrong_browser_feature_width(self):
        with self.assertRaisesRegex(ValueError, "Landmarks must be"):
            prepare_landmarks(np.zeros((30, 224), dtype=np.float32))

    def test_state_prefix_is_removed(self):
        tensor = torch.ones(1)
        state = _model_state({"state_dict": {"model.gc1.example": tensor}})
        self.assertEqual(state, {"gc1.example": tensor})

    def test_wlasl2000_labels_are_self_contained(self):
        labels = load_labels()
        self.assertEqual(len(labels), NUM_CLASSES)
        self.assertEqual(labels[0], "a")
        self.assertEqual(labels[-1], "zero")

    def test_checkpoint_bytes_are_hugging_face_weight(self):
        self.assertEqual(CHECKPOINT_PATH.stat().st_size, EXPECTED_CHECKPOINT_SIZE)
        digest = hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()
        self.assertEqual(digest, EXPECTED_CHECKPOINT_SHA256)

    def test_checkpoint_strictly_loads_and_runs(self):
        model, device = load_model()
        self.assertIsInstance(model, TGCN)
        with torch.inference_mode():
            logits = model(
                torch.zeros(1, MODEL_LANDMARK_COUNT, MODEL_FRAME_COUNT * 2, device=device)
            )
        self.assertEqual(tuple(logits.shape), (1, NUM_CLASSES))
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
