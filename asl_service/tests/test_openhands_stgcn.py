import hashlib
import unittest

import numpy as np
import torch

from asl_service.config import CHECKPOINT_PATH, INPUT_FEATURE_COUNT, NUM_CLASSES
from asl_service.inference import _model_state, load_labels, load_model
from asl_service.openhands_stgcn import OpenHandsSTGCN
from asl_service.stgcn_preprocessing import (
    landmark_motion_score,
    normalize_selected_landmarks,
    prepare_landmarks,
)


EXPECTED_CHECKPOINT_SIZE = 43_362_699
EXPECTED_CHECKPOINT_SHA256 = (
    "7a86185715e1a63a16dd87fbf6433e86bf1be4627193c181b148398a8603bc7a"
)


class OpenHandsStgcnContractTests(unittest.TestCase):
    def test_clip_level_shoulder_normalization_matches_openhands(self):
        frames = np.zeros((2, 27, 2), dtype=np.float32)
        frames[0, 3] = [0, 0]
        frames[0, 4] = [2, 0]
        frames[1, 3] = [2, 0]
        frames[1, 4] = [4, 0]

        normalized = normalize_selected_landmarks(frames)

        np.testing.assert_allclose(normalized[0, 3], [-1, 0])
        np.testing.assert_allclose(normalized[0, 4], [0, 0])
        np.testing.assert_allclose(normalized[1, 3], [0, 0])
        np.testing.assert_allclose(normalized[1, 4], [1, 0])

    def test_prepare_landmarks_returns_openhands_nctv_shape(self):
        frames = np.zeros((45, INPUT_FEATURE_COUNT), dtype=np.float32)
        frames[:, 11 * 3 : 11 * 3 + 3] = [0.25, 0.4, 0]
        frames[:, 12 * 3 : 12 * 3 + 3] = [0.75, 0.4, 0]

        tensor, motion_score = prepare_landmarks(frames)

        self.assertEqual(tuple(tensor.shape), (1, 2, 45, 27))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(motion_score, 0)

    def test_motion_score_uses_visible_hands(self):
        frames = np.zeros((4, 75, 3), dtype=np.float32)
        for index in range(4):
            frames[index, 33] = [0.2 + index * 0.05, 0.5, 0]
        self.assertGreater(landmark_motion_score(frames), 0.04)

    def test_rejects_wrong_browser_feature_width(self):
        with self.assertRaisesRegex(ValueError, "Landmarks must be"):
            prepare_landmarks(np.zeros((30, 224), dtype=np.float32))

    def test_lightning_state_prefix_is_removed(self):
        tensor = torch.ones(1)
        state = _model_state(
            {"state_dict": {"model.encoder.example": tensor}}
        )
        self.assertEqual(state, {"encoder.example": tensor})

    def test_wlasl2000_labels_are_self_contained(self):
        labels = load_labels()
        self.assertEqual(len(labels), NUM_CLASSES)
        self.assertEqual(labels[0], "a")
        self.assertEqual(labels[-1], "zero")

    def test_checkpoint_bytes_are_the_supplied_openhands_weight(self):
        self.assertEqual(CHECKPOINT_PATH.stat().st_size, EXPECTED_CHECKPOINT_SIZE)
        digest = hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()
        self.assertEqual(digest, EXPECTED_CHECKPOINT_SHA256)

    def test_checkpoint_strictly_loads_and_runs(self):
        model, device = load_model()
        self.assertIsInstance(model, OpenHandsSTGCN)
        with torch.inference_mode():
            logits = model(torch.zeros(1, 2, 30, 27, device=device))
        self.assertEqual(tuple(logits.shape), (1, NUM_CLASSES))
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
