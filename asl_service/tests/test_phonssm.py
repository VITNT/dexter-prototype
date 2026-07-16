import unittest

import numpy as np
import torch

from asl_service.config import FEATURE_COUNT, FRAME_COUNT
from asl_service.inference import _model_state, load_labels
from asl_service.phonssm_preprocessing import (
    landmark_motion_score,
    normalize_pose_hands,
    prepare_landmarks,
    sample_landmark_indices,
)


class PhonSsmContractTests(unittest.TestCase):
    def test_short_sequence_is_edge_padded(self):
        self.assertEqual(
            sample_landmark_indices(3, 5).tolist(),
            [0, 1, 2, 2, 2],
        )

    def test_long_sequence_is_uniformly_sampled_like_reference_training(self):
        self.assertEqual(
            sample_landmark_indices(7, 4).tolist(),
            [0, 2, 4, 6],
        )

    def test_shoulder_normalization_matches_reference_contract(self):
        frames = np.zeros((1, 75, 3), dtype=np.float32)
        frames[0, 11] = [0.25, 0.4, 0]
        frames[0, 12] = [0.75, 0.4, 0]
        frames[0, 33] = [0.5, 0.65, 0]

        normalized = normalize_pose_hands(frames)

        np.testing.assert_allclose(normalized[0, 11], [-0.5, 0, 0])
        np.testing.assert_allclose(normalized[0, 12], [0.5, 0, 0])
        np.testing.assert_allclose(normalized[0, 33], [0, 0.5, 0], atol=1e-6)

    def test_prepare_landmarks_returns_phonssm_input_shape(self):
        frames = np.zeros((45, FEATURE_COUNT), dtype=np.float32)
        frames[:, 11 * 3 : 11 * 3 + 3] = [0.25, 0.4, 0]
        frames[:, 12 * 3 : 12 * 3 + 3] = [0.75, 0.4, 0]

        tensor, motion_score = prepare_landmarks(frames)

        self.assertEqual(tuple(tensor.shape), (1, FRAME_COUNT, FEATURE_COUNT))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(motion_score, 0)

    def test_motion_score_uses_visible_hands(self):
        frames = np.zeros((4, 75, 3), dtype=np.float32)
        for index in range(4):
            frames[index, 33] = [0.2 + index * 0.05, 0.5, 0]
        self.assertGreater(landmark_motion_score(frames), 0.04)

    def test_rejects_wrong_feature_width(self):
        with self.assertRaisesRegex(ValueError, "Landmarks must be"):
            prepare_landmarks(np.zeros((30, 224), dtype=np.float32))

    def test_accepts_reference_checkpoint_wrapper(self):
        tensor = torch.ones(1)
        self.assertEqual(_model_state({"model_state_dict": {"module.weight": tensor}}), {"weight": tensor})

    def test_wlasl300_label_contract_is_self_contained(self):
        labels = load_labels()
        self.assertEqual(len(labels), 300)
        self.assertEqual(labels[0], "book")
        self.assertEqual(labels[-1], "money")


if __name__ == "__main__":
    unittest.main()
