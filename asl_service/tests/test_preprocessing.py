import unittest
from unittest.mock import patch

import cv2
import numpy as np

from asl_service.inference import load_labels
from asl_service.preprocessing import (
    _resize_and_crop,
    sample_frame_indices,
    select_active_frame_indices,
)


class WlaslContractTests(unittest.TestCase):
    def test_short_clip_is_edge_padded(self):
        indices = sample_frame_indices(46, 64)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 45)
        self.assertTrue(np.all(indices[1:] >= indices[:-1]))
        self.assertEqual(indices[:46].tolist(), list(range(46)))
        self.assertEqual(indices[46:].tolist(), [45] * 18)

    def test_short_clip_repeats_frames(self):
        self.assertEqual(sample_frame_indices(1, 4).tolist(), [0, 0, 0, 0])

    def test_long_clip_uses_contiguous_center_window(self):
        indices = sample_frame_indices(80, 64)
        self.assertEqual(indices.tolist(), list(range(8, 72)))

    def test_active_window_tracks_the_sign(self):
        energy = np.zeros(79, dtype=np.float32)
        energy[60:70] = 1
        indices = select_active_frame_indices(energy, 80, 64)
        self.assertLessEqual(indices[0], 60)
        self.assertGreaterEqual(indices[-1], 70)
        self.assertTrue(np.all(np.diff(indices) == 1))

    def test_webcam_frame_matches_the_square_wlasl_contract(self):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        with patch("asl_service.preprocessing.cv2.resize", wraps=cv2.resize) as resize:
            crop = _resize_and_crop(frame)

        self.assertEqual(resize.call_args_list[0].args[1], (256, 256))
        self.assertEqual(crop.shape, (224, 224, 3))

    def test_portrait_frame_matches_the_square_wlasl_contract(self):
        frame = np.zeros((640, 360, 3), dtype=np.uint8)
        with patch("asl_service.preprocessing.cv2.resize", wraps=cv2.resize) as resize:
            crop = _resize_and_crop(frame)

        self.assertEqual(resize.call_args_list[0].args[1], (256, 256))
        self.assertEqual(crop.shape, (224, 224, 3))

    def test_wlasl300_label_contract(self):
        labels = load_labels()
        self.assertEqual(len(labels), 300)
        self.assertEqual(labels[0], "book")
        self.assertEqual(labels[-1], "money")


if __name__ == "__main__":
    unittest.main()
