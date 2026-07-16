import unittest

import numpy as np

from asl_service.inference import load_labels
from asl_service.preprocessing import sample_frame_indices, select_active_frame_indices


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

    def test_wlasl2000_label_contract(self):
        labels = load_labels()
        self.assertEqual(len(labels), 2000)
        self.assertEqual(labels[0], "a")
        self.assertEqual(labels[-1], "zero")


if __name__ == "__main__":
    unittest.main()
