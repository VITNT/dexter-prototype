import unittest

import numpy as np

from asl_service.inference import load_labels
from asl_service.preprocessing import sample_frame_indices


class WlaslContractTests(unittest.TestCase):
    def test_uniform_sampling_keeps_clip_ends(self):
        indices = sample_frame_indices(46, 64)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 45)
        self.assertTrue(np.all(indices[1:] >= indices[:-1]))

    def test_short_clip_repeats_frames(self):
        self.assertEqual(sample_frame_indices(1, 4).tolist(), [0, 0, 0, 0])

    def test_wlasl300_label_contract(self):
        labels = load_labels()
        self.assertEqual(len(labels), 300)
        self.assertEqual(labels[0], "book")
        self.assertEqual(labels[-1], "money")


if __name__ == "__main__":
    unittest.main()

