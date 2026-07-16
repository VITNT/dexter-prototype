import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from tools.phonssm_inference import (
    EXPECTED_CHECKPOINT_SHA256,
    PhonSSMInferenceError,
    _git_lfs_help,
    load_labels,
    predict,
    verify_checkpoint,
)


class PhonSsmCliTests(unittest.TestCase):
    def test_reports_missing_git_lfs(self):
        with patch("tools.phonssm_inference.shutil.which", return_value=None):
            self.assertIn("brew install git-lfs", _git_lfs_help())

    def test_rejects_git_lfs_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best_model.pt"
            checkpoint.write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                f"oid sha256:{EXPECTED_CHECKPOINT_SHA256}\n"
                "size 38967545\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PhonSSMInferenceError, "only a Git LFS pointer"):
                verify_checkpoint(checkpoint)

    def test_rejects_checkpoint_checksum_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best_model.pt"
            checkpoint.write_bytes(b"not weights")
            with patch("tools.phonssm_inference.EXPECTED_CHECKPOINT_SIZE", len(b"not weights")):
                with self.assertRaisesRegex(PhonSSMInferenceError, "SHA-256 mismatch"):
                    verify_checkpoint(checkpoint)

    def test_loads_label_map_in_class_index_order(self):
        with tempfile.TemporaryDirectory() as directory:
            label_map = Path(directory) / "label_map.json"
            label_map.write_text(json.dumps({"second": 1, "first": 0}), encoding="utf-8")
            self.assertEqual(load_labels(label_map, 2), ["first", "second"])

    def test_rejects_wrong_input_shape_before_inference(self):
        with self.assertRaisesRegex(PhonSSMInferenceError, "Expected input shape"):
            predict(None, ["unused"], torch.zeros(1, 30, 64), torch.device("cpu"))

    def test_accepts_flattened_and_structured_input_shapes(self):
        class FakeModel:
            def __call__(self, tensor):
                self.last_shape = tuple(tensor.shape)
                return {"logits": torch.tensor([[0.0, 2.0]])}

        for shape in ((1, 30, 63), (1, 30, 21, 3)):
            with self.subTest(shape=shape):
                model = FakeModel()
                result = predict(
                    model,
                    ["first", "second"],
                    torch.zeros(shape),
                    torch.device("cpu"),
                )
                self.assertEqual(model.last_shape, shape)
                self.assertEqual(result["label"], "second")
                self.assertGreater(result["confidence"], 0.8)


if __name__ == "__main__":
    unittest.main()
