import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from asl_service import assets


class HostedAssetTests(unittest.IsolatedAsyncioTestCase):
    async def test_private_blob_content_response_is_written(self):
        class FakeAsyncBlobClient:
            async def get(self, pathname, *, access):
                self.pathname = pathname
                self.access = access
                return types.SimpleNamespace(status_code=200, content=b"checkpoint")

        fake_blob_module = types.ModuleType("vercel.blob")
        fake_blob_module.AsyncBlobClient = FakeAsyncBlobClient
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "models" / "wlasl300.pt"
            with patch.dict(sys.modules, {"vercel.blob": fake_blob_module}):
                await assets._download_private_blob("wlasl/wlasl300.pt", target)

            self.assertEqual(target.read_bytes(), b"checkpoint")

    async def test_missing_assets_require_private_blob_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(assets, "CHECKPOINT_PATH", root / "models" / "wlasl300.pt"),
                patch.object(assets, "OFFICIAL_I3D_ROOT", root / "I3D"),
                patch.object(assets, "CLASS_LIST_PATH", root / "I3D" / "preprocess" / "labels.txt"),
                patch.dict(os.environ, {}, clear=True),
            ):
                with self.assertRaisesRegex(FileNotFoundError, "private Blob access"):
                    await assets.ensure_wlasl_assets()

    async def test_private_blob_populates_ephemeral_runtime_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "models" / "wlasl300.pt"
            i3d_root = root / "I3D"
            labels = i3d_root / "preprocess" / "labels.txt"

            async def write_download(_pathname, target):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"asset")

            download = AsyncMock(side_effect=write_download)
            with (
                patch.object(assets, "CHECKPOINT_PATH", checkpoint),
                patch.object(assets, "OFFICIAL_I3D_ROOT", i3d_root),
                patch.object(assets, "CLASS_LIST_PATH", labels),
                patch.object(assets, "_download_private_blob", download),
                patch.dict(
                    os.environ,
                    {"BLOB_READ_WRITE_TOKEN": "test-token"},
                    clear=True,
                ),
            ):
                await assets.ensure_wlasl_assets()

            self.assertEqual(download.await_count, 3)
            self.assertTrue(checkpoint.exists())
            self.assertTrue((i3d_root / "pytorch_i3d.py").exists())
            self.assertTrue(labels.exists())


if __name__ == "__main__":
    unittest.main()
