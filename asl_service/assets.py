from __future__ import annotations

import asyncio
import os
from pathlib import Path

from .config import CHECKPOINT_PATH, CLASS_LIST_PATH, OFFICIAL_I3D_ROOT

_asset_lock = asyncio.Lock()


def required_assets_ready() -> bool:
    return (
        CHECKPOINT_PATH.exists()
        and (OFFICIAL_I3D_ROOT / "pytorch_i3d.py").exists()
        and CLASS_LIST_PATH.exists()
    )


async def _download_private_blob(pathname: str, target: Path) -> None:
    # Imported lazily so ordinary local development does not require Vercel's
    # Python SDK when all WLASL assets are already installed on disk.
    from vercel.blob import AsyncBlobClient

    client = AsyncBlobClient()
    result = await client.get(pathname, access="private")
    if result is None or result.status_code != 200 or result.stream is None:
        raise FileNotFoundError(f"Private Vercel Blob asset was not found: {pathname}")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_target = target.with_suffix(target.suffix + ".part")
    try:
        with temporary_target.open("wb") as output:
            async for chunk in result.stream:
                output.write(chunk)
        os.replace(temporary_target, target)
    finally:
        temporary_target.unlink(missing_ok=True)


async def ensure_wlasl_assets() -> None:
    if required_assets_ready():
        return
    if not os.getenv("VERCEL"):
        raise FileNotFoundError(
            "WLASL assets are missing. Follow asl_service/models/README.md."
        )

    async with _asset_lock:
        if required_assets_ready():
            return
        assets = (
            (os.getenv("WLASL_CHECKPOINT_BLOB", "wlasl/wlasl300.pt"), CHECKPOINT_PATH),
            (
                os.getenv("WLASL_I3D_BLOB", "wlasl/pytorch_i3d.py"),
                OFFICIAL_I3D_ROOT / "pytorch_i3d.py",
            ),
            (
                os.getenv("WLASL_LABELS_BLOB", "wlasl/wlasl_class_list.txt"),
                CLASS_LIST_PATH,
            ),
        )
        for pathname, target in assets:
            if not target.exists():
                await _download_private_blob(pathname, target)
