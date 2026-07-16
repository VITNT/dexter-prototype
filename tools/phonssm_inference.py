#!/usr/bin/env python3
"""Run inference with the official pretrained PhonSSM checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = (
    PROJECT_ROOT / "models" / "phonssm" / "checkpoints" / "20260116_230934"
)
DEFAULT_CHECKPOINT = DEFAULT_RUN_DIR / "best_model.pt"
DEFAULT_CONFIG = DEFAULT_RUN_DIR / "config.json"
DEFAULT_LABEL_MAP = PROJECT_ROOT / "data" / "processed" / "merged" / "label_map.json"
EXPECTED_CHECKPOINT_SIZE = 38_967_545
EXPECTED_CHECKPOINT_SHA256 = (
    "fda2a4f3ac58e820e411c796c894d9ebaf135659fcb62eb6fb97cce452e1e281"
)
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


class PhonSSMInferenceError(RuntimeError):
    """A user-actionable PhonSSM setup or inference failure."""


def _git_lfs_help() -> str:
    if shutil.which("git-lfs") is None:
        return (
            "Git LFS is not installed. On macOS run `brew install git-lfs`, "
            "then `git lfs install` and `git lfs pull`."
        )
    return "Run `git lfs pull` in the cloned phon-ssm repository."


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        suffix = f" {_git_lfs_help()}" if description == "checkpoint" else ""
        raise PhonSSMInferenceError(f"Missing {description} file: {path}.{suffix}")


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(path: Path) -> tuple[int, str]:
    _require_file(path, "checkpoint")
    with path.open("rb") as checkpoint_file:
        header = checkpoint_file.read(256)
    if header.startswith(LFS_POINTER_PREFIX):
        raise PhonSSMInferenceError(
            f"{path} is only a Git LFS pointer, not model weights. {_git_lfs_help()}"
        )

    size = path.stat().st_size
    if size != EXPECTED_CHECKPOINT_SIZE:
        raise PhonSSMInferenceError(
            f"Checkpoint size mismatch for {path}: expected "
            f"{EXPECTED_CHECKPOINT_SIZE:,} bytes, found {size:,} bytes."
        )

    checksum = checkpoint_sha256(path)
    if checksum != EXPECTED_CHECKPOINT_SHA256:
        raise PhonSSMInferenceError(
            f"Checkpoint SHA-256 mismatch for {path}: expected "
            f"{EXPECTED_CHECKPOINT_SHA256}, found {checksum}."
        )
    return size, checksum


def load_json(path: Path, description: str) -> dict[str, Any]:
    _require_file(path, description)
    try:
        contents = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PhonSSMInferenceError(f"Could not read {description} file {path}: {error}") from error
    if not isinstance(contents, dict):
        raise PhonSSMInferenceError(f"The {description} file must contain a JSON object: {path}")
    return contents


def load_labels(path: Path, expected_count: int) -> list[str]:
    label_map = load_json(path, "label-map")
    try:
        indexed_labels = {int(index): str(label) for label, index in label_map.items()}
    except (TypeError, ValueError) as error:
        raise PhonSSMInferenceError(
            f"The label-map values must be integer class indices: {path}"
        ) from error

    expected_indices = list(range(expected_count))
    if sorted(indexed_labels) != expected_indices:
        raise PhonSSMInferenceError(
            f"The label map must contain exactly {expected_count} contiguous classes."
        )
    return [indexed_labels[index] for index in expected_indices]


def _import_torch() -> Any:
    try:
        import torch
    except (ImportError, OSError) as error:
        raise PhonSSMInferenceError(
            "PyTorch could not be imported. Install the project dependencies with "
            "`python -m pip install -r asl_service/requirements.txt`."
        ) from error
    return torch


def _select_device(torch: Any, requested: str) -> Any:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(
    checkpoint_path: Path,
    config_path: Path,
    device_name: str = "auto",
) -> tuple[Any, Any, Any, int, str]:
    size, checksum = verify_checkpoint(checkpoint_path)
    config_values = load_json(config_path, "configuration")
    torch = _import_torch()

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from asl_service.phonssm import PhonSSM, PhonSSMConfig
    except (ImportError, OSError) as error:
        raise PhonSSMInferenceError(
            "Could not import the repository's PhonSSM implementation. Run this "
            "script from a complete Mk-41 checkout with its dependencies installed."
        ) from error

    config = PhonSSMConfig.from_dict(config_values)
    device = _select_device(torch, device_name)
    model = PhonSSM(config).to(device)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as error:
        raise PhonSSMInferenceError(
            f"PyTorch could not load checkpoint {checkpoint_path}: {error}"
        ) from error

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict) or not state_dict:
        raise PhonSSMInferenceError("The checkpoint does not contain a model state dictionary.")
    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
        if isinstance(key, str)
    }
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise PhonSSMInferenceError(
            f"The checkpoint weights do not match config.json: {error}"
        ) from error

    model.eval()
    return model, config, device, size, checksum


def load_input_tensor(path: Path, torch: Any) -> Any:
    _require_file(path, "input tensor")
    try:
        if path.suffix.lower() == ".npy":
            try:
                import numpy as np
            except (ImportError, OSError) as error:
                raise PhonSSMInferenceError(
                    "NumPy is required to load .npy input tensors."
                ) from error
            value = np.load(path, allow_pickle=False)
        else:
            value = torch.load(path, map_location="cpu", weights_only=True)
    except PhonSSMInferenceError:
        raise
    except Exception as error:
        raise PhonSSMInferenceError(f"Could not load input tensor {path}: {error}") from error

    if isinstance(value, dict) and "input" in value:
        value = value["input"]
    try:
        return torch.as_tensor(value, dtype=torch.float32)
    except (TypeError, ValueError) as error:
        raise PhonSSMInferenceError(f"Input file does not contain a numeric tensor: {path}") from error


def predict(model: Any, labels: list[str], input_tensor: Any, device: Any) -> dict[str, Any]:
    torch = _import_torch()
    tensor = torch.as_tensor(input_tensor, dtype=torch.float32)
    accepted_shapes = {(1, 30, 63), (1, 30, 21, 3)}
    if tuple(tensor.shape) not in accepted_shapes:
        raise PhonSSMInferenceError(
            "Expected input shape (1, 30, 63) or (1, 30, 21, 3), "
            f"found {tuple(tensor.shape)}."
        )

    with torch.inference_mode():
        outputs = model(tensor.to(device))
        if not isinstance(outputs, dict) or "logits" not in outputs:
            raise PhonSSMInferenceError("PhonSSM did not return the expected logits output.")
        probabilities = torch.softmax(outputs["logits"][0], dim=0)
        confidence, class_index = probabilities.max(dim=0)

    index = int(class_index.item())
    if index >= len(labels):
        raise PhonSSMInferenceError(
            f"Predicted class index {index} is absent from the label map."
        )
    return {
        "label": labels[index],
        "confidence": float(confidence.item()),
        "class_index": index,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the official 5,565-class PhonSSM checkpoint.",
        epilog=(
            "Example: python tools/phonssm_inference.py "
            "--input /path/to/landmarks.npy"
        ),
    )
    parser.add_argument("--input", type=Path, help="A .pt or .npy input tensor.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify and load the checkpoint without running a prediction.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        model, config, device, size, checksum = load_model(
            args.checkpoint, args.config, args.device
        )
        labels = load_labels(args.label_map, config.num_signs)
        print(f"Checkpoint path: {args.checkpoint.resolve()}")
        print(f"Checkpoint size: {size:,} bytes ({size / (1024 * 1024):.1f} MiB)")
        print(f"SHA-256: {checksum} (OK)")
        print(f"Labels: {len(labels):,}")
        print(f"Device: {device}")

        if args.verify_only:
            return 0
        if args.input is None:
            raise PhonSSMInferenceError("Pass --input PATH or use --verify-only.")
        torch = _import_torch()
        result = predict(model, labels, load_input_tensor(args.input, torch), device)
        print(f"Predicted sign: {result['label']}")
        print(f"Confidence: {result['confidence']:.6f}")
        print(f"Class index: {result['class_index']}")
        return 0
    except PhonSSMInferenceError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
