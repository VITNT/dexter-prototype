"""Export the released PyTorch LandmarkNet checkpoint for browser inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def reverse_label_map(converts: dict[str, list[str]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for display_label, forms in converts.items():
        for form in forms:
            result.setdefault(form, display_label)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("converts", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    classes = list(checkpoint["class_names"])
    with args.converts.open(encoding="utf-8") as source:
        labels = reverse_label_map(json.load(source))

    layer_names = ("fc.0", "fc.3", "fc.6")
    payload = {
        # This is a serialized upstream compatibility marker, not a UI label.
        "format": "ksl-static-landmarknet-v1",
        "source": "Phal-Sovandy/Khmer-Sign-Language-Translation-System",
        "license": "ISC",
        "checkpoint": args.checkpoint.name,
        "classes": classes,
        "displayLabels": [labels.get(name, name) for name in classes],
        "layers": [
            {
                "weight": state[f"{name}.weight"].detach().cpu().tolist(),
                "bias": state[f"{name}.bias"].detach().cpu().tolist(),
            }
            for name in layer_names
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
