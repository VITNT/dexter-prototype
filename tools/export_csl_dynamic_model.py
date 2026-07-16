"""Convert the released 30x126 Keras CSL word model to TensorFlow.js."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import h5py
import numpy as np


H5_PATHS = {
    "lstm/kernel": "model_weights/lstm/sequential/lstm/lstm_cell/kernel",
    "lstm/recurrent_kernel": "model_weights/lstm/sequential/lstm/lstm_cell/recurrent_kernel",
    "lstm/bias": "model_weights/lstm/sequential/lstm/lstm_cell/bias",
    "lstm_1/kernel": "model_weights/lstm_1/sequential/lstm_1/lstm_cell/kernel",
    "lstm_1/recurrent_kernel": "model_weights/lstm_1/sequential/lstm_1/lstm_cell/recurrent_kernel",
    "lstm_1/bias": "model_weights/lstm_1/sequential/lstm_1/lstm_cell/bias",
    "dense/kernel": "model_weights/dense/sequential/dense/kernel",
    "dense/bias": "model_weights/dense/sequential/dense/bias",
    "batch_normalization/gamma": "model_weights/batch_normalization/sequential/batch_normalization/gamma",
    "batch_normalization/beta": "model_weights/batch_normalization/sequential/batch_normalization/beta",
    "batch_normalization/moving_mean": "model_weights/batch_normalization/sequential/batch_normalization/moving_mean",
    "batch_normalization/moving_variance": "model_weights/batch_normalization/sequential/batch_normalization/moving_variance",
    "dense_1/kernel": "model_weights/dense_1/sequential/dense_1/kernel",
    "dense_1/bias": "model_weights/dense_1/sequential/dense_1/bias",
    "dense_2/kernel": "model_weights/dense_2/sequential/dense_2/kernel",
    "dense_2/bias": "model_weights/dense_2/sequential/dense_2/bias",
}


def extract_labels(server_source: Path, count: int) -> list[str]:
    tree = ast.parse(server_source.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Attribute) and target.attr == "actions" for target in node.targets):
            continue
        labels = list(ast.literal_eval(node.value.args[0]))
        if len(labels) < count:
            raise ValueError(f"Only {len(labels)} labels found for {count} model outputs.")
        return labels[:count]
    raise ValueError("Unable to locate the CSL action label array.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_model", type=Path)
    parser.add_argument("server_source", type=Path)
    parser.add_argument("topology", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    topology = json.loads(args.topology.read_text(encoding="utf-8"))
    specs = topology["weights"]
    chunks: list[bytes] = []
    with h5py.File(args.h5_model, "r") as source:
        for spec in specs:
            name = spec["name"]
            if name not in H5_PATHS:
                raise ValueError(f"No HDF5 mapping for TensorFlow.js weight {name!r}.")
            values = np.asarray(source[H5_PATHS[name]], dtype="<f4")
            if list(values.shape) != spec["shape"]:
                raise ValueError(f"Shape mismatch for {name}: {values.shape} != {spec['shape']}")
            chunks.append(values.tobytes(order="C"))

    output_count = next(spec["shape"][0] for spec in specs if spec["name"] == "dense_2/bias")
    labels = extract_labels(args.server_source, output_count)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights_name = "model.weights.bin"
    (args.output_dir / weights_name).write_bytes(b"".join(chunks))
    model_json = {
        "modelTopology": topology["modelTopology"],
        "format": "layers-model",
        "generatedBy": "Keras 3.10.0",
        "convertedBy": "Dexter deterministic CSL HDF5 exporter",
        "weightsManifest": [{"paths": [f"./{weights_name}"], "weights": specs}],
    }
    (args.output_dir / "model.json").write_text(
        json.dumps(model_json, separators=(",", ":")), encoding="utf-8"
    )
    (args.output_dir / "labels.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "adapter": "cadt_csl_hands_v1",
        "frameCount": 30,
        "frameWidth": 126,
        "classCount": output_count,
        "confidenceThreshold": 0.7,
        "source": "Phal-Sovandy/Khmer-Sign-Language-Translation-System",
        "sourceModel": args.h5_model.name,
        "license": "ISC",
        "limitations": "Released model has no idle class and no published signer-independent validation split."
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
