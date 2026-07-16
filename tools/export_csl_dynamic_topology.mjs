import { writeFile } from "node:fs/promises";
import * as tf from "@tensorflow/tfjs";

const output = process.argv[2];
if (!output) throw new Error("Usage: node export_csl_dynamic_topology.mjs <output.json>");

// Preserve the released serialized topology name for byte-for-byte conversion parity.
const model = tf.sequential({ name: "cadt_ksl_words" });
model.add(tf.layers.lstm({ name: "lstm", inputShape: [30, 126], units: 128, returnSequences: true, recurrentActivation: "sigmoid" }));
model.add(tf.layers.dropout({ name: "dropout", rate: 0.3 }));
model.add(tf.layers.lstm({ name: "lstm_1", units: 128, returnSequences: false, recurrentActivation: "sigmoid" }));
model.add(tf.layers.dropout({ name: "dropout_1", rate: 0.3 }));
model.add(tf.layers.dense({ name: "dense", units: 256, activation: "relu" }));
model.add(tf.layers.batchNormalization({ name: "batch_normalization" }));
model.add(tf.layers.dropout({ name: "dropout_2", rate: 0.2 }));
model.add(tf.layers.dense({ name: "dense_1", units: 128, activation: "relu" }));
model.add(tf.layers.dense({ name: "dense_2", units: 75, activation: "softmax" }));

const serialized = model.toJSON(null, false);
const payload = {
  modelTopology: typeof serialized === "string" ? JSON.parse(serialized) : serialized,
  weights: model.weights.map((weight) => ({
    name: weight.originalName || weight.name,
    shape: weight.shape,
    dtype: weight.dtype
  }))
};
await writeFile(output, JSON.stringify(payload), "utf8");
