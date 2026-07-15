import assert from "node:assert/strict";
import { readFile, stat } from "node:fs/promises";

const assetRoot = new URL("../../models/ksl/legacy-word-v1/", import.meta.url);
const model = JSON.parse(await readFile(new URL("model.json", assetRoot), "utf8"));
const labels = JSON.parse(await readFile(new URL("labels.json", assetRoot), "utf8"));
const metadata = JSON.parse(await readFile(new URL("metadata.json", assetRoot), "utf8"));
const manifest = model.weightsManifest[0];
const expectedBytes = manifest.weights.reduce(
  (total, weight) => total + weight.shape.reduce((size, value) => size * value, 1) * 4,
  0
);
const weights = await stat(new URL(manifest.paths[0], assetRoot));

assert.equal(labels.length, 75);
assert.equal(metadata.adapter, "ksl_legacy_hands_v1");
assert.equal(metadata.frameCount, 30);
assert.equal(metadata.frameWidth, 126);
assert.equal(weights.size, expectedBytes);
assert.deepEqual(
  model.modelTopology.config.layers[0].config.batch_input_shape,
  [null, 30, 126]
);

console.log(`Legacy KSL model asset verified: ${labels.length} labels, ${weights.size} weight bytes.`);
