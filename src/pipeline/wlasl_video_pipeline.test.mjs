import assert from "node:assert/strict";
import {
  normalizeRgbBytes,
  uniformSampleIndices,
  validateWlaslManifest
} from "./wlasl_video_pipeline.mjs";

assert.deepEqual(uniformSampleIndices(5, 3), [0, 2, 4]);
assert.deepEqual(uniformSampleIndices(1, 4), [0, 0, 0, 0]);
assert.deepEqual(Array.from(normalizeRgbBytes([0, 127.5, 255])), [-1, 0, 1]);

const config = validateWlaslManifest({
  enabled: true,
  modelPath: "model.json",
  labelsPath: "labels.json"
});
assert.equal(config.frameCount, 64);
assert.equal(config.imageSize, 224);
assert.throws(() => validateWlaslManifest({ enabled: false }), /not enabled/);

console.log("WLASL video preprocessing tests passed.");

