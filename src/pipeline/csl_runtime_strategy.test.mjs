import assert from "node:assert/strict";
import {
  averageProbabilityVectors,
  findStaticAgreement,
  normalizeCslLabel,
  resolveCslRuntimeConfig,
  selectShiftedWindows
} from "./csl_runtime_strategy.mjs";

const frames = Array.from({ length: 36 }, (_, index) => [index]);
assert.deepEqual(
  selectShiftedWindows(frames, 30, [0, 3, 6]).map(window => [window[0][0], window.at(-1)[0]]),
  [[6, 35], [3, 32], [0, 29]]
);
assert.deepEqual(selectShiftedWindows(frames.slice(0, 29), 30, [0, 3, 6]), []);
assert.deepEqual(averageProbabilityVectors([[0.8, 0.2], [0.6, 0.4]]), [0.7, 0.30000000000000004]);
assert.throws(() => averageProbabilityVectors([[1, 0], [1]]), /shared non-zero width/);
assert.equal(normalizeCslLabel(" សុខសប្បាយជាទេ? "), "សុខសប្បាយជាទេ");
assert.equal(
  findStaticAgreement("ផឹកទឹក", [{ word: "ផឹកទឹក", confidence: 0.8 }], 0.75)?.confidence,
  0.8
);
assert.equal(findStaticAgreement("ផឹកទឹក", [{ word: "ផឹកទឹក", confidence: 0.7 }], 0.75), null);

const metadata = {
  frameCount: 30,
  captureFrameCount: 36,
  windowFrameCount: 30,
  windowOffsetsFromEnd: [0, 3, 6],
  staticAgreement: { enabled: true }
};
assert.deepEqual(resolveCslRuntimeConfig(metadata).windowOffsetsFromEnd, [0, 3, 6]);
assert.equal(resolveCslRuntimeConfig(metadata).captureFrameCount, 36);
assert.equal(resolveCslRuntimeConfig(metadata).staticAgreement.enabled, true);

console.log("CSL runtime strategy tests passed: shifted windows, averaging, and static agreement.");
