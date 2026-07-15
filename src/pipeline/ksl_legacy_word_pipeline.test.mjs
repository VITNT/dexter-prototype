import assert from "node:assert/strict";
import {
  LEGACY_KSL_FRAME_COUNT,
  LEGACY_KSL_FRAME_WIDTH,
  buildLegacyKslHandFrame,
  prepareLegacyKslSequence
} from "./ksl_legacy_word_pipeline.mjs";

const hand = Array.from({ length: 21 }, (_, index) => ({
  x: 0.25 + index * 0.01,
  y: 0.50 - index * 0.02,
  z: -0.10 + index * 0.005
}));
const frame = buildLegacyKslHandFrame({ landmarks: [hand] });
assert.equal(frame.length, LEGACY_KSL_FRAME_WIDTH);
assert.deepEqual(frame.slice(0, 3), [0, 0, 0]);
assert.ok(Math.abs(frame[3] - 0.01) < 1e-12);
assert.ok(Math.abs(frame[4] + 0.02) < 1e-12);
assert.deepEqual(frame.slice(63), new Array(63).fill(0));

const sequence = prepareLegacyKslSequence(
  Array.from({ length: LEGACY_KSL_FRAME_COUNT + 2 }, () => frame)
);
assert.equal(sequence.length, LEGACY_KSL_FRAME_COUNT);
assert.equal(sequence[0].length, LEGACY_KSL_FRAME_WIDTH);
assert.equal(prepareLegacyKslSequence([frame]), null);

console.log("Legacy KSL word adapter tests passed: 30 frames x 126 wrist-relative values.");
