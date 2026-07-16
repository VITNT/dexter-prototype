import assert from "node:assert/strict";
import {
  CADT_CSL_FRAME_COUNT,
  CADT_CSL_FRAME_WIDTH,
  buildCadtCslHandFrame,
  interpolateCadtCslFrame,
  prepareCadtCslSequence
} from "./csl_cadt_word_pipeline.mjs";

const hand = Array.from({ length: 21 }, (_, index) => ({
  x: 0.25 + index * 0.01,
  y: 0.50 - index * 0.02,
  z: -0.10 + index * 0.005
}));
const frame = buildCadtCslHandFrame({ landmarks: [hand] });
assert.equal(frame.length, CADT_CSL_FRAME_WIDTH);
assert.deepEqual(frame.slice(0, 3), [0, 0, 0]);
assert.ok(Math.abs(frame[3] - 0.01) < 1e-12);
assert.ok(Math.abs(frame[4] + 0.02) < 1e-12);
assert.deepEqual(frame.slice(63), new Array(63).fill(0));

const rightOnlyFrame = buildCadtCslHandFrame({
  landmarks: [hand],
  handednesses: [[{ categoryName: "Right" }]]
});
assert.deepEqual(rightOnlyFrame.slice(0, 63), new Array(63).fill(0));
assert.ok(Math.abs(rightOnlyFrame[66] - 0.01) < 1e-12);

const endFrame = frame.map(value => value + 0.2);
const midpoint = interpolateCadtCslFrame(frame, endFrame, 0.5);
assert.ok(Math.abs(midpoint[3] - 0.11) < 1e-12);
assert.throws(() => interpolateCadtCslFrame([0], endFrame, 0.5), /126-value frames/);

const sequence = prepareCadtCslSequence(
  Array.from({ length: CADT_CSL_FRAME_COUNT + 2 }, () => frame)
);
assert.equal(sequence.length, CADT_CSL_FRAME_COUNT);
assert.equal(sequence[0].length, CADT_CSL_FRAME_WIDTH);
assert.equal(prepareCadtCslSequence([frame]), null);

console.log("CADT CSL word adapter tests passed: 30 frames x 126 wrist-relative values.");
