import assert from "node:assert/strict";
import {
  MIN_ASL_WORD_HAND_FRAMES,
  hasVisibleHandLandmarks,
  shouldSubmitAslWordClip
} from "./word_hand_presence.mjs";

assert.equal(hasVisibleHandLandmarks(null), false);
assert.equal(hasVisibleHandLandmarks({ landmarks: [] }), false);
assert.equal(hasVisibleHandLandmarks({ landmarks: [[]] }), false);
assert.equal(hasVisibleHandLandmarks({ landmarks: [[{ x: 0.5, y: 0.5, z: 0 }]] }), true);

assert.equal(
  shouldSubmitAslWordClip({ clipSize: 1024, trackedHandFrames: 0 }),
  false,
  "A non-empty video containing no tracked hands must not reach ASL inference."
);
assert.equal(
  shouldSubmitAslWordClip({
    clipSize: 1024,
    trackedHandFrames: MIN_ASL_WORD_HAND_FRAMES - 1
  }),
  false,
  "A momentary hand-tracker false positive must not unlock ASL inference."
);
assert.equal(
  shouldSubmitAslWordClip({
    clipSize: 1024,
    trackedHandFrames: MIN_ASL_WORD_HAND_FRAMES
  }),
  true
);
assert.equal(
  shouldSubmitAslWordClip({ clipSize: 0, trackedHandFrames: MIN_ASL_WORD_HAND_FRAMES }),
  false
);

console.log("Word hand-presence tests passed: idle ASL clips are rejected before inference.");
