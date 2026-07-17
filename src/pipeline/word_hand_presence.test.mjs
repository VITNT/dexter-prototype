import assert from "node:assert/strict";
import { hasVisibleHandLandmarks } from "./word_hand_presence.mjs";

assert.equal(hasVisibleHandLandmarks(null), false);
assert.equal(hasVisibleHandLandmarks({ landmarks: [] }), false);
assert.equal(hasVisibleHandLandmarks({ landmarks: [[]] }), false);
assert.equal(hasVisibleHandLandmarks({ landmarks: [[{ x: 0.5, y: 0.5, z: 0 }]] }), true);

console.log("Hand-presence tests passed: landmark visibility is detected for gated modes.");
