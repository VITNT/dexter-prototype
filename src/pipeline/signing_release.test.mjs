import assert from "node:assert/strict";
import {
    handSignatureDistance,
    normalizeHandShapeSignature,
    updatePauseProgress
} from "./signing_release.mjs";

const heldShape = [[
    0.40, 0.50,
    0.36, 0.44,
    0.42, 0.34,
    0.46, 0.35,
    0.50, 0.37,
    0.53, 0.40
]];
const translatedAndScaledShape = [[
    0.60, 0.64,
    0.54, 0.55,
    0.63, 0.40,
    0.69, 0.415,
    0.75, 0.445,
    0.795, 0.49
]];
const changedFingerShape = [[
    0.40, 0.50,
    0.36, 0.44,
    0.58, 0.48,
    0.56, 0.50,
    0.54, 0.52,
    0.52, 0.54
]];

const heldNormalized = normalizeHandShapeSignature(heldShape);
const movedNormalized = normalizeHandShapeSignature(translatedAndScaledShape);
const changedNormalized = normalizeHandShapeSignature(changedFingerShape);

assert.ok(
    handSignatureDistance(heldNormalized, movedNormalized) < 0.02,
    "Moving or resizing the same held sign must not rearm recognition."
);
assert.ok(
    handSignatureDistance(heldNormalized, changedNormalized) >= 0.12,
    "A clearly different handshape should rearm recognition."
);
assert.equal(
    handSignatureDistance(heldNormalized, []),
    Infinity,
    "Removing the hand should count as a release."
);

function pauseAtFps(fps) {
    let elapsedMs = 0;
    const deltaMs = 1000 / fps;
    while (elapsedMs < 500) {
        elapsedMs = updatePauseProgress({
            elapsedMs,
            deltaMs,
            isPaused: true,
            durationMs: 500
        });
    }
    return elapsedMs;
}

assert.equal(pauseAtFps(15), 500, "A 15 FPS camera should complete the 500 ms reaction timer.");
assert.equal(pauseAtFps(30), 500, "A 30 FPS camera should complete the 500 ms reaction timer.");
assert.equal(
    updatePauseProgress({ elapsedMs: 300, deltaMs: 100, isPaused: false, durationMs: 500 }),
    100,
    "Continued motion should drain the rearm timer instead of completing it."
);

console.log("Signing release tests passed: held signs stay locked and dynamic pauses are frame-rate independent.");
