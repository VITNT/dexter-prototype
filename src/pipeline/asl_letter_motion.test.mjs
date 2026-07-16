import assert from "node:assert/strict";
import {
    ASL_ACTIVE_FINGERTIP_THRESHOLD,
    ASL_ACTIVE_WRIST_THRESHOLD,
    getAslDynamicLetterIntent,
    hasAslZTrajectory,
    hasDynamicAslLetterMotion,
    hasEmergingAslLetterMotion,
    measureAslZTrajectory,
    measureAslLetterMotion
} from "./asl_letter_motion.mjs";

function handFrame({
    wristX = 0.5,
    wristY = 0.5,
    indexX = wristX + 0.01,
    indexY = wristY - 0.16,
    pinkyX = wristX + 0.08,
    pinkyY = wristY - 0.1
} = {}) {
    const frame = new Array(63).fill(0);
    for (let index = 0; index < 21; index++) {
        frame[index * 3] = wristX;
        frame[index * 3 + 1] = wristY;
    }
    frame[8 * 3] = indexX;
    frame[8 * 3 + 1] = indexY;
    frame[20 * 3] = pinkyX;
    frame[20 * 3 + 1] = pinkyY;
    return frame;
}

const heldI = Array.from({ length: 30 }, (_, index) => {
    const jitter = Math.sin(index) * 0.002;
    return handFrame({ wristX: 0.5 + jitter, wristY: 0.5 - jitter });
});
assert.equal(hasDynamicAslLetterMotion(heldI), false, "A held I must stay in static recognition.");

const pinkyDrawnJ = Array.from({ length: 30 }, (_, index) => {
    const progress = index / 29;
    const angle = progress * Math.PI * 2;
    return handFrame({
        wristX: 0.5 + Math.sin(index) * 0.001,
        wristY: 0.5,
        pinkyX: 0.58 + Math.sin(angle) * 0.045,
        pinkyY: 0.4 + (1 - Math.cos(angle)) * 0.025
    });
});
const jMotion = measureAslLetterMotion(pinkyDrawnJ);
assert.ok(jMotion.wristSpan < 0.01, "The J fixture keeps its wrist still.");
assert.ok(jMotion.pinkyTipSpan > 0.08, "The J fixture moves its pinky relative to the wrist.");
assert.equal(
    hasDynamicAslLetterMotion(pinkyDrawnJ),
    true,
    "Pinky motion must route J to the dynamic model even when the wrist stays still."
);
assert.equal(
    hasEmergingAslLetterMotion(pinkyDrawnJ.slice(0, 3)),
    true,
    "The UI should hide J's temporary I pose as soon as pinky motion emerges."
);
assert.equal(getAslDynamicLetterIntent(pinkyDrawnJ).J, true);
assert.equal(getAslDynamicLetterIntent(pinkyDrawnJ).Z, false, "A curved J path must not qualify as Z.");

const indexDrawnZ = Array.from({ length: 30 }, (_, index) => {
    const segment = Math.min(2, Math.floor(index / 10));
    const segmentProgress = (index % 10) / 9;
    const indexX = segment === 0
        ? 0.42 + segmentProgress * 0.16
        : (segment === 1
            ? 0.58 - segmentProgress * 0.16
            : 0.42 + segmentProgress * 0.16);
    const indexY = segment === 0
        ? 0.34
        : (segment === 1 ? 0.34 + segmentProgress * 0.1 : 0.44);
    return handFrame({
        indexX,
        indexY
    });
});
assert.equal(
    hasDynamicAslLetterMotion(indexDrawnZ),
    true,
    "Index-finger motion must continue to route Z to the dynamic model."
);
assert.equal(hasAslZTrajectory(indexDrawnZ), true, "A deliberate three-stroke Z must pass the trajectory gate.");
assert.equal(getAslDynamicLetterIntent(indexDrawnZ).Z, true);
const zTrajectory = measureAslZTrajectory(indexDrawnZ);
assert.ok(zTrajectory.horizontalReversals >= 2);
assert.ok(zTrajectory.horizontalRange > 0.1);
assert.ok(zTrajectory.verticalRange > 0.07);
const mirroredZ = indexDrawnZ.map(frame => frame.map((value, valueIndex) =>
    valueIndex % 3 === 0 ? 1 - value : value
));
assert.equal(hasAslZTrajectory(mirroredZ), true, "Mirroring must not change a deliberate Z trajectory.");
const noisyZ = indexDrawnZ.map((frame, index) => {
    const noisyFrame = frame.slice();
    noisyFrame[8 * 3] += Math.sin(index * 1.7) * 0.003;
    noisyFrame[8 * 3 + 1] += Math.cos(index * 1.3) * 0.002;
    return noisyFrame;
});
assert.equal(hasAslZTrajectory(noisyZ), true, "Normal fingertip tracking noise must not hide a deliberate Z.");
assert.equal(
    hasEmergingAslLetterMotion(indexDrawnZ.slice(0, 5)),
    true,
    "The UI should hide Z's temporary static pose as soon as index motion emerges."
);

const slightIndexMovement = Array.from({ length: 30 }, (_, index) => handFrame({
    indexX: 0.51 + Math.sin(index * 0.8) * 0.015,
    indexY: 0.34 + Math.cos(index * 0.6) * 0.006
}));
assert.equal(
    hasDynamicAslLetterMotion(slightIndexMovement, {
        wristThreshold: Number.POSITIVE_INFINITY,
        fingertipThreshold: ASL_ACTIVE_FINGERTIP_THRESHOLD
    }),
    true,
    "The old span-only gate would consider this slight movement active."
);
assert.equal(hasAslZTrajectory(slightIndexMovement), false, "Small index movement must not become Z.");
assert.equal(getAslDynamicLetterIntent(slightIndexMovement).Z, false);

const oneWayIndexSwipe = Array.from({ length: 30 }, (_, index) => handFrame({
    indexX: 0.44 + (index / 29) * 0.14,
    indexY: 0.34 + (index / 29) * 0.05
}));
assert.equal(hasAslZTrajectory(oneWayIndexSwipe), false, "A single diagonal swipe is not a Z.");

const horizontalWiggle = Array.from({ length: 30 }, (_, index) => {
    const segment = Math.floor(index / 10);
    const progress = (index % 10) / 9;
    const indexX = segment === 1 ? 0.58 - progress * 0.16 : 0.42 + progress * 0.16;
    return handFrame({ indexX, indexY: 0.34 });
});
assert.equal(hasAslZTrajectory(horizontalWiggle), false, "Horizontal back-and-forth motion without the diagonal drop is not Z.");

assert.equal(
    hasDynamicAslLetterMotion(pinkyDrawnJ, {
        wristThreshold: ASL_ACTIVE_WRIST_THRESHOLD,
        fingertipThreshold: ASL_ACTIVE_FINGERTIP_THRESHOLD
    }),
    true,
    "A fingertip-drawn J must not be rejected as idle during inference."
);

console.log("ASL letter motion tests passed: held I stays static while fingertip-drawn J/Z route dynamically.");
