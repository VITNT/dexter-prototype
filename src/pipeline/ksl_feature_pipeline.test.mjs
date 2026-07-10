import assert from "node:assert/strict";
import {
  KSL_BODY_POSE_INDICES,
  KSL_FEATURE_SCHEMA_V2,
  KSL_GRAPH_EDGES,
  KSL_LEFT_HAND_OFFSET,
  KSL_MODEL_FRAME_WIDTH,
  KSL_RAW_FRAME_WIDTH,
  KSL_RIGHT_HAND_OFFSET,
  KSL_SCHEMA_VERSION,
  KSL_TARGET_FRAME_COUNT,
  assignStableHandSlots,
  buildKslGraphFeatures,
  buildKslRawFrame,
  buildKslRawFrameFromMediaPipe,
  createKslHandSlotState,
  normalizeKslPositionFrames,
  prepareKslSequence
} from "./ksl_feature_pipeline.mjs";

function createPose() {
  const pose = Array.from({ length: 33 }, () => ({ x: 0, y: 0, z: 0 }));
  pose[11] = { x: 0.4, y: 0.5, z: 0.1 };
  pose[12] = { x: 0.6, y: 0.5, z: 0.1 };
  pose[13] = { x: 0.35, y: 0.65, z: 0.11 };
  pose[14] = { x: 0.65, y: 0.65, z: 0.11 };
  pose[15] = { x: 0.3, y: 0.8, z: 0.12 };
  pose[16] = { x: 0.7, y: 0.8, z: 0.12 };
  pose[23] = { x: 0.45, y: 0.9, z: 0.15 };
  pose[24] = { x: 0.55, y: 0.9, z: 0.15 };
  return pose;
}

function createHand(baseX, z = 0.02) {
  return Array.from({ length: 21 }, (_, index) => ({
    x: baseX + (index * 0.001),
    y: 0.78 + (index * 0.002),
    z: z + (index * 0.0005)
  }));
}

const pose = createPose();
const leftHand = createHand(0.25);
const rightHand = createHand(0.75);

assert.equal(KSL_SCHEMA_VERSION, "ksl-landmarks-v2");
assert.equal(KSL_RAW_FRAME_WIDTH, 150);
assert.equal(KSL_GRAPH_EDGES.length, 62);
assert.equal(KSL_MODEL_FRAME_WIDTH, 700);
assert.equal(KSL_FEATURE_SCHEMA_V2.featureLayout.handPresence.offset, 698);
assert.deepEqual(KSL_BODY_POSE_INDICES, [11, 12, 13, 14, 15, 16, 23, 24]);

const raw = buildKslRawFrame({ poseLandmarks: pose, leftHandLandmarks: leftHand, rightHandLandmarks: rightHand });
assert.equal(raw.length, KSL_RAW_FRAME_WIDTH);
assert.deepEqual(raw.slice(0, 3), [0.4, 0.5, 0.1]);
assert.deepEqual(raw.slice(KSL_LEFT_HAND_OFFSET * 3, KSL_LEFT_HAND_OFFSET * 3 + 3), [0.25, 0.78, 0.02]);

const poseWithDifferentFace = createPose();
poseWithDifferentFace[0] = { x: 999, y: -999, z: 42 };
assert.deepEqual(
  buildKslRawFrame({ poseLandmarks: poseWithDifferentFace, leftHandLandmarks: leftHand, rightHandLandmarks: rightHand }),
  raw,
  "face landmarks must not enter the KSL schema"
);

const slotState = createKslHandSlotState();
const slotted = assignStableHandSlots({
  poseLandmarks: pose,
  handLandmarks: [rightHand, leftHand],
  handednesses: [[{ categoryName: "Right" }], [{ categoryName: "Left" }]]
}, slotState);
assert.equal(slotted.leftHandLandmarks, leftHand);
assert.equal(slotted.rightHandLandmarks, rightHand);

const rightOnly = buildKslRawFrameFromMediaPipe({
  poseLandmarks: pose,
  handLandmarks: [rightHand],
  handednesses: [[{ categoryName: "Right" }]]
}, createKslHandSlotState());
assert.equal(rightOnly.leftPresent, false);
assert.equal(rightOnly.rightPresent, true);
assert.ok(rightOnly.frame.slice(KSL_LEFT_HAND_OFFSET * 3, KSL_RIGHT_HAND_OFFSET * 3).every(value => value === 0));

const normalized = normalizeKslPositionFrames([raw])[0];
assert.ok(Math.abs(normalized[0] + 0.5) < 1e-10);
assert.ok(Math.abs(normalized[3] - 0.5) < 1e-10);
assert.ok(Math.abs(normalized[2]) < 1e-10, "body z is shoulder-anchor relative");
assert.ok(Math.abs(normalized[KSL_LEFT_HAND_OFFSET * 3 + 2] - 0.1) < 1e-10, "hand z is scale-only");

const rawMissingLeft = buildKslRawFrame({ poseLandmarks: pose, rightHandLandmarks: createHand(0.77) });
const rawLeftReturns = buildKslRawFrame({
  poseLandmarks: pose,
  leftHandLandmarks: createHand(0.27),
  rightHandLandmarks: createHand(0.80)
});
const positionFrames = normalizeKslPositionFrames([raw, rawMissingLeft, rawLeftReturns]);
const graphFrames = buildKslGraphFeatures(positionFrames);
assert.ok(graphFrames.every(frame => frame.length === KSL_MODEL_FRAME_WIDTH));
assert.deepEqual(graphFrames[1].slice(KSL_LEFT_HAND_OFFSET * 9, KSL_LEFT_HAND_OFFSET * 9 + 9), new Array(9).fill(0));
assert.deepEqual(
  graphFrames[2].slice(KSL_LEFT_HAND_OFFSET * 9 + 3, KSL_LEFT_HAND_OFFSET * 9 + 9),
  new Array(6).fill(0),
  "motion is gated when a hand was absent; no fake dropout velocity"
);
assert.deepEqual(graphFrames[1].slice(698), [0, 1]);
assert.deepEqual(graphFrames[2].slice(698), [1, 1]);

const prepared = prepareKslSequence([raw, rawLeftReturns]);
assert.equal(prepared.length, KSL_TARGET_FRAME_COUNT);
assert.ok(prepared.every(frame => frame.length === KSL_MODEL_FRAME_WIDTH));
assert.ok(prepared.flat().every(Number.isFinite));
assert.throws(
  () => prepareKslSequence([new Array(225).fill(0), new Array(225).fill(0)]),
  /requires 150/,
  "legacy 225-value frames must not silently enter the face-free schema"
);

console.log("KSL feature pipeline tests passed: 60 frames, 150 raw values, 62 edges, 700 model values.");
