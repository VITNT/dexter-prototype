/**
 * Face-free KSL landmark preprocessing shared by recording, training, and inference.
 *
 * Raw node order (50 xyz nodes / 150 values):
 *   0..7   MediaPipe Pose 11,12,13,14,15,16,23,24
 *   8..28  signer's left hand 0..20
 *   29..49 signer's right hand 0..20
 *
 * Model feature order (700 values):
 *   0..449   node-major [x,y,z,vx,vy,vz,ax,ay,az] for 50 nodes
 *   450..697 edge-major [dx,dy,dz,distance] for 62 graph edges
 *   698..699 [leftHandPresent,rightHandPresent]
 */

export const KSL_SCHEMA_VERSION = "ksl-landmarks-v2";
export const KSL_TARGET_FRAME_COUNT = 60;
export const KSL_BODY_POSE_INDICES = Object.freeze([11, 12, 13, 14, 15, 16, 23, 24]);
export const KSL_BODY_NODE_COUNT = KSL_BODY_POSE_INDICES.length;
export const KSL_HAND_NODE_COUNT = 21;
export const KSL_NODE_COUNT = KSL_BODY_NODE_COUNT + (KSL_HAND_NODE_COUNT * 2);
export const KSL_RAW_FRAME_WIDTH = KSL_NODE_COUNT * 3;
export const KSL_LEFT_HAND_OFFSET = KSL_BODY_NODE_COUNT;
export const KSL_RIGHT_HAND_OFFSET = KSL_BODY_NODE_COUNT + KSL_HAND_NODE_COUNT;

export const KSL_HAND_BONES = freezePairs([
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]
]);

export const KSL_BODY_GRAPH_EDGES = freezePairs([
  [0, 1], [0, 2], [2, 4], [1, 3], [3, 5],
  [0, 6], [1, 7], [6, 7]
]);

export const KSL_CROSS_HAND_POINTS = Object.freeze([0, 4, 5, 8, 9, 12, 13, 16, 17, 20]);

export const KSL_GRAPH_EDGES = freezePairs([
  ...KSL_BODY_GRAPH_EDGES,
  ...KSL_HAND_BONES.map(([source, target]) => [
    KSL_LEFT_HAND_OFFSET + source,
    KSL_LEFT_HAND_OFFSET + target
  ]),
  ...KSL_HAND_BONES.map(([source, target]) => [
    KSL_RIGHT_HAND_OFFSET + source,
    KSL_RIGHT_HAND_OFFSET + target
  ]),
  [4, KSL_LEFT_HAND_OFFSET],
  [5, KSL_RIGHT_HAND_OFFSET],
  ...KSL_CROSS_HAND_POINTS.map(point => [
    KSL_LEFT_HAND_OFFSET + point,
    KSL_RIGHT_HAND_OFFSET + point
  ])
]);

export const KSL_NODE_KINEMATIC_WIDTH = KSL_NODE_COUNT * 9;
export const KSL_GRAPH_EDGE_WIDTH = KSL_GRAPH_EDGES.length * 4;
export const KSL_MODEL_FRAME_WIDTH = KSL_NODE_KINEMATIC_WIDTH + KSL_GRAPH_EDGE_WIDTH + 2;

export const KSL_NODE_NAMES = Object.freeze([
  "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
  "left_pose_wrist", "right_pose_wrist", "left_hip", "right_hip",
  ...Array.from({ length: KSL_HAND_NODE_COUNT }, (_, index) => `left_hand_${index}`),
  ...Array.from({ length: KSL_HAND_NODE_COUNT }, (_, index) => `right_hand_${index}`)
]);

export const KSL_FEATURE_SCHEMA_V2 = deepFreeze({
  id: KSL_SCHEMA_VERSION,
  version: 2,
  targetFrameCount: KSL_TARGET_FRAME_COUNT,
  rawFrameWidth: KSL_RAW_FRAME_WIDTH,
  modelFrameWidth: KSL_MODEL_FRAME_WIDTH,
  nodeCount: KSL_NODE_COUNT,
  poseIndices: [...KSL_BODY_POSE_INDICES],
  nodeNames: [...KSL_NODE_NAMES],
  graphEdges: KSL_GRAPH_EDGES.map(edge => [...edge]),
  normalization: {
    anchor: "shoulder_midpoint",
    scale: "planar_shoulder_width",
    fallbackAnchor: "hip_midpoint_then_visible_wrist",
    fallbackScale: "shoulder_to_hip_then_one"
  },
  featureLayout: {
    nodeKinematics: {
      offset: 0,
      width: KSL_NODE_KINEMATIC_WIDTH,
      perNode: ["x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az"]
    },
    graphRelations: {
      offset: KSL_NODE_KINEMATIC_WIDTH,
      width: KSL_GRAPH_EDGE_WIDTH,
      perEdge: ["dx", "dy", "dz", "distance"]
    },
    handPresence: {
      offset: KSL_NODE_KINEMATIC_WIDTH + KSL_GRAPH_EDGE_WIDTH,
      width: 2,
      order: ["left", "right"]
    }
  }
});

// Short alias for consumers that do not need the version in the import name.
export const KSL_FEATURE_SCHEMA = KSL_FEATURE_SCHEMA_V2;

if (KSL_GRAPH_EDGES.length !== 62 || KSL_MODEL_FRAME_WIDTH !== 700) {
  throw new Error("The KSL v2 schema must contain 62 graph edges and 700 model features.");
}

/** Build one 150-value raw KSL frame from already-slotted landmarks. */
export function buildKslRawFrame({
  poseLandmarks = [],
  leftHandLandmarks = [],
  rightHandLandmarks = []
} = {}) {
  if (poseLandmarks?.length > 0 && poseLandmarks.length <= KSL_BODY_POSE_INDICES.at(-1)) {
    throw new Error("poseLandmarks must use the full 33-point MediaPipe Pose layout.");
  }

  const frame = [];
  for (const poseIndex of KSL_BODY_POSE_INDICES) {
    frame.push(...pointValues(poseLandmarks?.[poseIndex]));
  }
  appendLandmarkRange(frame, leftHandLandmarks, KSL_HAND_NODE_COUNT);
  appendLandmarkRange(frame, rightHandLandmarks, KSL_HAND_NODE_COUNT);
  return frame;
}

/**
 * Assign MediaPipe detections to semantic left/right slots without ever copying
 * one detection into both slots. Labels, pose wrists, and recent wrist positions
 * are combined so a one-frame handedness flip does not swap the sequence.
 */
export function assignStableHandSlots({
  handLandmarks = [],
  handednesses = [],
  poseLandmarks = []
} = {}, state = createKslHandSlotState(), options = {}) {
  const config = {
    swapHandednessLabels: false,
    labelMismatchCost: 0.75,
    poseWristWeight: 4,
    historyWeight: 2,
    historyMaxMissingFrames: 8,
    ...options
  };
  const slotState = ensureHandSlotState(state);
  const detections = (Array.isArray(handLandmarks) ? handLandmarks : [])
    .map((landmarks, detectionIndex) => ({
      landmarks,
      detectionIndex,
      label: handednessLabel(handednesses?.[detectionIndex], config.swapHandednessLabels),
      wrist: landmarkPoint(landmarks?.[0])
    }))
    .filter(detection => Array.isArray(detection.landmarks) && detection.landmarks.length > 0 && detection.wrist);

  const poseWrists = {
    left: landmarkPoint(poseLandmarks?.[15]),
    right: landmarkPoint(poseLandmarks?.[16])
  };
  let selectedLeft = null;
  let selectedRight = null;
  if (detections.length === 1) {
    const detection = detections[0];
    const leftCost = slotCost("left", detection, poseWrists, slotState, config);
    const rightCost = slotCost("right", detection, poseWrists, slotState, config);
    if (leftCost <= rightCost) selectedLeft = detection;
    else selectedRight = detection;
  } else if (detections.length > 1) {
    // Select the globally cheapest one-to-one assignment rather than greedily
    // trusting detection array order or duplicating a detection into both slots.
    let best = null;
    for (const leftCandidate of detections) {
      for (const rightCandidate of detections) {
        if (leftCandidate === rightCandidate) continue;
        const cost = slotCost("left", leftCandidate, poseWrists, slotState, config)
          + slotCost("right", rightCandidate, poseWrists, slotState, config);
        const tieBreaker = (leftCandidate.detectionIndex * 1000) + rightCandidate.detectionIndex;
        if (!best || cost < best.cost || (cost === best.cost && tieBreaker < best.tieBreaker)) {
          best = { left: leftCandidate, right: rightCandidate, cost, tieBreaker };
        }
      }
    }
    selectedLeft = best?.left || null;
    selectedRight = best?.right || null;
  }

  updateSlotHistory(slotState.left, selectedLeft, config.historyMaxMissingFrames);
  updateSlotHistory(slotState.right, selectedRight, config.historyMaxMissingFrames);

  return {
    leftHandLandmarks: selectedLeft?.landmarks || [],
    rightHandLandmarks: selectedRight?.landmarks || [],
    leftPresent: Boolean(selectedLeft),
    rightPresent: Boolean(selectedRight),
    state: slotState
  };
}

/** Create mutable temporal state for stable hand-slot assignment. */
export function createKslHandSlotState() {
  return {
    left: { wrist: null, missingFrames: 0 },
    right: { wrist: null, missingFrames: 0 }
  };
}

/** Convenience adapter for raw MediaPipe HandLandmarker/PoseLandmarker output. */
export function buildKslRawFrameFromMediaPipe(input = {}, state = createKslHandSlotState(), options = {}) {
  const poseLandmarks = input.poseLandmarks || input.poseResult?.landmarks?.[0] || [];
  const handLandmarks = input.handLandmarks || input.handResult?.landmarks || [];
  const handednesses = input.handednesses || input.handResult?.handednesses || [];
  const slots = assignStableHandSlots(
    { poseLandmarks, handLandmarks, handednesses },
    state,
    options
  );
  return {
    frame: buildKslRawFrame({ poseLandmarks, ...slots }),
    ...slots
  };
}

/** Validate and sanitize raw KSL frames without accepting legacy width 225. */
export function validateKslRawFrames(frames) {
  if (!Array.isArray(frames) || frames.length === 0) {
    throw new Error("A KSL sequence must contain at least one raw frame.");
  }
  return frames.map((frame, frameIndex) => {
    if (!Array.isArray(frame) && !ArrayBuffer.isView(frame)) {
      throw new Error(`KSL frame ${frameIndex} is not an array.`);
    }
    if (frame.length !== KSL_RAW_FRAME_WIDTH) {
      throw new Error(
        `KSL frame ${frameIndex} has width ${frame.length}; schema ${KSL_SCHEMA_VERSION} requires ${KSL_RAW_FRAME_WIDTH}.`
      );
    }
    return Array.from(frame, value => Number.isFinite(value) ? value : 0);
  });
}

/** Presence-aware linear resampling. Missing points are never interpolated into ghosts. */
export function resampleKslRawFrames(sourceFrames, targetFrameCount = KSL_TARGET_FRAME_COUNT) {
  const frames = validateKslRawFrames(sourceFrames);
  if (!Number.isInteger(targetFrameCount) || targetFrameCount < 2) {
    throw new Error("targetFrameCount must be an integer of at least 2.");
  }
  if (frames.length === targetFrameCount) return frames;
  if (frames.length < 2) throw new Error("At least two KSL frames are required for resampling.");

  const output = [];
  for (let targetIndex = 0; targetIndex < targetFrameCount; targetIndex++) {
    const sourcePosition = targetIndex * (frames.length - 1) / (targetFrameCount - 1);
    const lowerIndex = Math.floor(sourcePosition);
    const upperIndex = Math.min(lowerIndex + 1, frames.length - 1);
    const amount = sourcePosition - lowerIndex;
    const chosenFrame = amount < 0.5 ? frames[lowerIndex] : frames[upperIndex];
    const frame = new Array(KSL_RAW_FRAME_WIDTH).fill(0);

    for (let pointIndex = 0; pointIndex < KSL_NODE_COUNT; pointIndex++) {
      const lowerPresent = pointIsPresent(frames[lowerIndex], pointIndex);
      const upperPresent = pointIsPresent(frames[upperIndex], pointIndex);
      const offset = pointIndex * 3;
      for (let axis = 0; axis < 3; axis++) {
        frame[offset + axis] = lowerPresent && upperPresent
          ? frames[lowerIndex][offset + axis] * (1 - amount) + frames[upperIndex][offset + axis] * amount
          : chosenFrame[offset + axis];
      }
    }
    output.push(frame);
  }
  return output;
}

/** Shoulder-centered and shoulder-width-scaled position frames. */
export function normalizeKslPositionFrames(sourceFrames) {
  const frames = validateKslRawFrames(sourceFrames);
  return frames.map(frame => {
    const shoulderMidpoint = midpoint(frame, 0, 1);
    const hipMidpoint = midpoint(frame, 6, 7);
    const anchor = shoulderMidpoint || hipMidpoint || firstPresentPoint(frame, [4, 5, 8, 29]);
    let scale = planarDistance(frame, 0, 1);

    if (scale < 0.03 && shoulderMidpoint && hipMidpoint) {
      scale = Math.hypot(
        shoulderMidpoint[0] - hipMidpoint[0],
        shoulderMidpoint[1] - hipMidpoint[1]
      );
    }
    if (scale < 0.03) scale = 1;

    const normalized = [];
    for (let pointIndex = 0; pointIndex < KSL_NODE_COUNT; pointIndex++) {
      if (!pointIsPresent(frame, pointIndex)) {
        normalized.push(0, 0, 0);
        continue;
      }
      const offset = pointIndex * 3;
      const normalizedZ = pointIndex < KSL_BODY_NODE_COUNT
        ? (frame[offset + 2] - anchor[2]) / scale
        : frame[offset + 2] / scale;
      normalized.push(
        (frame[offset] - anchor[0]) / scale,
        (frame[offset + 1] - anchor[1]) / scale,
        normalizedZ
      );
    }
    return normalized;
  });
}

/** Expand normalized 150-value positions into the deterministic 700-value model schema. */
export function buildKslGraphFeatures(positionFrames) {
  const frames = validateKslRawFrames(positionFrames);
  return frames.map((frame, frameIndex) => {
    const previousFrame = frameIndex > 0 ? frames[frameIndex - 1] : null;
    const twoFramesBack = frameIndex > 1 ? frames[frameIndex - 2] : null;
    const features = [];

    for (let pointIndex = 0; pointIndex < KSL_NODE_COUNT; pointIndex++) {
      const offset = pointIndex * 3;
      const present = pointIsPresent(frame, pointIndex);
      const previousPresent = previousFrame && pointIsPresent(previousFrame, pointIndex);
      const twoBackPresent = twoFramesBack && pointIsPresent(twoFramesBack, pointIndex);
      const position = present ? frame.slice(offset, offset + 3) : [0, 0, 0];
      const velocity = present && previousPresent
        ? position.map((value, axis) => value - previousFrame[offset + axis])
        : [0, 0, 0];
      const acceleration = present && previousPresent && twoBackPresent
        ? position.map((value, axis) => value - (2 * previousFrame[offset + axis]) + twoFramesBack[offset + axis])
        : [0, 0, 0];
      features.push(...position, ...velocity, ...acceleration);
    }

    for (const [sourcePoint, targetPoint] of KSL_GRAPH_EDGES) {
      if (!pointIsPresent(frame, sourcePoint) || !pointIsPresent(frame, targetPoint)) {
        features.push(0, 0, 0, 0);
        continue;
      }
      const sourceOffset = sourcePoint * 3;
      const targetOffset = targetPoint * 3;
      const dx = frame[targetOffset] - frame[sourceOffset];
      const dy = frame[targetOffset + 1] - frame[sourceOffset + 1];
      const dz = frame[targetOffset + 2] - frame[sourceOffset + 2];
      features.push(dx, dy, dz, Math.hypot(dx, dy, dz));
    }

    features.push(
      rangeIsPresent(frame, KSL_LEFT_HAND_OFFSET, KSL_HAND_NODE_COUNT) ? 1 : 0,
      rangeIsPresent(frame, KSL_RIGHT_HAND_OFFSET, KSL_HAND_NODE_COUNT) ? 1 : 0
    );
    if (features.length !== KSL_MODEL_FRAME_WIDTH) {
      throw new Error(`Unexpected KSL feature width ${features.length}.`);
    }
    return features;
  });
}

/** Complete runtime/training preprocessing: validate -> resample -> normalize -> derive. */
export function prepareKslSequence(rawFrames, options = {}) {
  const targetFrameCount = options.targetFrameCount ?? KSL_TARGET_FRAME_COUNT;
  const checked = validateKslRawFrames(rawFrames);
  const resampled = checked.length === targetFrameCount
    ? checked
    : resampleKslRawFrames(checked, targetFrameCount);
  return buildKslGraphFeatures(normalizeKslPositionFrames(resampled));
}

/** Stateful browser helper; processFrame returns raw data and prepareSequence returns model data. */
export function createKslFeatureExtractor(options = {}) {
  let handSlotState = createKslHandSlotState();
  let rawFrames = [];
  return {
    processFrame(input = {}) {
      const extracted = buildKslRawFrameFromMediaPipe(input, handSlotState, options);
      handSlotState = extracted.state;
      rawFrames.push(extracted.frame);
      return extracted;
    },
    prepareSequence(sequenceOptions = {}) {
      return prepareKslSequence(rawFrames, { ...options, ...sequenceOptions });
    },
    getRawFrames() {
      return rawFrames.map(frame => frame.slice());
    },
    reset() {
      handSlotState = createKslHandSlotState();
      rawFrames = [];
    }
  };
}

export function pointIsPresent(frame, pointIndex) {
  const offset = pointIndex * 3;
  return !(frame[offset] === 0 && frame[offset + 1] === 0 && frame[offset + 2] === 0);
}

function rangeIsPresent(frame, startPoint, pointCount) {
  for (let pointIndex = startPoint; pointIndex < startPoint + pointCount; pointIndex++) {
    if (pointIsPresent(frame, pointIndex)) return true;
  }
  return false;
}

function midpoint(frame, firstPoint, secondPoint) {
  if (!pointIsPresent(frame, firstPoint) || !pointIsPresent(frame, secondPoint)) return null;
  const firstOffset = firstPoint * 3;
  const secondOffset = secondPoint * 3;
  return [
    (frame[firstOffset] + frame[secondOffset]) / 2,
    (frame[firstOffset + 1] + frame[secondOffset + 1]) / 2,
    (frame[firstOffset + 2] + frame[secondOffset + 2]) / 2
  ];
}

function planarDistance(frame, firstPoint, secondPoint) {
  if (!pointIsPresent(frame, firstPoint) || !pointIsPresent(frame, secondPoint)) return 0;
  const firstOffset = firstPoint * 3;
  const secondOffset = secondPoint * 3;
  return Math.hypot(
    frame[firstOffset] - frame[secondOffset],
    frame[firstOffset + 1] - frame[secondOffset + 1]
  );
}

function firstPresentPoint(frame, candidates) {
  for (const pointIndex of candidates) {
    if (pointIsPresent(frame, pointIndex)) {
      const offset = pointIndex * 3;
      return [frame[offset], frame[offset + 1], frame[offset + 2]];
    }
  }
  return [0, 0, 0];
}

function appendLandmarkRange(output, landmarks, pointCount) {
  for (let pointIndex = 0; pointIndex < pointCount; pointIndex++) {
    output.push(...pointValues(landmarks?.[pointIndex]));
  }
}

function pointValues(point) {
  if (!point) return [0, 0, 0];
  return [point.x, point.y, point.z].map(value => Number.isFinite(value) ? value : 0);
}

function landmarkPoint(point) {
  if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y)) return null;
  return {
    x: point.x,
    y: point.y,
    z: Number.isFinite(point.z) ? point.z : 0
  };
}

function handednessLabel(entry, swap) {
  let category = entry;
  if (Array.isArray(category)) category = category[0];
  if (Array.isArray(category?.categories)) category = category.categories[0];
  let label = String(category?.categoryName || category?.displayName || category?.label || "").toLowerCase();
  if (label !== "left" && label !== "right") return null;
  if (swap) label = label === "left" ? "right" : "left";
  return label;
}

function slotCost(side, detection, poseWrists, state, config) {
  let cost = detection.label && detection.label !== side ? config.labelMismatchCost : 0;
  if (poseWrists[side]) cost += distance3d(detection.wrist, poseWrists[side]) * config.poseWristWeight;
  if (state[side].wrist) cost += distance3d(detection.wrist, state[side].wrist) * config.historyWeight;
  return cost;
}

function updateSlotHistory(slot, detection, historyMaxMissingFrames) {
  if (detection) {
    slot.wrist = { ...detection.wrist };
    slot.missingFrames = 0;
    return;
  }
  slot.missingFrames += 1;
  if (slot.missingFrames > historyMaxMissingFrames) slot.wrist = null;
}

function ensureHandSlotState(state) {
  if (!state || typeof state !== "object") return createKslHandSlotState();
  if (!state.left) state.left = { wrist: null, missingFrames: 0 };
  if (!state.right) state.right = { wrist: null, missingFrames: 0 };
  return state;
}

function distance3d(first, second) {
  return Math.hypot(first.x - second.x, first.y - second.y, first.z - second.z);
}

function freezePairs(pairs) {
  return Object.freeze(pairs.map(pair => Object.freeze([...pair])));
}

function deepFreeze(value) {
  if (!value || typeof value !== "object" || Object.isFrozen(value)) return value;
  Object.freeze(value);
  for (const child of Object.values(value)) deepFreeze(child);
  return value;
}
