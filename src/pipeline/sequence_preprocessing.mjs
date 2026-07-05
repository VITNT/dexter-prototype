export const DEFAULT_SEQUENCE_OPTIONS = Object.freeze({
  targetFrameCount: 30,
  expectedFrameWidth: 225,
  valuesPerLandmark: 3,
  missingValue: 0,
  anchorCandidates: [0, 99, 162]
});

export function prepareSequenceForModel(frames, options = {}) {
  const config = { ...DEFAULT_SEQUENCE_OPTIONS, ...options };
  const checked = validateFrames(frames, config.expectedFrameWidth);
  const filled = forwardFillMissingLandmarks(checked, config);
  const normalized = normalizeFrames(filled, config);
  return resampleFrames(normalized, config.targetFrameCount);
}

export function validateFrames(frames, expectedFrameWidth = DEFAULT_SEQUENCE_OPTIONS.expectedFrameWidth) {
  if (!Array.isArray(frames) || frames.length === 0) {
    throw new Error("A sign sequence must contain at least one frame.");
  }

  return frames.map((frame, index) => {
    if (!Array.isArray(frame)) {
      throw new Error(`Frame ${index} is not an array.`);
    }
    if (frame.length !== expectedFrameWidth) {
      throw new Error(`Frame ${index} has width ${frame.length}; expected ${expectedFrameWidth}.`);
    }
    return frame.map(value => Number.isFinite(value) ? value : 0);
  });
}

export function forwardFillMissingLandmarks(frames, options = {}) {
  const config = { ...DEFAULT_SEQUENCE_OPTIONS, ...options };
  const output = frames.map(frame => frame.slice());

  for (let frameIndex = 1; frameIndex < output.length; frameIndex++) {
    for (let offset = 0; offset < output[frameIndex].length; offset += config.valuesPerLandmark) {
      const x = output[frameIndex][offset];
      const y = output[frameIndex][offset + 1];

      if (x === config.missingValue && y === config.missingValue) {
        output[frameIndex][offset] = output[frameIndex - 1][offset];
        output[frameIndex][offset + 1] = output[frameIndex - 1][offset + 1];
        output[frameIndex][offset + 2] = output[frameIndex - 1][offset + 2];
      }
    }
  }

  return output;
}

export function normalizeFrames(frames, options = {}) {
  const config = { ...DEFAULT_SEQUENCE_OPTIONS, ...options };

  return frames.map(frame => {
    const anchor = findAnchor(frame, config);
    if (!anchor) return frame.slice();

    const normalized = [];
    for (let offset = 0; offset < frame.length; offset += config.valuesPerLandmark) {
      const x = frame[offset];
      const y = frame[offset + 1];
      const z = frame[offset + 2];

      if (x === config.missingValue && y === config.missingValue) {
        normalized.push(0, 0, 0);
      } else {
        normalized.push(x - anchor.x, y - anchor.y, z);
      }
    }

    return normalized;
  });
}

export function resampleFrames(frames, targetFrameCount = DEFAULT_SEQUENCE_OPTIONS.targetFrameCount) {
  if (frames.length === targetFrameCount) {
    return frames.map(frame => frame.slice());
  }

  if (targetFrameCount < 2) {
    throw new Error("targetFrameCount must be at least 2.");
  }

  if (frames.length === 1) {
    return Array.from({ length: targetFrameCount }, () => frames[0].slice());
  }

  const lastSourceIndex = frames.length - 1;
  const lastTargetIndex = targetFrameCount - 1;
  const output = [];

  for (let targetIndex = 0; targetIndex < targetFrameCount; targetIndex++) {
    const sourcePosition = (targetIndex / lastTargetIndex) * lastSourceIndex;
    const leftIndex = Math.floor(sourcePosition);
    const rightIndex = Math.min(leftIndex + 1, lastSourceIndex);
    const ratio = sourcePosition - leftIndex;
    output.push(interpolateFrame(frames[leftIndex], frames[rightIndex], ratio));
  }

  return output;
}

export function flattenLandmarks(landmarks, width = 63) {
  const frame = new Array(width).fill(0);
  if (!Array.isArray(landmarks)) return frame;

  for (let index = 0; index < landmarks.length && index * 3 + 2 < width; index++) {
    frame[index * 3] = landmarks[index]?.x ?? 0;
    frame[index * 3 + 1] = landmarks[index]?.y ?? 0;
    frame[index * 3 + 2] = landmarks[index]?.z ?? 0;
  }

  return frame;
}

export function buildHolisticFrame({ poseLandmarks = [], leftHandLandmarks = [], rightHandLandmarks = [] } = {}) {
  return [
    ...flattenLandmarks(poseLandmarks, 99),
    ...flattenLandmarks(leftHandLandmarks, 63),
    ...flattenLandmarks(rightHandLandmarks, 63)
  ];
}

function findAnchor(frame, config) {
  for (const offset of config.anchorCandidates) {
    const x = frame[offset];
    const y = frame[offset + 1];
    if (Number.isFinite(x) && Number.isFinite(y) && !(x === 0 && y === 0)) {
      return { x, y };
    }
  }
  return null;
}

function interpolateFrame(leftFrame, rightFrame, ratio) {
  if (ratio === 0) return leftFrame.slice();

  return leftFrame.map((leftValue, index) => {
    const rightValue = rightFrame[index];
    return leftValue * (1 - ratio) + rightValue * ratio;
  });
}
