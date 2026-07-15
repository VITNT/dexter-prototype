export const LEGACY_KSL_FRAME_COUNT = 30;
export const LEGACY_KSL_FRAME_WIDTH = 126;

// Match the released whole-word model exactly: MediaPipe detection order,
// wrist-relative XYZ coordinates, two zero-padded 21-point hands.
export function buildLegacyKslHandFrame(handData) {
  const frame = new Array(LEGACY_KSL_FRAME_WIDTH).fill(0);
  const hands = handData?.landmarks || [];
  for (let handIndex = 0; handIndex < Math.min(2, hands.length); handIndex++) {
    const hand = hands[handIndex];
    const wrist = hand?.[0];
    if (!wrist) continue;
    for (let landmarkIndex = 0; landmarkIndex < Math.min(21, hand.length); landmarkIndex++) {
      const point = hand[landmarkIndex];
      const offset = handIndex * 63 + landmarkIndex * 3;
      frame[offset] = Number(point.x) - Number(wrist.x);
      frame[offset + 1] = Number(point.y) - Number(wrist.y);
      frame[offset + 2] = Number(point.z) - Number(wrist.z);
    }
  }
  return frame.map(value => Number.isFinite(value) ? value : 0);
}

export function prepareLegacyKslSequence(frames, frameCount = LEGACY_KSL_FRAME_COUNT) {
  if (!Array.isArray(frames) || frames.length < frameCount) return null;
  return frames.slice(-frameCount).map((frame, index) => {
    if (!Array.isArray(frame) || frame.length !== LEGACY_KSL_FRAME_WIDTH) {
      throw new Error(`Legacy KSL frame ${index} must contain ${LEGACY_KSL_FRAME_WIDTH} values.`);
    }
    return frame.map(value => Number.isFinite(Number(value)) ? Number(value) : 0);
  });
}
