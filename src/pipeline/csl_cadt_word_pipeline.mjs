export const CADT_CSL_FRAME_COUNT = 30;
export const CADT_CSL_FRAME_WIDTH = 126;

function assignStableHandSlots(handData) {
  const slots = [null, null];
  const unassigned = [];
  const hands = handData?.landmarks || [];

  for (let index = 0; index < hands.length; index++) {
    const category = handData?.handednesses?.[index]?.[0];
    const label = String(category?.categoryName || category?.displayName || "").toLowerCase();
    const slot = label === "left" ? 0 : (label === "right" ? 1 : -1);
    if (slot >= 0 && !slots[slot]) slots[slot] = hands[index];
    else unassigned.push(hands[index]);
  }

  for (const hand of unassigned) {
    const emptySlot = slots.indexOf(null);
    if (emptySlot < 0) break;
    slots[emptySlot] = hand;
  }
  return slots;
}

// Match the released whole-word model exactly: MediaPipe detection order,
// wrist-relative XYZ coordinates, two zero-padded 21-point hands.
export function buildCadtCslHandFrame(handData) {
  const frame = new Array(CADT_CSL_FRAME_WIDTH).fill(0);
  // MediaPipe's detection-array order can change between frames. Prefer its
  // handedness labels so each physical hand stays in one model input slot.
  // Unknown hands retain detection order as a compatibility fallback.
  const hands = assignStableHandSlots(handData);
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

export function interpolateCadtCslFrame(startFrame, endFrame, ratio) {
  if (!Array.isArray(startFrame) || !Array.isArray(endFrame)
      || startFrame.length !== CADT_CSL_FRAME_WIDTH
      || endFrame.length !== CADT_CSL_FRAME_WIDTH) {
    throw new Error(`CADT CSL interpolation requires two ${CADT_CSL_FRAME_WIDTH}-value frames.`);
  }
  const amount = Math.max(0, Math.min(1, Number(ratio) || 0));
  return startFrame.map((start, index) => {
    const from = Number(start);
    const to = Number(endFrame[index]);
    if (!Number.isFinite(from) || !Number.isFinite(to)) return 0;
    return from + (to - from) * amount;
  });
}

export function prepareCadtCslSequence(frames, frameCount = CADT_CSL_FRAME_COUNT) {
  if (!Array.isArray(frames) || frames.length < frameCount) return null;
  return frames.slice(-frameCount).map((frame, index) => {
    if (!Array.isArray(frame) || frame.length !== CADT_CSL_FRAME_WIDTH) {
      throw new Error(`CADT CSL frame ${index} must contain ${CADT_CSL_FRAME_WIDTH} values.`);
    }
    return frame.map(value => Number.isFinite(Number(value)) ? Number(value) : 0);
  });
}
