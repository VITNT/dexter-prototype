export const MIN_ASL_WORD_HAND_FRAMES = 3;

export function hasVisibleHandLandmarks(handData) {
  const hands = handData?.landmarks;
  return Array.isArray(hands) && hands.some(hand => Array.isArray(hand) && hand.length > 0);
}

export function shouldSubmitAslWordClip({
  clipSize,
  trackedHandFrames,
  minimumTrackedFrames = MIN_ASL_WORD_HAND_FRAMES
} = {}) {
  return Number(clipSize) > 0
    && Number(trackedHandFrames) >= Math.max(1, Number(minimumTrackedFrames) || 0);
}
