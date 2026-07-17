export function hasVisibleHandLandmarks(handData) {
  const hands = handData?.landmarks;
  return Array.isArray(hands) && hands.some(hand => Array.isArray(hand) && hand.length > 0);
}
