export const RELEASE_LANDMARK_INDICES = [0, 4, 8, 12, 16, 20];

export function buildHandSignature(handData) {
    return (handData?.landmarks || [])
        .filter(hand => hand?.length)
        .map(hand => RELEASE_LANDMARK_INDICES.flatMap(index => [
            hand[index]?.x || 0,
            hand[index]?.y || 0
        ]))
        .sort((left, right) => left[0] - right[0]);
}

export function copyHandSignature(signature) {
    return signature.map(hand => hand.slice());
}

// Remove hand position and scale so ordinary camera sway does not look like a
// deliberate release. Only a meaningful finger-configuration change remains.
export function normalizeHandShapeSignature(signature) {
    return signature.map(hand => {
        const wristX = hand[0];
        const wristY = hand[1];
        const centered = [];
        let scale = 0;

        for (let index = 0; index < hand.length; index += 2) {
            const x = hand[index] - wristX;
            const y = hand[index + 1] - wristY;
            centered.push(x, y);
            scale = Math.max(scale, Math.hypot(x, y));
        }

        const safeScale = scale > 0.0001 ? scale : 1;
        return centered.map(value => value / safeScale);
    });
}

export function handSignatureDistance(reference, current) {
    if (!reference.length || !current.length || reference.length !== current.length) return Infinity;
    let distance = 0;
    let points = 0;
    for (let handIndex = 0; handIndex < reference.length; handIndex++) {
        for (let index = 0; index < reference[handIndex].length; index += 2) {
            distance += Math.hypot(
                current[handIndex][index] - reference[handIndex][index],
                current[handIndex][index + 1] - reference[handIndex][index + 1]
            );
            points++;
        }
    }
    return points ? distance / points : Infinity;
}

export function updatePauseProgress({
    elapsedMs,
    deltaMs,
    isPaused,
    durationMs,
    maxTickMs = 100,
    movementDecay = 2
}) {
    const delta = Math.max(0, Math.min(maxTickMs, deltaMs));
    return isPaused
        ? Math.min(durationMs, elapsedMs + delta)
        : Math.max(0, elapsedMs - delta * movementDecay);
}

// Dynamic letters finish in a valid static handshape (J finishes as I and Z
// finishes as D). A pause is therefore not a release: the hand must leave that
// final shape continuously before static-letter recognition can be armed again.
export function updateReleaseProgress({
    elapsedMs,
    deltaMs,
    isReleased,
    durationMs,
    maxTickMs = 100
}) {
    const delta = Math.max(0, Math.min(maxTickMs, deltaMs));
    return isReleased
        ? Math.min(durationMs, elapsedMs + delta)
        : 0;
}
