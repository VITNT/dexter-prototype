const INDEX_MCP = 5;
const INDEX_PIP = 6;
const INDEX_DIP = 7;
const INDEX_TIP = 8;
const MIDDLE_MCP = 9;
const MIDDLE_PIP = 10;
const MIDDLE_DIP = 11;
const MIDDLE_TIP = 12;
const RING_PIP = 14;
const RING_TIP = 16;
const PINKY_MCP = 17;
const PINKY_PIP = 18;
const PINKY_TIP = 20;
const THUMB_TIP = 4;
const WRIST = 0;

const EPSILON = 1e-6;
const S_FRONT_THRESHOLD = 0.08;
const S_AMBIGUOUS_FRONT_THRESHOLD = 0.035;
const CLOSED_SLOT_MIN_MARGIN = 0.08;
const CLOSED_SLOT_MAX_DISTANCE = 0.27;

function vector(point) {
    return {
        x: Number(point?.x) || 0,
        y: Number(point?.y) || 0,
        z: Number(point?.z) || 0
    };
}

function subtract(left, right) {
    return { x: left.x - right.x, y: left.y - right.y, z: left.z - right.z };
}

function multiply(value, scalar) {
    return { x: value.x * scalar, y: value.y * scalar, z: value.z * scalar };
}

function dot(left, right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}

function cross(left, right) {
    return {
        x: left.y * right.z - left.z * right.y,
        y: left.z * right.x - left.x * right.z,
        z: left.x * right.y - left.y * right.x
    };
}

function length(value) {
    return Math.hypot(value.x, value.y, value.z);
}

function normalize(value) {
    const magnitude = length(value);
    return magnitude > EPSILON ? multiply(value, 1 / magnitude) : null;
}

function mean(values) {
    return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function validHand(hand) {
    return Array.isArray(hand)
        && hand.length >= 21
        && hand.slice(0, 21).every(point => Number.isFinite(point?.x) && Number.isFinite(point?.y));
}

// The x axis always runs anatomically from the index knuckle to the pinky
// knuckle, so mirroring does not swap T/N/M. The y axis follows the palm toward
// the fingers. The depth axis is oriented toward the camera (negative MediaPipe
// z), allowing thumb-on-top S to be separated from tucked-thumb T/N/M.
export function buildPalmCoordinateFrame(hand) {
    if (!validHand(hand)) return null;

    const origin = vector(hand[INDEX_MCP]);
    const acrossPalm = subtract(vector(hand[PINKY_MCP]), origin);
    const scale = length(acrossPalm);
    const xAxis = normalize(acrossPalm);
    if (!xAxis || scale < 0.02) return null;

    const palmUp = subtract(vector(hand[MIDDLE_MCP]), vector(hand[WRIST]));
    const palmUpWithoutX = subtract(palmUp, multiply(xAxis, dot(palmUp, xAxis)));
    const yAxis = normalize(palmUpWithoutX);
    if (!yAxis) return null;

    let zAxis = normalize(cross(xAxis, yAxis));
    if (!zAxis) return null;
    if (zAxis.z > 0) zAxis = multiply(zAxis, -1);

    return {
        scale,
        project(point) {
            const offset = subtract(vector(point), origin);
            return {
                x: dot(offset, xAxis) / scale,
                y: dot(offset, yAxis) / scale,
                z: dot(offset, zAxis) / scale
            };
        }
    };
}

export function measureClosedThumbPlacement(hand) {
    const frame = buildPalmCoordinateFrame(hand);
    if (!frame) return null;

    const thumb = frame.project(hand[THUMB_TIP]);
    const fingerTips = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
        .map(index => frame.project(hand[index]));
    const fingerPips = [INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
        .map(index => frame.project(hand[index]));
    const averageFingerTipDepth = mean(fingerTips.map(point => point.z));
    const thumbFrontness = thumb.z - averageFingerTipDepth;
    const minimumFingerTipY = Math.min(...fingerTips.map(point => point.y));

    const slots = [
        { letter: "T", x: (fingerPips[0].x + fingerPips[1].x) / 2 },
        { letter: "N", x: (fingerPips[1].x + fingerPips[2].x) / 2 },
        { letter: "M", x: (fingerPips[2].x + fingerPips[3].x) / 2 }
    ].map(slot => ({ ...slot, distance: Math.abs(thumb.x - slot.x) }))
        .sort((left, right) => left.distance - right.distance);

    return {
        thumb,
        thumbFrontness,
        minimumFingerTipY,
        indexPipX: fingerPips[0].x,
        nearestSlot: slots[0],
        secondSlot: slots[1],
        slotMargin: slots[1].distance - slots[0].distance
    };
}

export function classifyClosedThumbLetter(hand) {
    const measurement = measureClosedThumbPlacement(hand);
    if (!measurement) return null;

    const { thumb, thumbFrontness, minimumFingerTipY, indexPipX } = measurement;

    // A keeps the thumb outside the index side of the fist. E places the thumb
    // down by the curled fingertips. These guards keep the new T/N/M slots from
    // stealing the existing A and E poses.
    if (thumb.x < indexPipX - 0.12 && thumb.y > minimumFingerTipY + 0.08) return "A";
    if (thumb.y <= minimumFingerTipY + 0.05 && thumbFrontness < S_FRONT_THRESHOLD) return "E";

    if (thumbFrontness >= S_FRONT_THRESHOLD) return "S";
    if (thumbFrontness > S_AMBIGUOUS_FRONT_THRESHOLD) return null;

    if (
        measurement.nearestSlot.distance > CLOSED_SLOT_MAX_DISTANCE
        || measurement.slotMargin < CLOSED_SLOT_MIN_MARGIN
    ) {
        return null;
    }
    return measurement.nearestSlot.letter;
}

export function measureIndexMiddleCrossing(hand) {
    const frame = buildPalmCoordinateFrame(hand);
    if (!frame) return null;

    const index = [INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP]
        .map(index => frame.project(hand[index]));
    const middle = [MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP]
        .map(index => frame.project(hand[index]));
    const order = index.map((point, joint) => point.x - middle[joint].x);
    return {
        mcpOrder: order[0],
        pipOrder: order[1],
        dipOrder: order[2],
        tipOrder: order[3],
        distalCrossing: Math.max(order[2], order[3])
    };
}

export function hasCrossedIndexMiddle(hand) {
    const crossing = measureIndexMiddleCrossing(hand);
    if (!crossing || crossing.mcpOrder >= -0.08) return false;

    // Require a clear reversal at the distal joints. A small positive value at
    // only one joint is commonly landmark jitter on U; a real R reverses both
    // DIP and tip order, or has one decisive tip reversal.
    return (crossing.dipOrder > 0.02 && crossing.tipOrder > 0.02)
        || crossing.tipOrder > 0.07;
}
