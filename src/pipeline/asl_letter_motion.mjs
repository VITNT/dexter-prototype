const LANDMARK_VALUE_WIDTH = 3;
const WRIST_INDEX = 0;
const INDEX_TIP_INDEX = 8;
const PINKY_TIP_INDEX = 20;

export const ASL_DYNAMIC_WRIST_THRESHOLD = 0.08;
export const ASL_DYNAMIC_FINGERTIP_THRESHOLD = 0.045;
export const ASL_ACTIVE_WRIST_THRESHOLD = 0.03;
export const ASL_ACTIVE_FINGERTIP_THRESHOLD = 0.02;
// This lower threshold only changes the live UI while a letter is unresolved;
// it never accepts a dynamic letter or routes a clip to the model by itself.
export const ASL_PENDING_FINGERTIP_THRESHOLD = 0.015;
export const ASL_Z_PENDING_HORIZONTAL_RANGE = 0.03;
export const ASL_Z_MIN_HORIZONTAL_RANGE = 0.065;
export const ASL_Z_MIN_VERTICAL_RANGE = 0.025;
export const ASL_Z_MIN_HORIZONTAL_PATH = 0.16;
export const ASL_Z_REVERSAL_EXCURSION = 0.025;
export const ASL_Z_MIN_REVERSALS = 2;

function readPoint(frame, landmarkIndex) {
    const offset = landmarkIndex * LANDMARK_VALUE_WIDTH;
    const x = frame?.[offset];
    const y = frame?.[offset + 1];
    return Number.isFinite(x) && Number.isFinite(y) ? { x, y } : null;
}

function pointSpan(points) {
    if (points.length < 2) return 0;

    let minX = points[0].x;
    let maxX = points[0].x;
    let minY = points[0].y;
    let maxY = points[0].y;

    for (const point of points.slice(1)) {
        minX = Math.min(minX, point.x);
        maxX = Math.max(maxX, point.x);
        minY = Math.min(minY, point.y);
        maxY = Math.max(maxY, point.y);
    }

    return Math.hypot(maxX - minX, maxY - minY);
}

function landmarkSpan(frames, landmarkIndex, { relativeToWrist = false } = {}) {
    const points = [];

    for (const frame of frames || []) {
        const point = readPoint(frame, landmarkIndex);
        if (!point) continue;

        if (relativeToWrist) {
            const wrist = readPoint(frame, WRIST_INDEX);
            if (!wrist) continue;
            points.push({ x: point.x - wrist.x, y: point.y - wrist.y });
        } else {
            points.push(point);
        }
    }

    return pointSpan(points);
}

function relativeLandmarkPoints(frames, landmarkIndex) {
    const points = [];
    for (const frame of frames || []) {
        const point = readPoint(frame, landmarkIndex);
        const wrist = readPoint(frame, WRIST_INDEX);
        if (!point || !wrist) continue;
        points.push({ x: point.x - wrist.x, y: point.y - wrist.y });
    }
    return points;
}

function smoothPoints(points) {
    if (points.length < 3) return points.slice();
    return points.map((point, index) => {
        const start = Math.max(0, index - 1);
        const end = Math.min(points.length - 1, index + 1);
        let x = 0;
        let y = 0;
        let count = 0;
        for (let sample = start; sample <= end; sample++) {
            x += points[sample].x;
            y += points[sample].y;
            count++;
        }
        return { x: x / count, y: y / count };
    });
}

function countMeaningfulReversals(values, minimumExcursion) {
    if (values.length < 3) return 0;
    let anchor = values[0];
    let extreme = anchor;
    let direction = 0;
    let reversals = 0;

    for (const value of values.slice(1)) {
        if (direction === 0) {
            const delta = value - anchor;
            if (Math.abs(delta) >= minimumExcursion) {
                direction = Math.sign(delta);
                extreme = value;
            }
            continue;
        }

        if (direction > 0) {
            if (value > extreme) extreme = value;
            else if (extreme - value >= minimumExcursion) {
                reversals++;
                direction = -1;
                extreme = value;
            }
        } else if (value < extreme) {
            extreme = value;
        } else if (value - extreme >= minimumExcursion) {
            reversals++;
            direction = 1;
            extreme = value;
        }
    }
    return reversals;
}

// J and Z can be drawn mostly by rotating the hand, leaving the wrist nearly
// stationary. Track the signing fingertips relative to the wrist so that motion
// is not mistaken for the static I or D handshapes.
export function measureAslLetterMotion(frames) {
    const wristSpan = landmarkSpan(frames, WRIST_INDEX);
    const indexTipSpan = landmarkSpan(frames, INDEX_TIP_INDEX, { relativeToWrist: true });
    const pinkyTipSpan = landmarkSpan(frames, PINKY_TIP_INDEX, { relativeToWrist: true });

    return {
        wristSpan,
        indexTipSpan,
        pinkyTipSpan,
        fingertipSpan: Math.max(indexTipSpan, pinkyTipSpan)
    };
}

export function hasDynamicAslLetterMotion(frames, {
    wristThreshold = ASL_DYNAMIC_WRIST_THRESHOLD,
    fingertipThreshold = ASL_DYNAMIC_FINGERTIP_THRESHOLD
} = {}) {
    const motion = measureAslLetterMotion(frames);
    return motion.wristSpan > wristThreshold || motion.fingertipSpan > fingertipThreshold;
}

export function measureAslZTrajectory(frames) {
    const points = smoothPoints(relativeLandmarkPoints(frames, INDEX_TIP_INDEX));
    if (points.length < 3) {
        return {
            pointCount: points.length,
            horizontalRange: 0,
            verticalRange: 0,
            horizontalPath: 0,
            verticalPath: 0,
            horizontalReversals: 0
        };
    }

    const xs = points.map(point => point.x);
    const ys = points.map(point => point.y);
    let horizontalPath = 0;
    let verticalPath = 0;
    for (let index = 1; index < points.length; index++) {
        horizontalPath += Math.abs(points[index].x - points[index - 1].x);
        verticalPath += Math.abs(points[index].y - points[index - 1].y);
    }

    return {
        pointCount: points.length,
        horizontalRange: Math.max(...xs) - Math.min(...xs),
        verticalRange: Math.max(...ys) - Math.min(...ys),
        horizontalPath,
        verticalPath,
        horizontalReversals: countMeaningfulReversals(xs, ASL_Z_REVERSAL_EXCURSION)
    };
}

export function hasAslZTrajectory(frames, {
    minimumHorizontalRange = ASL_Z_MIN_HORIZONTAL_RANGE,
    minimumVerticalRange = ASL_Z_MIN_VERTICAL_RANGE,
    minimumHorizontalPath = ASL_Z_MIN_HORIZONTAL_PATH,
    minimumReversals = ASL_Z_MIN_REVERSALS
} = {}) {
    const trajectory = measureAslZTrajectory(frames);
    return trajectory.horizontalRange >= minimumHorizontalRange
        && trajectory.verticalRange >= minimumVerticalRange
        && trajectory.horizontalPath >= minimumHorizontalPath
        && trajectory.horizontalReversals >= minimumReversals;
}

export function getAslDynamicLetterIntent(frames) {
    const motion = measureAslLetterMotion(frames);
    return {
        J: motion.pinkyTipSpan > ASL_DYNAMIC_FINGERTIP_THRESHOLD,
        Z: hasAslZTrajectory(frames)
    };
}

export function hasEmergingAslLetterMotion(frames) {
    const motion = measureAslLetterMotion(frames);
    const zTrajectory = measureAslZTrajectory(frames);
    return motion.pinkyTipSpan > ASL_PENDING_FINGERTIP_THRESHOLD
        || zTrajectory.horizontalRange >= ASL_Z_PENDING_HORIZONTAL_RANGE;
}
