// ==========================================
// DEXTER TRAINING FACTORY
// ==========================================

const ASL_SEQUENCE_LENGTH = 30;
const KSL_SEQUENCE_LENGTH = 60;
const KSL_SCHEMA_VERSION = 'ksl-landmarks-v2';
const KSL_RAW_FRAME_WIDTH = 150;
const LEGACY_HOLISTIC_FRAME_WIDTH = 225;
const ASL_FRAME_WIDTH = 63;
const MIN_RAW_RECORDINGS_PER_CLASS = 5;
const VALIDATION_FRACTION = 0.2;

const TCN_DROPOUT_RATE = 0.35;
const TCN_L2_RATE = 0.0005;
const LABEL_SMOOTHING = 0.04;

const KSL_BODY_POSE_INDICES = [11, 12, 13, 14, 15, 16, 23, 24];
const KSL_BODY_NODE_COUNT = KSL_BODY_POSE_INDICES.length;
const HAND_NODE_COUNT = 21;
const KSL_NODE_COUNT = KSL_BODY_NODE_COUNT + (HAND_NODE_COUNT * 2);
const KSL_NODE_KINEMATIC_WIDTH = KSL_NODE_COUNT * 9;
const VISUAL_HEIGHT = 24;
const VISUAL_WIDTH = 48;
const VISUAL_CHANNELS = 1;
const VISUAL_FRAME_SIZE = VISUAL_HEIGHT * VISUAL_WIDTH * VISUAL_CHANNELS;

const HAND_BONES = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]
];

// Node order: 8 upper-body nodes, 21 left-hand nodes, 21 right-hand nodes.
// These explicit edges make the built-in-layer encoder graph-aware without a
// custom TensorFlow.js layer, so downloaded models remain serializable.
const BODY_GRAPH_EDGES = [
    [0, 1], [0, 2], [2, 4], [1, 3], [3, 5],
    [0, 6], [1, 7], [6, 7]
];
const LEFT_HAND_OFFSET = KSL_BODY_NODE_COUNT;
const RIGHT_HAND_OFFSET = KSL_BODY_NODE_COUNT + HAND_NODE_COUNT;
const CROSS_HAND_POINTS = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20];
const KSL_GRAPH_EDGES = [
    ...BODY_GRAPH_EDGES,
    ...HAND_BONES.map(([a, b]) => [LEFT_HAND_OFFSET + a, LEFT_HAND_OFFSET + b]),
    ...HAND_BONES.map(([a, b]) => [RIGHT_HAND_OFFSET + a, RIGHT_HAND_OFFSET + b]),
    [4, LEFT_HAND_OFFSET],
    [5, RIGHT_HAND_OFFSET],
    ...CROSS_HAND_POINTS.map(pointIndex => [
        LEFT_HAND_OFFSET + pointIndex,
        RIGHT_HAND_OFFSET + pointIndex
    ])
];
const KSL_GRAPH_EDGE_WIDTH = KSL_GRAPH_EDGES.length * 4;
const KSL_MODEL_FRAME_WIDTH = KSL_NODE_KINEMATIC_WIDTH + KSL_GRAPH_EDGE_WIDTH + 2;

if (KSL_MODEL_FRAME_WIDTH !== 700) {
    throw new Error(`Unexpected KSL feature width: ${KSL_MODEL_FRAME_WIDTH}.`);
}

function getSequenceLength(modelFamily) {
    return modelFamily === 'asl' ? ASL_SEQUENCE_LENGTH : KSL_SEQUENCE_LENGTH;
}

function getTcnKernelSizes(sequenceLength) {
    return sequenceLength > ASL_SEQUENCE_LENGTH
        ? [3, 5, 9, 17, 33]
        : [3, 5, 9, 17];
}

// TensorFlow.js 4.10 does not provide serializable causal padding for Conv1D.
// A one-column image plus built-in ZeroPadding2D/Conv2D gives the same result.
function applyCausalTemporalConvolution(input, {
    filters,
    kernelSize,
    sequenceLength,
    name
}) {
    const inputChannels = input.shape[input.shape.length - 1];
    const leftPadding = kernelSize - 1;

    let output = tf.layers.reshape({
        targetShape: [sequenceLength, 1, inputChannels],
        name: `${name}_expand`
    }).apply(input);
    output = tf.layers.zeroPadding2d({
        padding: [[leftPadding, 0], [0, 0]],
        name: `${name}_causal_padding`
    }).apply(output);
    output = tf.layers.conv2d({
        filters,
        kernelSize: [kernelSize, 1],
        padding: 'valid',
        kernelRegularizer: tf.regularizers.l2({ l2: TCN_L2_RATE }),
        name: `${name}_kernel`
    }).apply(output);

    return tf.layers.reshape({
        targetShape: [sequenceLength, filters],
        name: `${name}_squeeze`
    }).apply(output);
}

function applyTcnStack(input, { filters, sequenceLength, namePrefix }) {
    let output = input;
    getTcnKernelSizes(sequenceLength).forEach((kernelSize, blockIndex) => {
        const residual = output;
        const blockName = `${namePrefix}_block_${blockIndex + 1}`;

        output = applyCausalTemporalConvolution(output, {
            filters,
            kernelSize,
            sequenceLength,
            name: `${blockName}_conv`
        });
        output = tf.layers.activation({
            activation: 'relu',
            name: `${blockName}_relu`
        }).apply(output);
        output = tf.layers.dropout({
            rate: TCN_DROPOUT_RATE,
            name: `${blockName}_dropout`
        }).apply(output);
        output = tf.layers.add({
            name: `${blockName}_residual`
        }).apply([residual, output]);
        output = tf.layers.activation({
            activation: 'relu',
            name: `${blockName}_output`
        }).apply(output);
    });
    return output;
}

function buildAslTcnClassifier(numClasses) {
    const input = tf.input({
        shape: [ASL_SEQUENCE_LENGTH, ASL_FRAME_WIDTH],
        name: 'landmark_sequence'
    });
    let output = tf.layers.conv1d({
        filters: 64,
        kernelSize: 1,
        padding: 'same',
        kernelRegularizer: tf.regularizers.l2({ l2: TCN_L2_RATE }),
        name: 'input_projection'
    }).apply(input);
    output = applyTcnStack(output, {
        filters: 64,
        sequenceLength: ASL_SEQUENCE_LENGTH,
        namePrefix: 'tcn'
    });
    output = tf.layers.globalAveragePooling1d({ name: 'temporal_pooling' }).apply(output);
    output = tf.layers.dense({
        units: 64,
        activation: 'relu',
        kernelRegularizer: tf.regularizers.l2({ l2: TCN_L2_RATE }),
        name: 'classification_features'
    }).apply(output);
    output = tf.layers.dropout({
        rate: TCN_DROPOUT_RATE,
        name: 'classification_dropout'
    }).apply(output);
    const prediction = tf.layers.dense({
        units: numClasses,
        activation: 'softmax',
        name: 'class_probabilities'
    }).apply(output);

    return tf.model({
        inputs: input,
        outputs: prediction,
        name: 'legacy_asl_tcn_classifier'
    });
}

function buildKslLandmarkEmbedding(landmarkInput) {
    // Every frame already contains graph-edge vectors/distances. Shared 1x1
    // projections learn a spatial representation before any temporal mixing.
    let output = tf.layers.conv1d({
        filters: 128,
        kernelSize: 1,
        padding: 'same',
        activation: 'relu',
        kernelRegularizer: tf.regularizers.l2({ l2: TCN_L2_RATE }),
        name: 'spatial_graph_projection_1'
    }).apply(landmarkInput);
    output = tf.layers.dropout({ rate: 0.2, name: 'spatial_graph_dropout' }).apply(output);
    output = tf.layers.conv1d({
        filters: 96,
        kernelSize: 1,
        padding: 'same',
        activation: 'relu',
        kernelRegularizer: tf.regularizers.l2({ l2: TCN_L2_RATE }),
        name: 'spatial_graph_projection_2'
    }).apply(output);
    output = applyTcnStack(output, {
        filters: 96,
        sequenceLength: KSL_SEQUENCE_LENGTH,
        namePrefix: 'landmark_tcn'
    });
    output = tf.layers.globalAveragePooling1d({
        name: 'landmark_temporal_pooling'
    }).apply(output);
    return tf.layers.dense({
        units: 96,
        activation: 'relu',
        kernelRegularizer: tf.regularizers.l2({ l2: TCN_L2_RATE }),
        name: 'landmark_embedding'
    }).apply(output);
}

function buildKslVisualEmbedding(visualInput) {
    if (typeof tf.layers.timeDistributed !== 'function') {
        throw new Error('This TensorFlow.js build does not provide TimeDistributed layers. Train without visualFrames or upgrade TensorFlow.js.');
    }

    let output = tf.layers.timeDistributed({
        layer: tf.layers.conv2d({
            filters: 12,
            kernelSize: [3, 3],
            padding: 'same',
            activation: 'relu',
            kernelRegularizer: tf.regularizers.l2({ l2: TCN_L2_RATE })
        }),
        name: 'hand_crop_conv_1'
    }).apply(visualInput);
    output = tf.layers.timeDistributed({
        layer: tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }),
        name: 'hand_crop_pool_1'
    }).apply(output);
    output = tf.layers.timeDistributed({
        layer: tf.layers.conv2d({
            filters: 24,
            kernelSize: [3, 3],
            padding: 'same',
            activation: 'relu',
            kernelRegularizer: tf.regularizers.l2({ l2: TCN_L2_RATE })
        }),
        name: 'hand_crop_conv_2'
    }).apply(output);
    output = tf.layers.timeDistributed({
        layer: tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }),
        name: 'hand_crop_pool_2'
    }).apply(output);
    output = tf.layers.timeDistributed({
        layer: tf.layers.globalAveragePooling2d({}),
        name: 'hand_crop_spatial_pool'
    }).apply(output);
    output = tf.layers.conv1d({
        filters: 48,
        kernelSize: 5,
        padding: 'same',
        activation: 'relu',
        kernelRegularizer: tf.regularizers.l2({ l2: TCN_L2_RATE }),
        name: 'hand_crop_temporal_conv'
    }).apply(output);
    output = tf.layers.dropout({ rate: 0.3, name: 'hand_crop_temporal_dropout' }).apply(output);
    output = tf.layers.globalAveragePooling1d({
        name: 'hand_crop_temporal_pooling'
    }).apply(output);
    return tf.layers.dense({
        units: 64,
        activation: 'relu',
        kernelRegularizer: tf.regularizers.l2({ l2: TCN_L2_RATE }),
        name: 'hand_crop_embedding'
    }).apply(output);
}

function buildKslGraphTcnClassifier(numClasses, usesHandCrops) {
    const landmarkInput = tf.input({
        shape: [KSL_SEQUENCE_LENGTH, KSL_MODEL_FRAME_WIDTH],
        name: 'ksl_graph_sequence'
    });
    const landmarkEmbedding = buildKslLandmarkEmbedding(landmarkInput);
    const inputs = [landmarkInput];
    let fusedEmbedding = landmarkEmbedding;

    if (usesHandCrops) {
        const visualInput = tf.input({
            shape: [KSL_SEQUENCE_LENGTH, VISUAL_HEIGHT, VISUAL_WIDTH, VISUAL_CHANNELS],
            name: 'two_hand_crops'
        });
        inputs.push(visualInput);
        const visualEmbedding = buildKslVisualEmbedding(visualInput);
        fusedEmbedding = tf.layers.concatenate({ name: 'late_modal_fusion' })
            .apply([landmarkEmbedding, visualEmbedding]);
    }

    let output = tf.layers.dense({
        units: 128,
        activation: 'relu',
        kernelRegularizer: tf.regularizers.l2({ l2: TCN_L2_RATE }),
        name: 'fusion_features'
    }).apply(fusedEmbedding);
    output = tf.layers.dropout({
        rate: 0.4,
        name: 'fusion_dropout'
    }).apply(output);
    const prediction = tf.layers.dense({
        units: numClasses,
        activation: 'softmax',
        name: 'class_probabilities'
    }).apply(output);

    return tf.model({
        inputs: usesHandCrops ? inputs : landmarkInput,
        outputs: prediction,
        name: usesHandCrops
            ? 'ksl_graph_tcn_hand_crop_fusion'
            : 'ksl_graph_tcn_classifier'
    });
}

// Focal cross-entropy keeps correctly classified/easy clips from dominating the
// gradient, so visually similar hard negatives continue to influence training.
// This is a training-only function; the saved inference graph uses built-in layers.
function focalCategoricalCrossentropy(yTrue, yPred) {
    return tf.tidy(() => {
        const clipped = tf.clipByValue(yPred, 1e-7, 1 - 1e-7);
        const trueClassProbability = tf.sum(tf.mul(yTrue, clipped), -1);
        const crossEntropy = tf.neg(tf.sum(tf.mul(yTrue, tf.log(clipped)), -1));
        const hardExampleWeight = tf.pow(tf.sub(1, trueClassProbability), 2);
        return tf.mul(hardExampleWeight, crossEntropy);
    });
}

function pointIsPresent(frame, pointIndex) {
    const offset = pointIndex * 3;
    return !(frame[offset] === 0 && frame[offset + 1] === 0 && frame[offset + 2] === 0);
}

function rangeIsPresent(frame, startPoint, pointCount) {
    for (let point = startPoint; point < startPoint + pointCount; point++) {
        if (pointIsPresent(frame, point)) return true;
    }
    return false;
}

function forwardFillRange(frames, startPoint, pointCount) {
    for (let frameIndex = 1; frameIndex < frames.length; frameIndex++) {
        if (rangeIsPresent(frames[frameIndex], startPoint, pointCount)) continue;
        const start = startPoint * 3;
        const end = (startPoint + pointCount) * 3;
        for (let valueIndex = start; valueIndex < end; valueIndex++) {
            frames[frameIndex][valueIndex] = frames[frameIndex - 1][valueIndex];
        }
    }
}

function sanitizeData(sourceFrames) {
    if (!Array.isArray(sourceFrames) || sourceFrames.length === 0) return sourceFrames;
    const frames = sourceFrames.map(frame => [...frame]);
    const frameLength = frames[0].length;

    // Keep KSL missing-hand frames as zero. Hand-presence channels are part of
    // the new schema and must distinguish genuine one-hand signs from dropouts.
    if (frameLength === ASL_FRAME_WIDTH) forwardFillRange(frames, 0, HAND_NODE_COUNT);
    return frames;
}

function resampleFrames(frames, targetLength) {
    if (frames.length === targetLength) return frames.map(frame => [...frame]);
    if (frames.length < 2) throw new Error('A recording needs at least two frames to be resampled.');

    const result = [];
    for (let targetIndex = 0; targetIndex < targetLength; targetIndex++) {
        const sourcePosition = targetIndex * (frames.length - 1) / (targetLength - 1);
        const lowerIndex = Math.floor(sourcePosition);
        const upperIndex = Math.min(lowerIndex + 1, frames.length - 1);
        const amount = sourcePosition - lowerIndex;
        const frame = new Array(frames[0].length).fill(0);
        for (let pointIndex = 0; pointIndex < frame.length / 3; pointIndex++) {
            const lowerPresent = pointIsPresent(frames[lowerIndex], pointIndex);
            const upperPresent = pointIsPresent(frames[upperIndex], pointIndex);
            const chosenIndex = amount < 0.5 ? lowerIndex : upperIndex;
            for (let axis = 0; axis < 3; axis++) {
                const valueIndex = (pointIndex * 3) + axis;
                frame[valueIndex] = lowerPresent && upperPresent
                    ? frames[lowerIndex][valueIndex] * (1 - amount) + frames[upperIndex][valueIndex] * amount
                    : frames[chosenIndex][valueIndex];
            }
        }
        result.push(frame);
    }
    return result;
}

function extractKslPositionFrames(frames) {
    if (frames[0].length === KSL_RAW_FRAME_WIDTH) {
        return frames.map(frame => [...frame]);
    }

    return frames.map(frame => {
        const reduced = [];
        KSL_BODY_POSE_INDICES.forEach(poseIndex => {
            const offset = poseIndex * 3;
            reduced.push(frame[offset], frame[offset + 1], frame[offset + 2]);
        });
        reduced.push(...frame.slice(99, 162));
        reduced.push(...frame.slice(162, 225));
        return reduced;
    });
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

function normalizeKslPositionFrames(frames) {
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

function normalizeAslFrames(frames) {
    return frames.map(frame => {
        let anchorOffset = 0;
        if (!pointIsPresent(frame, 0)) {
            const firstVisible = Array.from({ length: HAND_NODE_COUNT }, (_, index) => index)
                .find(pointIndex => pointIsPresent(frame, pointIndex));
            anchorOffset = (firstVisible || 0) * 3;
        }
        const anchorX = frame[anchorOffset];
        const anchorY = frame[anchorOffset + 1];
        return frame.flatMap((value, valueIndex) => {
            if (valueIndex % 3 !== 0) return [];
            const pointIndex = valueIndex / 3;
            if (!pointIsPresent(frame, pointIndex)) return [0, 0, 0];
            return [
                frame[valueIndex] - anchorX,
                frame[valueIndex + 1] - anchorY,
                frame[valueIndex + 2]
            ];
        });
    });
}

function buildKslGraphFeatures(positionFrames) {
    return positionFrames.map((frame, frameIndex) => {
        const previousFrame = frameIndex > 0 ? positionFrames[frameIndex - 1] : null;
        const twoFramesBack = frameIndex > 1 ? positionFrames[frameIndex - 2] : null;
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
                ? position.map((value, axis) =>
                    value - (2 * previousFrame[offset + axis]) + twoFramesBack[offset + axis]
                )
                : [0, 0, 0];
            features.push(...position, ...velocity, ...acceleration);
        }

        KSL_GRAPH_EDGES.forEach(([sourcePoint, targetPoint]) => {
            if (!pointIsPresent(frame, sourcePoint) || !pointIsPresent(frame, targetPoint)) {
                features.push(0, 0, 0, 0);
                return;
            }
            const sourceOffset = sourcePoint * 3;
            const targetOffset = targetPoint * 3;
            const dx = frame[targetOffset] - frame[sourceOffset];
            const dy = frame[targetOffset + 1] - frame[sourceOffset + 1];
            const dz = frame[targetOffset + 2] - frame[sourceOffset + 2];
            features.push(dx, dy, dz, Math.hypot(dx, dy, dz));
        });

        features.push(
            rangeIsPresent(frame, LEFT_HAND_OFFSET, HAND_NODE_COUNT) ? 1 : 0,
            rangeIsPresent(frame, RIGHT_HAND_OFFSET, HAND_NODE_COUNT) ? 1 : 0
        );
        return features;
    });
}

// ==========================================
// DATA AUGMENTATION (TRAINING ONLY)
// ==========================================

function timeWarp(frames, speedFactor) {
    const result = [];
    for (let targetIndex = 0; targetIndex < frames.length; targetIndex++) {
        const sourcePosition = Math.min(targetIndex * speedFactor, frames.length - 1);
        const lowerIndex = Math.floor(sourcePosition);
        const upperIndex = Math.min(lowerIndex + 1, frames.length - 1);
        const amount = sourcePosition - lowerIndex;
        result.push(frames[lowerIndex].map((value, valueIndex) =>
            value * (1 - amount) + frames[upperIndex][valueIndex] * amount
        ));
    }
    return result;
}

function timeWarpVisualFrames(frames, speedFactor) {
    return frames.map((_, targetIndex) => {
        const sourceIndex = Math.min(Math.round(targetIndex * speedFactor), frames.length - 1);
        return frames[sourceIndex];
    });
}

function spatialJitter(frames, sigma = 0.005) {
    return frames.map(frame => {
        const jittered = [...frame];
        for (let pointIndex = 0; pointIndex < frame.length / 3; pointIndex++) {
            if (!pointIsPresent(frame, pointIndex)) continue;
            const offset = pointIndex * 3;
            for (let axis = 0; axis < 3; axis++) {
                const u1 = Math.max(Math.random(), Number.EPSILON);
                const u2 = Math.random();
                const noise = sigma * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
                jittered[offset + axis] += noise;
            }
        }
        return jittered;
    });
}

function swapPointData(frame, firstPoint, secondPoint) {
    for (let axis = 0; axis < 3; axis++) {
        const firstOffset = (firstPoint * 3) + axis;
        const secondOffset = (secondPoint * 3) + axis;
        const value = frame[firstOffset];
        frame[firstOffset] = frame[secondOffset];
        frame[secondOffset] = value;
    }
}

function mirrorFrames(frames) {
    const frameWidth = frames[0].length;
    return frames.map(frame => {
        const mirrored = [...frame];
        for (let valueIndex = 0; valueIndex < frameWidth; valueIndex += 3) {
            mirrored[valueIndex] = -mirrored[valueIndex];
        }

        if (frameWidth === KSL_RAW_FRAME_WIDTH) {
            [[0, 1], [2, 3], [4, 5], [6, 7]].forEach(([left, right]) =>
                swapPointData(mirrored, left, right)
            );
            for (let pointIndex = 0; pointIndex < HAND_NODE_COUNT; pointIndex++) {
                swapPointData(
                    mirrored,
                    LEFT_HAND_OFFSET + pointIndex,
                    RIGHT_HAND_OFFSET + pointIndex
                );
            }
        }
        return mirrored;
    });
}

function mirrorVisualFrames(frames) {
    return frames.map(frame => {
        const mirrored = new Array(VISUAL_FRAME_SIZE);
        for (let row = 0; row < VISUAL_HEIGHT; row++) {
            for (let column = 0; column < VISUAL_WIDTH; column++) {
                mirrored[(row * VISUAL_WIDTH) + column] =
                    frame[(row * VISUAL_WIDTH) + (VISUAL_WIDTH - column - 1)];
            }
        }
        return mirrored;
    });
}

function makeLabel(labelIndex, numClasses, smoothing = 0) {
    const offValue = smoothing / numClasses;
    const label = new Array(numClasses).fill(offValue);
    label[labelIndex] = 1 - smoothing + offValue;
    return label;
}

function createSample(positionFrames, visualFrames, labelIndex, numClasses, modelFamily, smoothing) {
    return {
        feature: modelFamily === 'asl' ? positionFrames : buildKslGraphFeatures(positionFrames),
        visual: visualFrames,
        label: makeLabel(labelIndex, numClasses, smoothing),
        labelIndex
    };
}

function augmentSample(positionFrames, visualFrames, labelIndex, numClasses, modelFamily) {
    const augmented = [];
    const add = (positions, visuals) => augmented.push(createSample(
        positions,
        visuals,
        labelIndex,
        numClasses,
        modelFamily,
        LABEL_SMOOTHING
    ));

    add(positionFrames, visualFrames);
    add(timeWarp(positionFrames, 0.85), visualFrames ? timeWarpVisualFrames(visualFrames, 0.85) : null);
    add(timeWarp(positionFrames, 1.15), visualFrames ? timeWarpVisualFrames(visualFrames, 1.15) : null);
    add(spatialJitter(positionFrames, 0.004), visualFrames);
    add(spatialJitter(positionFrames, 0.007), visualFrames);

    const mirroredPositions = mirrorFrames(positionFrames);
    const mirroredVisuals = visualFrames ? mirrorVisualFrames(visualFrames) : null;
    add(mirroredPositions, mirroredVisuals);
    add(spatialJitter(mirroredPositions, 0.004), mirroredVisuals);
    return augmented;
}

function balanceTrainingSamples(samples, numClasses) {
    const samplesByClass = Array.from({ length: numClasses }, () => []);
    samples.forEach(sample => samplesByClass[sample.labelIndex].push(sample));
    const targetCount = Math.max(...samplesByClass.map(group => group.length));
    const balanced = [];

    samplesByClass.forEach((group, classIndex) => {
        if (group.length === 0) {
            throw new Error(`No training samples are available for class index ${classIndex}.`);
        }
        balanced.push(...group);
        for (let count = group.length; count < targetCount; count++) {
            balanced.push(group[Math.floor(Math.random() * group.length)]);
        }
    });
    return balanced;
}

function shuffleInPlace(items) {
    for (let index = items.length - 1; index > 0; index--) {
        const swapIndex = Math.floor(Math.random() * (index + 1));
        [items[index], items[swapIndex]] = [items[swapIndex], items[index]];
    }
    return items;
}

function metadataString(recording, camelName, snakeName) {
    const value = recording[camelName] ?? recording[snakeName];
    return typeof value === 'string' && value.trim() ? value.trim() : null;
}

function normalizeLabel(value) {
    return value.normalize('NFKC').trim().replace(/\s+/g, ' ').toLocaleLowerCase();
}

function chooseValidationGrouping(rawData) {
    const signerIds = rawData.map(item => metadataString(item, 'signerId', 'signer_id'));
    if (signerIds.every(Boolean) && new Set(signerIds).size > 1) {
        return {
            mode: 'signer',
            key: item => metadataString(item, 'signerId', 'signer_id'),
            warning: null
        };
    }

    const sessionIds = rawData.map(item => metadataString(item, 'sessionId', 'session_id'));
    if (sessionIds.every(Boolean) && new Set(sessionIds).size > 1) {
        return {
            mode: 'session',
            key: item => {
                const signer = metadataString(item, 'signerId', 'signer_id') || 'unknown-signer';
                return `${signer}::${metadataString(item, 'sessionId', 'session_id')}`;
            },
            warning: 'Validation is session-independent, but not signer-independent.'
        };
    }

    return {
        mode: 'recording',
        key: item => metadataString(item, 'recordingId', 'recording_id') || item._sourceName,
        warning: 'WARNING: validation is split only by recording. It is not signer-independent and can still report misleadingly high accuracy.'
    };
}

function splitByRecording(rawData, uniqueLabels) {
    const trainRaw = [];
    const valRaw = [];
    uniqueLabels.forEach(label => {
        const recordings = shuffleInPlace(rawData.filter(item => item.label === label).slice());
        const validationCount = Math.min(
            recordings.length - 1,
            Math.max(1, Math.round(recordings.length * VALIDATION_FRACTION))
        );
        valRaw.push(...recordings.slice(0, validationCount));
        trainRaw.push(...recordings.slice(validationCount));
    });
    return { trainRaw, valRaw };
}

function scoreValidationGroups(rawData, uniqueLabels, grouping, selectedGroups) {
    let score = 0;
    for (const label of uniqueLabels) {
        const labelItems = rawData.filter(item => item.label === label);
        const validationCount = labelItems.filter(item => selectedGroups.has(grouping.key(item))).length;
        const trainingCount = labelItems.length - validationCount;
        if (validationCount === 0 || trainingCount === 0) score += 10000;
        score += Math.abs((validationCount / labelItems.length) - VALIDATION_FRACTION);
    }
    return score;
}

function splitByMetadataGroup(rawData, uniqueLabels, grouping) {
    const groups = [...new Set(rawData.map(grouping.key))];
    const validationGroupCount = Math.min(
        groups.length - 1,
        Math.max(1, Math.round(groups.length * VALIDATION_FRACTION))
    );
    let bestGroups = null;
    let bestScore = Infinity;

    for (let attempt = 0; attempt < 500; attempt++) {
        const candidate = new Set(shuffleInPlace(groups.slice()).slice(0, validationGroupCount));
        const score = scoreValidationGroups(rawData, uniqueLabels, grouping, candidate);
        if (score < bestScore) {
            bestScore = score;
            bestGroups = candidate;
        }
        if (score < 1) break;
    }

    if (!bestGroups || bestScore >= 10000) {
        throw new Error(
            `A ${grouping.mode}-independent validation split cannot include every class on both sides. ` +
            `Record every label with at least two ${grouping.mode}s.`
        );
    }

    return {
        trainRaw: rawData.filter(item => !bestGroups.has(grouping.key(item))),
        valRaw: rawData.filter(item => bestGroups.has(grouping.key(item)))
    };
}

function splitRawRecordings(rawData, uniqueLabels) {
    const grouping = chooseValidationGrouping(rawData);
    const split = grouping.mode === 'recording'
        ? splitByRecording(rawData, uniqueLabels)
        : splitByMetadataGroup(rawData, uniqueLabels, grouping);
    return { ...split, grouping };
}

function normalizeVisualSequence(visualFrames, recordingNumber) {
    return visualFrames.map((frame, frameIndex) => {
        let flattened = frame;
        if (Array.isArray(frame) && frame.length === VISUAL_HEIGHT && Array.isArray(frame[0])) {
            flattened = frame.flat();
        }
        if (!Array.isArray(flattened) || flattened.length !== VISUAL_FRAME_SIZE) {
            throw new Error(
                `Recording ${recordingNumber}, visual frame ${frameIndex + 1} must contain ${VISUAL_FRAME_SIZE} grayscale values.`
            );
        }
        return flattened.map(value => {
            if (!Number.isFinite(value)) {
                throw new Error(`Recording ${recordingNumber}, visual frame ${frameIndex + 1} contains a non-numeric value.`);
            }
            return Math.max(0, Math.min(1, value > 1 ? value / 255 : value));
        });
    });
}

function buildPositionFrames(recording, modelFamily) {
    let frames = sanitizeData(recording.frames);
    if (modelFamily === 'asl') return normalizeAslFrames(frames);

    // Legacy holistic clips were 30 frames. Resample before reducing them so old
    // raw recordings can train the new 60-frame KSL architecture.
    if (frames[0].length === LEGACY_HOLISTIC_FRAME_WIDTH && frames.length !== KSL_SEQUENCE_LENGTH) {
        frames = resampleFrames(frames, KSL_SEQUENCE_LENGTH);
    }
    return normalizeKslPositionFrames(extractKslPositionFrames(frames));
}

function createVisualTensor(samples) {
    const values = new Float32Array(samples.length * KSL_SEQUENCE_LENGTH * VISUAL_FRAME_SIZE);
    let offset = 0;
    samples.forEach(sample => sample.visual.forEach(frame => {
        values.set(frame, offset);
        offset += VISUAL_FRAME_SIZE;
    }));
    return tf.tensor5d(values, [
        samples.length,
        KSL_SEQUENCE_LENGTH,
        VISUAL_HEIGHT,
        VISUAL_WIDTH,
        VISUAL_CHANNELS
    ]);
}

function argMax(values) {
    let bestIndex = 0;
    for (let index = 1; index < values.length; index++) {
        if (values[index] > values[bestIndex]) bestIndex = index;
    }
    return bestIndex;
}

async function evaluateValidation(model, modelInputs, samples, uniqueLabels) {
    const predictionTensor = model.predict(modelInputs);
    const probabilities = await predictionTensor.array();
    predictionTensor.dispose();

    const matrix = Array.from(
        { length: uniqueLabels.length },
        () => new Array(uniqueLabels.length).fill(0)
    );
    let correct = 0;
    let confidenceTotal = 0;
    probabilities.forEach((row, sampleIndex) => {
        const predicted = argMax(row);
        const actual = samples[sampleIndex].labelIndex;
        matrix[actual][predicted]++;
        confidenceTotal += row[predicted];
        if (predicted === actual) correct++;
    });

    const perClass = uniqueLabels.map((label, classIndex) => {
        const truePositive = matrix[classIndex][classIndex];
        const actualCount = matrix[classIndex].reduce((sum, count) => sum + count, 0);
        const predictedCount = matrix.reduce((sum, row) => sum + row[classIndex], 0);
        const precision = predictedCount ? truePositive / predictedCount : 0;
        const recall = actualCount ? truePositive / actualCount : 0;
        const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
        return { label, precision, recall, f1, support: actualCount };
    });
    const macroF1 = perClass.reduce((sum, item) => sum + item.f1, 0) / perClass.length;
    const balancedAccuracy = perClass.reduce((sum, item) => sum + item.recall, 0) / perClass.length;
    const confusions = [];
    matrix.forEach((row, actualIndex) => row.forEach((count, predictedIndex) => {
        if (actualIndex !== predictedIndex && count > 0) {
            confusions.push({
                actual: uniqueLabels[actualIndex],
                predicted: uniqueLabels[predictedIndex],
                count
            });
        }
    }));
    confusions.sort((a, b) => b.count - a.count);

    return {
        accuracy: correct / samples.length,
        balancedAccuracy,
        macroF1,
        averageConfidence: confidenceTotal / samples.length,
        matrix,
        perClass,
        confusions
    };
}

function escapeHtml(value) {
    return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
}

function formatPercent(value) {
    return `${(value * 100).toFixed(1)}%`;
}

function renderEvaluationReport(evaluation, uniqueLabels, grouping, usesHandCrops) {
    const report = document.getElementById('evaluationReport');
    if (!report) return;
    const confusionSummary = evaluation.confusions.length
        ? evaluation.confusions.slice(0, 8).map(item =>
            `<li><strong>${escapeHtml(item.actual)}</strong> predicted as <strong>${escapeHtml(item.predicted)}</strong>: ${item.count}</li>`
        ).join('')
        : '<li>No mistakes in this validation split. Test with an unseen signer before trusting this result.</li>';
    const rows = evaluation.perClass.map(item => `
        <tr>
            <td>${escapeHtml(item.label)}</td>
            <td>${item.support}</td>
            <td>${formatPercent(item.precision)}</td>
            <td>${formatPercent(item.recall)}</td>
            <td>${formatPercent(item.f1)}</td>
        </tr>
    `).join('');
    const matrixHeader = uniqueLabels.map(label => `<th>${escapeHtml(label)}</th>`).join('');
    const matrixRows = evaluation.matrix.map((matrixRow, actualIndex) => `
        <tr>
            <th>${escapeHtml(uniqueLabels[actualIndex])}</th>
            ${matrixRow.map((count, predictedIndex) =>
                `<td class="${actualIndex === predictedIndex ? 'matrix-hit' : (count ? 'matrix-miss' : '')}">${count}</td>`
            ).join('')}
        </tr>
    `).join('');
    const warning = grouping.warning
        ? `<p class="trainer-warning">${escapeHtml(grouping.warning)}</p>`
        : '<p class="trainer-success">Validation signers were fully held out from training.</p>';

    report.hidden = false;
    report.innerHTML = `
        <h2>Raw validation report</h2>
        <div class="metric-grid">
            <div><span>Accuracy</span><strong>${formatPercent(evaluation.accuracy)}</strong></div>
            <div><span>Balanced accuracy</span><strong>${formatPercent(evaluation.balancedAccuracy)}</strong></div>
            <div><span>Macro F1</span><strong>${formatPercent(evaluation.macroF1)}</strong></div>
            <div><span>Average confidence</span><strong>${formatPercent(evaluation.averageConfidence)}</strong></div>
        </div>
        <p class="trainer-note">Split by ${escapeHtml(grouping.mode)}. Hand-crop stream: ${usesHandCrops ? 'enabled' : 'disabled'}.</p>
        ${warning}
        <h3>Most common confusions</h3>
        <ul class="confusion-list">${confusionSummary}</ul>
        <div class="table-scroll">
            <table class="metrics-table">
                <thead><tr><th>Class</th><th>Raw clips</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
                <tbody>${rows}</tbody>
            </table>
        </div>
        <details class="matrix-details">
            <summary>Full confusion matrix (rows are actual, columns are predicted)</summary>
            <div class="table-scroll">
                <table class="metrics-table confusion-matrix">
                    <thead><tr><th>Actual / predicted</th>${matrixHeader}</tr></thead>
                    <tbody>${matrixRows}</tbody>
                </table>
            </div>
        </details>
    `;
}

function triggerJsonDownload(filename, data) {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
}

// ==========================================
// TRAINING PIPELINE
// ==========================================

async function startTraining() {
    const fileInput = document.getElementById('fileInput');
    const statusText = document.getElementById('status');
    const report = document.getElementById('evaluationReport');
    const modelFamily = document.getElementById('modelFamily')?.value || 'csl';
    const isAsl = modelFamily === 'asl';
    const sequenceLength = getSequenceLength(modelFamily);
    const subModelName = document.getElementById('subModelName')?.value.trim() || '';
    const usingSubModel = !isAsl && subModelName.length > 0;

    if (report) report.hidden = true;
    if (!fileInput.files.length) {
        alert('Please upload your JSON files first.');
        return;
    }

    statusText.textContent = 'Step 1: validating raw recordings and metadata...';
    await new Promise(resolve => setTimeout(resolve, 50));

    const rawData = [];
    let uniqueLabels;
    let sourceFrameWidth = null;
    let usesHandCrops = false;
    let visualFallbackMessage = '';

    try {
        for (const file of fileInput.files) {
            const recording = JSON.parse(await file.text());
            recording._sourceName = file.name;
            rawData.push(recording);
        }

        rawData.forEach((item, recordingIndex) => {
            const number = recordingIndex + 1;
            if (typeof item.label !== 'string' || !item.label.trim()) {
                throw new Error(`Recording ${number} is missing a valid label.`);
            }
            item.label = normalizeLabel(item.label);
            if (!Array.isArray(item.frames) || item.frames.length === 0) {
                throw new Error(`Recording ${number} has no landmark frames.`);
            }

            const width = item.frames[0]?.length;
            const validWidths = isAsl
                ? [ASL_FRAME_WIDTH]
                : [KSL_RAW_FRAME_WIDTH, LEGACY_HOLISTIC_FRAME_WIDTH];
            if (!validWidths.includes(width)) {
                throw new Error(
                    `${isAsl ? 'ASL' : 'KSL'} recording ${number} has ${width || 0} values per frame; expected ${validWidths.join(' or ')}.`
                );
            }
            if (sourceFrameWidth === null) sourceFrameWidth = width;
            if (width !== sourceFrameWidth) {
                throw new Error('Do not mix 150-value and legacy 225-value KSL recordings in one training run.');
            }

            const allowedFrameCounts = !isAsl && width === LEGACY_HOLISTIC_FRAME_WIDTH
                ? [ASL_SEQUENCE_LENGTH, KSL_SEQUENCE_LENGTH]
                : [sequenceLength];
            if (!allowedFrameCounts.includes(item.frames.length)) {
                throw new Error(
                    `Recording ${number} has ${item.frames.length} frames; expected ${allowedFrameCounts.join(' or ')}.`
                );
            }
            item.frames.forEach((frame, frameIndex) => {
                if (!Array.isArray(frame) || frame.length !== width || frame.some(value => !Number.isFinite(value))) {
                    throw new Error(`Recording ${number}, frame ${frameIndex + 1} has an invalid landmark layout.`);
                }
            });

            if (!isAsl && width === KSL_RAW_FRAME_WIDTH) {
                if (item.schemaVersion !== KSL_SCHEMA_VERSION) {
                    throw new Error(`Recording ${number} must declare schemaVersion "${KSL_SCHEMA_VERSION}".`);
                }
                if (item.frameCount !== KSL_SEQUENCE_LENGTH || item.rawFrameWidth !== KSL_RAW_FRAME_WIDTH) {
                    throw new Error(
                        `Recording ${number} metadata must declare frameCount ${KSL_SEQUENCE_LENGTH} and rawFrameWidth ${KSL_RAW_FRAME_WIDTH}.`
                    );
                }
            }
        });

        uniqueLabels = [...new Set(rawData.map(item => item.label))].sort();
        if (uniqueLabels.length < 2) {
            throw new Error('Training requires recordings from at least two classes.');
        }

        const rawCountsByLabel = Object.fromEntries(uniqueLabels.map(label => [label, 0]));
        rawData.forEach(item => rawCountsByLabel[item.label]++);
        const undersupplied = Object.entries(rawCountsByLabel)
            .filter(([, count]) => count < MIN_RAW_RECORDINGS_PER_CLASS);
        if (undersupplied.length) {
            throw new Error(
                `At least ${MIN_RAW_RECORDINGS_PER_CLASS} raw recordings per class are required. Missing: ` +
                undersupplied.map(([label, count]) => `${label} (${count})`).join(', ')
            );
        }

        if (!isAsl) {
            const recordingsWithCrops = rawData.filter(item => Array.isArray(item.visualFrames));
            usesHandCrops = recordingsWithCrops.length === rawData.length;
            if (usesHandCrops) {
                rawData.forEach((item, recordingIndex) => {
                    if (item.visualFrames.length !== KSL_SEQUENCE_LENGTH) {
                        throw new Error(`Recording ${recordingIndex + 1} must contain ${KSL_SEQUENCE_LENGTH} visualFrames.`);
                    }
                    if (item.visualShape && JSON.stringify(item.visualShape) !== JSON.stringify([VISUAL_HEIGHT, VISUAL_WIDTH, VISUAL_CHANNELS])) {
                        throw new Error(`Recording ${recordingIndex + 1} visualShape must be [${VISUAL_HEIGHT}, ${VISUAL_WIDTH}, ${VISUAL_CHANNELS}].`);
                    }
                    item._normalizedVisualFrames = normalizeVisualSequence(item.visualFrames, recordingIndex + 1);
                });
            } else if (recordingsWithCrops.length > 0) {
                visualFallbackMessage = `${recordingsWithCrops.length}/${rawData.length} recordings contain hand crops, so this run is landmark-only. Every recording must contain crops to enable fusion.`;
                console.warn(visualFallbackMessage);
            } else {
                visualFallbackMessage = 'No visualFrames found; this run is landmark-only.';
            }
        }
    } catch (error) {
        console.error('Unable to prepare training data:', error);
        statusText.textContent = `Training stopped: ${error.message}`;
        alert(error.message);
        return;
    }

    let trainRaw;
    let valRaw;
    let grouping;
    try {
        ({ trainRaw, valRaw, grouping } = splitRawRecordings(rawData, uniqueLabels));
    } catch (error) {
        statusText.textContent = `Training stopped: ${error.message}`;
        alert(error.message);
        return;
    }
    if (!isAsl && grouping.mode === 'recording') {
        const message =
            'KSL training requires at least two signer IDs or two session IDs, with every label represented in both groups. ' +
            'A recording-only split would recreate the misleading 100% validation result this pipeline is designed to prevent.';
        statusText.textContent = `Training stopped: ${message}`;
        alert(message);
        return;
    }
    if (grouping.warning) console.warn(grouping.warning);

    statusText.textContent =
        `Split by ${grouping.mode}: ${trainRaw.length} train / ${valRaw.length} raw validation clips. ` +
        (usesHandCrops ? 'Two-hand crop fusion enabled.' : visualFallbackMessage);
    await new Promise(resolve => setTimeout(resolve, 50));

    function buildSamples(items, augment) {
        const samples = [];
        items.forEach(item => {
            const positions = buildPositionFrames(item, modelFamily);
            const visualFrames = usesHandCrops ? item._normalizedVisualFrames : null;
            const labelIndex = uniqueLabels.indexOf(item.label);
            if (augment) {
                samples.push(...augmentSample(
                    positions,
                    visualFrames,
                    labelIndex,
                    uniqueLabels.length,
                    modelFamily
                ));
            } else {
                // Validation is one untouched feature sequence per raw recording.
                samples.push(createSample(
                    positions,
                    visualFrames,
                    labelIndex,
                    uniqueLabels.length,
                    modelFamily,
                    0
                ));
            }
        });
        return samples;
    }

    const unbalancedTrainSamples = buildSamples(trainRaw, true);
    const trainSamples = balanceTrainingSamples(unbalancedTrainSamples, uniqueLabels.length);
    const valSamples = buildSamples(valRaw, false);
    shuffleInPlace(trainSamples);

    console.log(
        `Training: ${trainSamples.length} augmented/balanced samples from ${trainRaw.length} raw clips. ` +
        `Validation: ${valSamples.length} unaugmented raw clips.`
    );
    const xsTrain = tf.tensor3d(trainSamples.map(sample => sample.feature));
    const ysTrain = tf.tensor2d(trainSamples.map(sample => sample.label));
    const xsVal = tf.tensor3d(valSamples.map(sample => sample.feature));
    const ysVal = tf.tensor2d(valSamples.map(sample => sample.label));
    const visualTrain = usesHandCrops ? createVisualTensor(trainSamples) : null;
    const visualVal = usesHandCrops ? createVisualTensor(valSamples) : null;
    const trainInputs = usesHandCrops ? [xsTrain, visualTrain] : xsTrain;
    const valInputs = usesHandCrops ? [xsVal, visualVal] : xsVal;

    statusText.textContent = isAsl
        ? `Step 2: building legacy ASL TCN (${ASL_SEQUENCE_LENGTH} x ${ASL_FRAME_WIDTH})...`
        : `Step 2: building graph-TCN (${KSL_SEQUENCE_LENGTH} x ${KSL_MODEL_FRAME_WIDTH})${usesHandCrops ? ' with hand-crop fusion' : ''}...`;
    await new Promise(resolve => setTimeout(resolve, 50));

    let model;
    try {
        model = isAsl
            ? buildAslTcnClassifier(uniqueLabels.length)
            : buildKslGraphTcnClassifier(uniqueLabels.length, usesHandCrops);
    } catch (error) {
        [xsTrain, ysTrain, xsVal, ysVal, visualTrain, visualVal]
            .filter(Boolean)
            .forEach(tensor => tensor.dispose());
        statusText.textContent = `Training stopped: ${error.message}`;
        alert(error.message);
        return;
    }

    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: isAsl ? 'categoricalCrossentropy' : focalCategoricalCrossentropy,
        metrics: ['accuracy']
    });
    model.summary();

    statusText.textContent = `Step 3: training on ${trainRaw.length} raw clips; validating on ${valRaw.length} untouched clips...`;
    await new Promise(resolve => setTimeout(resolve, 50));

    let bestValAcc = 0;
    let bestValLoss = Infinity;
    let bestWeights = null;
    let epochsSinceImprovement = 0;
    const patience = 12;

    try {
        await model.fit(trainInputs, ysTrain, {
            epochs: 80,
            batchSize: usesHandCrops ? 8 : 16,
            validationData: [valInputs, ysVal],
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    const accuracy = logs.acc ?? logs.accuracy ?? 0;
                    const validationAccuracy = logs.val_acc ?? logs.val_accuracy ?? 0;
                    bestValAcc = Math.max(bestValAcc, validationAccuracy);

                    if (logs.val_loss < bestValLoss - 1e-4) {
                        bestValLoss = logs.val_loss;
                        epochsSinceImprovement = 0;
                        if (bestWeights) bestWeights.forEach(tensor => tensor.dispose());
                        bestWeights = model.getWeights().map(tensor => tensor.clone());
                    } else {
                        epochsSinceImprovement++;
                    }
                    if (epochsSinceImprovement >= patience) model.stopTraining = true;

                    console.log(
                        `Epoch ${epoch + 1}/80 | loss ${logs.loss.toFixed(4)} | ` +
                        `acc ${formatPercent(accuracy)} | val loss ${logs.val_loss.toFixed(4)} | ` +
                        `val acc ${formatPercent(validationAccuracy)}`
                    );
                    statusText.textContent = epochsSinceImprovement >= patience
                        ? 'Early stopping: restoring the lowest-validation-loss checkpoint...'
                        : `Epoch ${epoch + 1}/80 | train ${formatPercent(accuracy)} | raw validation ${formatPercent(validationAccuracy)} | best ${formatPercent(bestValAcc)}`;
                    await new Promise(resolve => setTimeout(resolve, 0));
                }
            }
        });

        if (bestWeights) {
            model.setWeights(bestWeights);
            bestWeights.forEach(tensor => tensor.dispose());
            bestWeights = null;
        }

        statusText.textContent = 'Evaluating per-class precision, recall, F1, and confusions...';
        await new Promise(resolve => setTimeout(resolve, 50));
        const evaluation = await evaluateValidation(model, valInputs, valSamples, uniqueLabels);
        renderEvaluationReport(evaluation, uniqueLabels, grouping, usesHandCrops);

        const modelExportName = isAsl
            ? 'asl-model'
            : (usingSubModel ? `${subModelName}_model` : 'csl_model');
        const labelsExportName = isAsl
            ? 'asl-labels.json'
            : (usingSubModel ? `${subModelName}_labels.json` : 'csl_labels.json');
        const metadataExportName = `${modelExportName}_metadata.json`;
        const metadata = {
            schemaVersion: isAsl ? 'asl-legacy-v1' : KSL_SCHEMA_VERSION,
            rawFrameWidth: isAsl ? ASL_FRAME_WIDTH : KSL_RAW_FRAME_WIDTH,
            modelFrameWidth: isAsl ? ASL_FRAME_WIDTH : KSL_MODEL_FRAME_WIDTH,
            frameCount: sequenceLength,
            architecture: isAsl
                ? 'legacy-tcn'
                : (usesHandCrops ? 'graph-edge-tcn-late-hand-crop-fusion' : 'graph-edge-tcn'),
            usesHandCrops,
            visualShape: usesHandCrops ? [VISUAL_HEIGHT, VISUAL_WIDTH, VISUAL_CHANNELS] : null,
            validationGroup: grouping.mode,
            labels: uniqueLabels,
            acceptedSourceFrameWidths: isAsl
                ? [ASL_FRAME_WIDTH]
                : [KSL_RAW_FRAME_WIDTH, LEGACY_HOLISTIC_FRAME_WIDTH],
            trainingSourceFrameWidth: sourceFrameWidth,
            layoutId: isAsl ? 'hand21_xyz' : 'upper8_hands2x21_xyz',
            normalization: isAsl
                ? 'wrist-relative'
                : 'shoulder-midpoint-relative_shoulder-width-scaled',
            kslNodeOrder: isAsl ? null : {
                pose: KSL_BODY_POSE_INDICES,
                leftHand: '0..20',
                rightHand: '0..20'
            },
            featureChannels: isAsl
                ? ['x', 'y', 'z']
                : ['node_xyz', 'node_velocity_xyz', 'node_acceleration_xyz', 'graph_edge_dx_dy_dz_distance', 'left_right_hand_presence'],
            trainingLoss: isAsl
                ? 'categorical-crossentropy-with-label-smoothing-0.04'
                : 'focal-categorical-crossentropy-gamma-2-with-label-smoothing-0.04',
            trainedAt: new Date().toISOString()
        };

        statusText.textContent = `Step 4: complete. Raw validation accuracy ${formatPercent(evaluation.accuracy)}, macro F1 ${formatPercent(evaluation.macroF1)}. Downloading files...`;
        await model.save(`downloads://${modelExportName}`);
        await new Promise(resolve => setTimeout(resolve, 800));
        triggerJsonDownload(labelsExportName, uniqueLabels);
        await new Promise(resolve => setTimeout(resolve, 800));
        triggerJsonDownload(metadataExportName, metadata);

        statusText.textContent =
            `Done: ${uniqueLabels.length} classes, raw validation ${formatPercent(evaluation.accuracy)}, ` +
            `macro F1 ${formatPercent(evaluation.macroF1)}. ` +
            (grouping.warning || 'Validation used unseen signers.') +
            (usesHandCrops ? ' Two-hand crop fusion is enabled.' : ` ${visualFallbackMessage}`);
    } catch (error) {
        console.error('Training failed:', error);
        if (bestWeights) bestWeights.forEach(tensor => tensor.dispose());
        statusText.textContent = `Training failed: ${error.message}`;
        alert(error.message);
    } finally {
        [xsTrain, ysTrain, xsVal, ysVal, visualTrain, visualVal]
            .filter(Boolean)
            .forEach(tensor => tensor.dispose());
    }
}
