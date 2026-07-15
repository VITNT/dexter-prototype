const HAND_LANDMARK_COUNT = 21;
const FEATURES_PER_HAND = HAND_LANDMARK_COUNT * 2;
const MODEL_INPUT_WIDTH = FEATURES_PER_HAND * 2;

function finiteNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

function normalizeAxis(values) {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + ((value - mean) ** 2), 0) / values.length;
  const standardDeviation = Math.sqrt(variance) + 1e-6;
  return values.map((value) => (value - mean) / standardDeviation);
}

export function normalizeStaticHand(hand) {
  if (!Array.isArray(hand) || hand.length < HAND_LANDMARK_COUNT) {
    return new Array(FEATURES_PER_HAND).fill(0);
  }

  const points = hand.slice(0, HAND_LANDMARK_COUNT);
  const xs = normalizeAxis(points.map((point) => finiteNumber(point?.x)));
  const ys = normalizeAxis(points.map((point) => finiteNumber(point?.y)));
  return [...xs, ...ys];
}

// Preserve MediaPipe detection order to match the released upstream model.
export function buildStaticKslFeatures(hands) {
  const detectedHands = Array.isArray(hands) ? hands.slice(0, 2) : [];
  const features = detectedHands.flatMap(normalizeStaticHand);
  while (features.length < MODEL_INPUT_WIDTH) features.push(0);
  return features.slice(0, MODEL_INPUT_WIDTH);
}

function dense(input, weight, bias, useRelu) {
  return weight.map((row, rowIndex) => {
    let value = finiteNumber(bias[rowIndex]);
    for (let column = 0; column < input.length; column++) {
      value += input[column] * finiteNumber(row[column]);
    }
    return useRelu ? Math.max(0, value) : value;
  });
}

function softmax(logits) {
  const maximum = Math.max(...logits);
  const exponentials = logits.map((value) => Math.exp(value - maximum));
  const denominator = exponentials.reduce((sum, value) => sum + value, 0) || 1;
  return exponentials.map((value) => value / denominator);
}

export function validateStaticKslModel(model) {
  const classCount = model?.classes?.length;
  const layers = model?.layers;
  const valid = Number.isInteger(classCount)
    && classCount > 0
    && Array.isArray(model?.displayLabels)
    && model.displayLabels.length === classCount
    && Array.isArray(layers)
    && layers.length === 3
    && layers[0]?.weight?.length === 128
    && layers[0]?.weight?.[0]?.length === MODEL_INPUT_WIDTH
    && layers[0]?.bias?.length === 128
    && layers[1]?.weight?.length === 64
    && layers[1]?.weight?.[0]?.length === 128
    && layers[1]?.bias?.length === 64
    && layers[2]?.weight?.length === classCount
    && layers[2]?.weight?.[0]?.length === 64
    && layers[2]?.bias?.length === classCount;

  if (!valid) throw new Error("Invalid static KSL LandmarkNet model asset.");
  return model;
}

export function predictStaticKsl(model, hands) {
  validateStaticKslModel(model);
  const features = buildStaticKslFeatures(hands);
  const hidden1 = dense(features, model.layers[0].weight, model.layers[0].bias, true);
  const hidden2 = dense(hidden1, model.layers[1].weight, model.layers[1].bias, true);
  const probabilities = softmax(dense(hidden2, model.layers[2].weight, model.layers[2].bias, false));

  return probabilities
    .map((confidence, index) => ({
      form: model.classes[index],
      word: model.displayLabels[index] || model.classes[index],
      confidence
    }))
    .sort((a, b) => b.confidence - a.confidence);
}

export const STATIC_KSL_INPUT_WIDTH = MODEL_INPUT_WIDTH;
