export function selectShiftedWindows(frames, windowFrameCount, offsetsFromEnd = [0]) {
  if (!Array.isArray(frames) || !Number.isInteger(windowFrameCount) || windowFrameCount < 1) return [];
  const starts = new Set();
  const windows = [];
  for (const rawOffset of offsetsFromEnd) {
    const offset = Math.max(0, Math.trunc(Number(rawOffset) || 0));
    const start = frames.length - windowFrameCount - offset;
    if (start < 0 || starts.has(start)) continue;
    starts.add(start);
    windows.push(frames.slice(start, start + windowFrameCount));
  }
  return windows;
}

export function averageProbabilityVectors(vectors) {
  if (!Array.isArray(vectors) || vectors.length === 0) return [];
  const width = vectors[0]?.length || 0;
  if (!width || vectors.some(vector => !vector || vector.length !== width)) {
    throw new Error("Probability vectors must have one shared non-zero width.");
  }
  const average = new Array(width).fill(0);
  for (const vector of vectors) {
    for (let index = 0; index < width; index++) {
      const value = Number(vector[index]);
      average[index] += Number.isFinite(value) ? value : 0;
    }
  }
  return average.map(value => value / vectors.length);
}

export function normalizeCslLabel(value) {
  return String(value || "")
    .normalize("NFKC")
    .trim()
    .toLocaleLowerCase()
    .replace(/[?!.។៕၊၊]+$/u, "")
    .replace(/\s+/g, " ");
}

export function findStaticAgreement(temporalLabel, staticPredictions, minimumConfidence = 0.75) {
  const target = normalizeCslLabel(temporalLabel);
  if (!target || !Array.isArray(staticPredictions)) return null;
  return staticPredictions.find(prediction => (
    Number(prediction?.confidence) >= minimumConfidence
    && normalizeCslLabel(prediction?.word) === target
  )) || null;
}

export function resolveCslRuntimeConfig(metadata = {}) {
  const windowFrameCount = metadata.windowFrameCount || metadata.frameCount || 30;
  return {
    captureFrameCount: metadata.captureFrameCount || windowFrameCount,
    windowFrameCount,
    windowOffsetsFromEnd: metadata.windowOffsetsFromEnd || [0],
    staticAgreement: { enabled: false, ...(metadata.staticAgreement || {}) }
  };
}
