export const WLASL_DEFAULTS = Object.freeze({
  frameCount: 64,
  imageSize: 224,
  channels: 3,
  normalization: "minus_one_to_one",
  inputLayout: "NTHWC",
  clipDurationMs: 2500,
  confidenceThreshold: 0.5
});

export function validateWlaslManifest(manifest) {
  if (!manifest || manifest.enabled !== true) {
    throw new Error("The WLASL bundle is not enabled.");
  }
  if (!manifest.modelPath || !manifest.labelsPath) {
    throw new Error("WLASL manifest requires modelPath and labelsPath.");
  }
  const config = { ...WLASL_DEFAULTS, ...manifest };
  if (!Number.isInteger(config.frameCount) || config.frameCount < 2) {
    throw new Error("WLASL frameCount must be an integer of at least 2.");
  }
  if (!Number.isInteger(config.imageSize) || config.imageSize < 16) {
    throw new Error("WLASL imageSize must be an integer of at least 16.");
  }
  if (config.inputLayout !== "NTHWC") {
    throw new Error("Browser WLASL inference currently supports NTHWC models only.");
  }
  if (!["zero_to_one", "minus_one_to_one"].includes(config.normalization)) {
    throw new Error(`Unsupported WLASL normalization: ${config.normalization}.`);
  }
  return config;
}

export function uniformSampleIndices(sourceLength, targetLength) {
  if (!Number.isInteger(sourceLength) || sourceLength < 1) {
    throw new Error("At least one source video frame is required.");
  }
  if (!Number.isInteger(targetLength) || targetLength < 2) {
    throw new Error("targetLength must be an integer of at least 2.");
  }
  if (sourceLength === 1) return new Array(targetLength).fill(0);
  return Array.from({ length: targetLength }, (_, index) =>
    Math.round(index * (sourceLength - 1) / (targetLength - 1))
  );
}

export function normalizeRgbBytes(bytes, normalization = WLASL_DEFAULTS.normalization) {
  const scale = normalization === "zero_to_one" ? 1 / 255 : 2 / 255;
  const offset = normalization === "zero_to_one" ? 0 : -1;
  return Float32Array.from(bytes, value => value * scale + offset);
}
