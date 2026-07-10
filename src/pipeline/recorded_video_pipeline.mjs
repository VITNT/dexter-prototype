import { prepareSequenceForModel } from "./sequence_preprocessing.mjs";
import { KSL_SCHEMA_VERSION, prepareKslSequence } from "./ksl_feature_pipeline.mjs";
import { translateLabel } from "./model_registry.mjs";

export async function classifyRecordedSign({
  video,
  trackers,
  tf,
  model,
  labels,
  catalog,
  languageId,
  targetLanguage,
  modelConfig,
  fps = modelConfig?.fps || 15,
  maxSeconds = 10
}) {
  if (!video) throw new Error("A decoded HTMLVideoElement is required.");
  if (!trackers?.extractFrame) throw new Error("trackers.extractFrame(video, timestampMs) is required.");
  if (!tf?.tensor3d) throw new Error("TensorFlow.js must be provided as { tf }.");
  if (!model?.predict) throw new Error("A TensorFlow.js model with predict() is required.");

  await waitForVideoMetadata(video);
  const rawFrames = await extractSequenceFromVideo(video, trackers.extractFrame, { fps, maxSeconds });
  const preparedFrames = modelConfig?.schemaVersion === KSL_SCHEMA_VERSION
    ? prepareKslSequence(rawFrames, {
        targetFrameCount: modelConfig?.frameCount || 60
      })
    : prepareSequenceForModel(rawFrames, {
        targetFrameCount: modelConfig?.frameCount || 30,
        expectedFrameWidth: modelConfig?.frameWidth || 225
      });

  if ((model.inputs?.length || 1) > 1) {
    throw new Error(
      "This model also requires two-hand crop frames. Use the live predictor or provide a crop-aware recorded-video extractor."
    );
  }

  const prediction = predictTopK({ tf, model, labels, frames: preparedFrames, topK: 5 });
  const best = prediction[0];
  const translation = catalog
    ? translateLabel(catalog, languageId, best.label, targetLanguage)
    : { text: best.label, source: "label_fallback" };

  return {
    best,
    topK: prediction,
    translation,
    frames: preparedFrames
  };
}

export async function extractSequenceFromVideo(video, extractFrame, options = {}) {
  const fps = options.fps || 15;
  const maxSeconds = options.maxSeconds || 10;
  const duration = Number.isFinite(video.duration) ? Math.min(video.duration, maxSeconds) : maxSeconds;
  const frameCount = Math.max(2, Math.ceil(duration * fps));
  const frames = [];

  for (let frameIndex = 0; frameIndex < frameCount; frameIndex++) {
    const timeSeconds = frameIndex / fps;
    await seekVideo(video, Math.min(timeSeconds, duration));
    const timestampMs = Math.round(video.currentTime * 1000);
    frames.push(await extractFrame(video, timestampMs));
  }

  return frames;
}

export function predictTopK({ tf, model, labels, frames, topK = 5 }) {
  const input = tf.tensor3d([frames]);
  const output = model.predict(input);

  try {
    const probabilities = Array.from(output.dataSync());
    return probabilities
      .map((confidence, index) => ({
        label: labels[index] ?? `class_${index}`,
        confidence
      }))
      .sort((left, right) => right.confidence - left.confidence)
      .slice(0, topK);
  } finally {
    input.dispose();
    if (output?.dispose) output.dispose();
  }
}

export function createObjectUrlVideo(file) {
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.preload = "auto";
  video.src = URL.createObjectURL(file);
  return video;
}

export function waitForVideoMetadata(video) {
  if (video.readyState >= 1) return Promise.resolve();

  return new Promise((resolve, reject) => {
    const cleanup = () => {
      video.removeEventListener("loadedmetadata", onLoaded);
      video.removeEventListener("error", onError);
    };
    const onLoaded = () => {
      cleanup();
      resolve();
    };
    const onError = () => {
      cleanup();
      reject(new Error("Failed to load recorded video metadata."));
    };

    video.addEventListener("loadedmetadata", onLoaded, { once: true });
    video.addEventListener("error", onError, { once: true });
  });
}

function seekVideo(video, timeSeconds) {
  if (Math.abs(video.currentTime - timeSeconds) < 0.001 && video.readyState >= 2) {
    return Promise.resolve();
  }

  return new Promise((resolve, reject) => {
    const cleanup = () => {
      video.removeEventListener("seeked", onSeeked);
      video.removeEventListener("error", onError);
    };
    const onSeeked = () => {
      cleanup();
      resolve();
    };
    const onError = () => {
      cleanup();
      reject(new Error("Failed while seeking recorded video."));
    };

    video.addEventListener("seeked", onSeeked, { once: true });
    video.addEventListener("error", onError, { once: true });
    video.currentTime = timeSeconds;
  });
}
