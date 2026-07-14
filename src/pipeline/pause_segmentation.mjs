export const PAUSE_SEGMENTATION_DEFAULTS = Object.freeze({
  minPauseMs: 400,
  minWordMs: 300,
  paddingMs: 120
});

/**
 * Split one recorded landmark stream into isolated word ranges. A word starts
 * when either hand is visible and ends only after a sustained neutral pause.
 * Short tracking dropouts remain inside the current word.
 */
export function segmentPausedWords(frames, timestamps, options = {}) {
  if (!Array.isArray(frames) || !Array.isArray(timestamps) || frames.length !== timestamps.length) {
    throw new Error("frames and timestamps must be arrays with equal lengths.");
  }
  if (frames.length < 2) return [];

  const config = { ...PAUSE_SEGMENTATION_DEFAULTS, ...options };
  const isActive = options.isActive || holisticHandsPresent;
  const segments = [];
  let wordStart = null;
  let lastActive = null;
  let pauseStart = null;

  const closeWord = () => {
    if (wordStart === null || lastActive === null) return;
    const activeDurationMs = timestamps[lastActive] - timestamps[wordStart];
    if (activeDurationMs >= config.minWordMs) {
      segments.push(buildPaddedRange(
        wordStart,
        lastActive,
        timestamps,
        config.paddingMs
      ));
    }
    wordStart = null;
    lastActive = null;
    pauseStart = null;
  };

  for (let index = 0; index < frames.length; index++) {
    const active = Boolean(isActive(frames[index], index));
    if (active) {
      if (wordStart === null) wordStart = index;
      lastActive = index;
      pauseStart = null;
      continue;
    }

    if (wordStart === null) continue;
    if (pauseStart === null) pauseStart = index;
    if (timestamps[index] - timestamps[pauseStart] >= config.minPauseMs) closeWord();
  }

  closeWord();
  return segments.map((segment, index) => ({ ...segment, index }));
}

/** The runtime holistic layout is pose33 + left hand21 + right hand21, xyz. */
export function holisticHandsPresent(frame) {
  if (!Array.isArray(frame) || frame.length < 225) return false;
  return rangeHasLandmark(frame, 99, 162) || rangeHasLandmark(frame, 162, 225);
}

function rangeHasLandmark(frame, start, end) {
  for (let offset = start; offset < end; offset += 3) {
    if (frame[offset] !== 0 || frame[offset + 1] !== 0 || frame[offset + 2] !== 0) return true;
  }
  return false;
}

function buildPaddedRange(startIndex, lastActiveIndex, timestamps, paddingMs) {
  const clipStart = timestamps[0];
  const activeStartMs = timestamps[startIndex] - clipStart;
  const activeEndMs = timestamps[lastActiveIndex] - clipStart;
  let frameStart = startIndex;
  let frameEnd = lastActiveIndex + 1;

  while (frameStart > 0 && timestamps[startIndex] - timestamps[frameStart - 1] <= paddingMs) frameStart--;
  while (
    frameEnd < timestamps.length
    && timestamps[frameEnd] - timestamps[lastActiveIndex] <= paddingMs
  ) frameEnd++;

  return {
    frameStart,
    frameEnd,
    startMs: Math.max(0, timestamps[frameStart] - clipStart),
    endMs: Math.max(0, timestamps[frameEnd - 1] - clipStart),
    activeStartMs,
    activeEndMs
  };
}
