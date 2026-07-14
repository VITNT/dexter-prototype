import assert from "node:assert/strict";
import { holisticHandsPresent, segmentPausedWords } from "./pause_segmentation.mjs";

const frame = active => {
  const values = new Array(225).fill(0);
  if (active) values[99] = 0.5;
  return values;
};

assert.equal(holisticHandsPresent(frame(false)), false);
assert.equal(holisticHandsPresent(frame(true)), true);

const timestamps = Array.from({ length: 70 }, (_, index) => index * 100);
const frames = timestamps.map((_, index) => frame(
  (index >= 5 && index <= 15)
  || (index >= 18 && index <= 25) // 200 ms dropout: same word
  || (index >= 35 && index <= 50)
));

const segments = segmentPausedWords(frames, timestamps, {
  minPauseMs: 400,
  minWordMs: 300,
  paddingMs: 100
});

assert.equal(segments.length, 2);
assert.deepEqual(
  segments.map(({ activeStartMs, activeEndMs }) => [activeStartMs, activeEndMs]),
  [[500, 2500], [3500, 5000]]
);
assert.ok(segments[0].endMs < segments[1].startMs);

const noise = timestamps.map((_, index) => frame(index >= 2 && index <= 3));
assert.deepEqual(segmentPausedWords(noise, timestamps), []);

console.log("Pause segmentation tests passed: tracking gaps merge and neutral pauses split words.");
