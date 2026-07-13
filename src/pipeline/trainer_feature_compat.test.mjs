import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import { prepareKslSequence, KSL_FEATURE_SCHEMA } from "./ksl_feature_pipeline.mjs";

const trainerSource = fs.readFileSync(new URL("../../trainer.js", import.meta.url), "utf8");
const context = vm.createContext({ console });
new vm.Script(trainerSource, { filename: "trainer.js" }).runInContext(context);

const trainerPrepare = vm.runInContext(
  "frames => buildKslGraphFeatures(normalizeKslPositionFrames(extractKslPositionFrames(frames)))",
  context
);
const trainerNormalizeLabel = vm.runInContext("value => normalizeLabel(value)", context);

assert.equal(trainerNormalizeLabel("  Thank   You  "), "thank you");
assert.equal(trainerNormalizeLabel("ＨＥＬＬＯ"), "hello");

const frames = Array.from({ length: KSL_FEATURE_SCHEMA.targetFrameCount }, (_, frameIndex) => {
  const frame = new Array(KSL_FEATURE_SCHEMA.rawFrameWidth).fill(0);
  for (let pointIndex = 0; pointIndex < 50; pointIndex++) {
    if (pointIndex >= 29 && frameIndex % 11 === 0) continue;
    const offset = pointIndex * 3;
    frame[offset] = 0.2 + (pointIndex * 0.003) + (frameIndex * 0.001);
    frame[offset + 1] = 0.3 + (pointIndex * 0.002) - (frameIndex * 0.0005);
    frame[offset + 2] = (pointIndex * 0.0007) + (frameIndex * 0.0001);
  }
  return frame;
});

const sharedFeatures = prepareKslSequence(frames);
const trainerFeatures = trainerPrepare(frames);
assert.equal(trainerFeatures.length, sharedFeatures.length);

let largestDifference = 0;
for (let frameIndex = 0; frameIndex < sharedFeatures.length; frameIndex++) {
  assert.equal(trainerFeatures[frameIndex].length, KSL_FEATURE_SCHEMA.modelFrameWidth);
  for (let valueIndex = 0; valueIndex < sharedFeatures[frameIndex].length; valueIndex++) {
    largestDifference = Math.max(
      largestDifference,
      Math.abs(sharedFeatures[frameIndex][valueIndex] - trainerFeatures[frameIndex][valueIndex])
    );
  }
}

assert.ok(largestDifference < 1e-12, `Trainer/runtime feature drift: ${largestDifference}`);
console.log(`Trainer/runtime KSL features match exactly (${KSL_FEATURE_SCHEMA.targetFrameCount}x${KSL_FEATURE_SCHEMA.modelFrameWidth}).`);
