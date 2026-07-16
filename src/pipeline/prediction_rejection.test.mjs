import assert from "node:assert/strict";
import { hasSufficientTopTwoMargin } from "./prediction_rejection.mjs";

assert.equal(hasSufficientTopTwoMargin([{ conf: 0.80 }, { conf: 0.60 }], 0.12), true);
assert.equal(hasSufficientTopTwoMargin([{ conf: 0.75 }, { conf: 0.68 }], 0.12), false);
assert.equal(hasSufficientTopTwoMargin([{ confidence: 0.90 }], 0.12), true);
assert.equal(hasSufficientTopTwoMargin([], 0.12), false);

console.log("Prediction rejection tests passed: ambiguous top-two results are rejected.");
