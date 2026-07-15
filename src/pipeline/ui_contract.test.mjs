import assert from "node:assert/strict";
import fs from "node:fs";

const pages = [
  { html: "index.html", script: "script.js" },
  { html: "recorder.html", script: "recorder.js" },
  { html: "trainer.html", script: "trainer.js" }
];

for (const page of pages) {
  const html = fs.readFileSync(new URL(`../../${page.html}`, import.meta.url), "utf8");
  const script = fs.readFileSync(new URL(`../../${page.script}`, import.meta.url), "utf8");

  const ids = [...html.matchAll(/\bid=["']([^"']+)["']/g)].map(match => match[1]);
  assert.equal(new Set(ids).size, ids.length, `${page.html} contains duplicate element IDs.`);

  const handlers = [...html.matchAll(/\bonclick=["'](?:window\.)?([A-Za-z_$][\w$]*)/g)]
    .map(match => match[1]);
  for (const handler of handlers) {
    const escaped = handler.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const declaration = new RegExp(`(?:window\\.${escaped}\\s*=|function\\s+${escaped}\\s*\\()`);
    assert.match(script, declaration, `${page.html} calls missing handler ${handler} in ${page.script}.`);
  }

  const moduleSources = [...html.matchAll(/<script[^>]+src=["']([^"']+)["']/g)]
    .map(match => match[1])
    .filter(source => source.startsWith("./") || !source.includes("://"));
  for (const source of moduleSources) {
    const cleanSource = source.replace(/^\.\//, "");
    assert.ok(
      fs.existsSync(new URL(`../../${cleanSource}`, import.meta.url)),
      `${page.html} references missing script ${source}.`
    );
  }
}

const translatorHtml = fs.readFileSync(new URL("../../index.html", import.meta.url), "utf8");
const translatorScript = fs.readFileSync(new URL("../../script.js", import.meta.url), "utf8");
const translatorStyles = fs.readFileSync(new URL("../../styles.css", import.meta.url), "utf8");

assert.match(translatorHtml, /id=["']captureCountdown["']/, "Translator must expose the pre-capture countdown.");
assert.match(translatorScript, /CAPTURE_COUNTDOWN_STEP_MS\s*=\s*500/, "Countdown steps must allow 500 ms to react.");
assert.match(translatorScript, /STATIC_HOLD_DURATION_MS\s*=\s*260/, "Static letters must not require a long hold.");
assert.match(translatorScript, /STATIC_RELEASE_DURATION_MS\s*=\s*160/, "Different letters must rearm quickly.");
assert.match(translatorScript, /MIN_CAPTURE_HAND_CONFIDENCE\s*=\s*0\.65/, "Landmark modes must require reliable hand tracking.");
assert.match(translatorScript, /letterSessionCountdownComplete/, "ASL Letters must remember its one session countdown.");
assert.match(translatorScript, /gestureEstimator\.estimate/, "ASL Letters must use the browser-only Fingerpose classifier.");
assert.doesNotMatch(translatorScript, /predict-letter/, "ASL Letters must not depend on the Python inference backend.");
assert.doesNotMatch(translatorScript, /canonicalizeHandLandmarks|classifyClosedFistThumb/, "Static letters must not use landmark geometry rules.");
assert.doesNotMatch(translatorHtml, /onnxruntime|ort\.min\.js/, "ASL Letters must not load a second inference runtime.");
assert.match(translatorScript, /dynamicLetterPathScore/, "Ordinary whole-hand movement must not trigger Z.");
assert.match(
  translatorScript,
  /requiresTrackedHands\s*=\s*currentTrackingMode\s*!==\s*["']asl_word["']/,
  "WLASL recording must not depend on MediaPipe hand tracking."
);
assert.match(translatorScript, /lockRecognition:\s*false/, "WLASL results must not use the timer-test hand release gate.");
assert.match(translatorScript, /if \(!captureIsActive\)/, "Frame processing must be gated until capture starts.");
assert.match(translatorStyles, /\.video-container\.is-dimmed::after/, "Inactive camera states must be visibly dimmed.");

console.log("UI contracts passed: unique IDs, callable click handlers, and local scripts are present.");
