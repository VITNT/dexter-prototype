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
assert.match(translatorScript, /CAPTURE_COUNTDOWN_STEP_MS\s*=\s*300/, "Countdown numbers must advance every 300 ms.");
assert.match(translatorScript, /STATIC_HOLD_DURATION_MS\s*=\s*260/, "Static letters must not require a long hold.");
assert.match(translatorScript, /STATIC_RELEASE_DURATION_MS\s*=\s*160/, "Different letters must rearm quickly.");
assert.match(translatorScript, /MIN_CAPTURE_HAND_CONFIDENCE\s*=\s*0\.65/, "Landmark modes must require reliable hand tracking.");
assert.match(translatorScript, /letterSessionCountdownComplete/, "ASL Letters must remember its one session countdown.");
assert.match(translatorScript, /gestureEstimator\.estimate/, "ASL Letters must use the browser-only Fingerpose classifier.");
assert.match(translatorScript, /classifyClosedThumbLetter/, "Closed-fist ASL letters must use palm-relative thumb slots.");
assert.match(translatorScript, /hasCrossedIndexMiddle/, "R must be detected from crossed-finger geometry before orientation routing.");
assert.doesNotMatch(translatorScript, /predict-letter/, "ASL Letters must not depend on the Python inference backend.");
assert.doesNotMatch(translatorScript, /canonicalizeHandLandmarks|classifyClosedFistThumb/, "Static letters must not use landmark geometry rules.");
assert.doesNotMatch(translatorHtml, /onnxruntime|ort\.min\.js/, "ASL Letters must not load a second inference runtime.");
assert.match(
  translatorScript,
  /getAslDynamicLetterIntent/,
  "J/Z model routing must use letter-specific motion intent instead of cumulative jitter."
);
assert.match(translatorScript, /DYNAMIC_LETTER_REQUIRED_HITS\s*=\s*4/, "J/Z must use four confirmation hits.");
assert.match(
  translatorScript,
  /hasEmergingAslLetterMotion\(recentFrames\)/,
  "Emerging J/Z fingertip motion must suppress temporary static-letter guesses."
);
assert.match(
  translatorScript,
  /Motion: Detecting…/,
  "Unresolved dynamic letters must use a label-neutral UI state."
);
assert.match(
  translatorScript,
  /Pose: Checking…/,
  "Unconfirmed static guesses must not display a potentially incorrect letter."
);
assert.match(
  translatorScript,
  /lockDynamicLetterRecognition\(primaryWord,\s*releaseSignature\)/,
  "Accepted J/Z letters must enter the dynamic-letter release gate."
);
assert.match(
  translatorScript,
  /handSignatureDistance\(acceptedDynamicSignature,\s*currentSignature\)\s*>=\s*STATIC_RELEASE_SHAPE_THRESHOLD/,
  "J/Z release must accept a changed handshape without requiring the hand to leave the frame."
);
assert.match(
  translatorScript,
  /const visualOnly\s*=\s*currentTrackingMode\s*===\s*["']asl_word["']/,
  "The ASL-word countdown must be visual-only."
);
assert.match(
  translatorScript,
  /\|\| !latestHandsVisible\s*\n\s*\) return;/,
  "Every visible countdown, including ASL Words, must wait for tracked hands."
);
assert.match(
  translatorScript,
  /if \(!latestHandsVisible\) \{\s*if \(captureCountdownTimer\) cancelCaptureCountdown\(\);[\s\S]{0,160}showWaitingForInputGuide\(\);/,
  "ASL Words must cancel its visual timer and return to waiting when idle."
);
assert.match(translatorScript, /lockRecognition:\s*false/, "WLASL results must not use the timer-test hand release gate.");
assert.doesNotMatch(translatorScript, /if \(!captureIsActive \|\| !wlaslBackendReady/, "Countdown state must not gate ASL-word recording.");
assert.match(translatorScript, /if \(visualOnly\) return;/, "ASL-word countdown completion must not clear recognition state.");
assert.match(
  translatorScript,
  /shouldSubmitAslWordClip\(\{ clipSize: clip\.size, trackedHandFrames \}\)/,
  "ASL Words must reject clips that contain no sustained hand tracking before inference."
);
assert.match(
  translatorScript,
  /else if \(latestHandsVisible\) \{\s*analyzeMovement\(detectedHands, detectedPose, currentTimeMs\);/,
  "Landmark word recognition, including KSL, must never analyze pose-only frames."
);
assert.match(translatorScript, /if \(!captureIsActive\)/, "Frame processing must be gated until capture starts.");
assert.match(translatorStyles, /\.video-container\.is-dimmed::after/, "Inactive camera states must be visibly dimmed.");
assert.match(translatorHtml, /id=["']speakTranslationBtn["']/, "Translator must expose a final-translation speech control.");
assert.match(translatorScript, /new SpeechSynthesisUtterance\(speakableTranslation\)/, "Speech must use the final interpreted text.");
assert.match(translatorScript, /setFinalTranslation\(translation[^;]+speakable:/, "Only a successful interpretation may enable speech.");
assert.match(translatorHtml, /onclick=["']backspaceWord\(\)["']/, "Translator must expose a backspace control.");
assert.match(translatorScript, /window\.backspaceWord\s*=\s*function/, "Backspace control must have a callable handler.");
assert.match(translatorScript, /sentenceBuilderDisplay\.lastChild\?\.remove\(\)/, "Backspace must remove the latest recognized item.");

console.log("UI contracts passed: unique IDs, callable click handlers, and local scripts are present.");
