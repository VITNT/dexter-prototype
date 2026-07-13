import { HandLandmarker, PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";
import {
    KSL_FEATURE_SCHEMA,
    assignStableHandSlots,
    buildKslRawFrame as buildKslRawFrameFromSlots,
    createKslHandSlotState,
    resampleKslRawFrames
} from "./src/pipeline/ksl_feature_pipeline.mjs";

const ASL_SEQUENCE_LENGTH = 30;
const KSL_SEQUENCE_LENGTH = KSL_FEATURE_SCHEMA.targetFrameCount;
const KSL_SCHEMA_VERSION = KSL_FEATURE_SCHEMA.id;
const KSL_LAYOUT_ID = "upper8_hands2x21_xyz";
const KSL_RAW_FRAME_WIDTH = KSL_FEATURE_SCHEMA.rawFrameWidth;
const KSL_CAPTURE_DURATION_MS = 2000;
const VISUAL_HEIGHT = 24;
const VISUAL_HAND_WIDTH = 24;
const VISUAL_WIDTH = VISUAL_HAND_WIDTH * 2;

let cameraStream;
const videoEl = document.getElementById("videoEl");
const cameraToggle = document.getElementById("camBtn");
const modeToggle = document.getElementById("modeBtn");
const recordBtn = document.getElementById("recordBtn");
const burstBtn = document.getElementById("burstBtn");
const statusEl = document.getElementById("recordStatus");
const qualityEl = document.getElementById("qualityIndicator");
const schemaEl = document.getElementById("routingVote");
const countdownEl = document.getElementById("countdownOverlay");
const overlayCanvas = document.getElementsByClassName("canvasEl")[0];
const ctx = overlayCanvas.getContext("2d");

const cropCanvas = document.createElement("canvas");
cropCanvas.width = VISUAL_WIDTH;
cropCanvas.height = VISUAL_HEIGHT;
const cropCtx = cropCanvas.getContext("2d", { willReadFrequently: true });

let activeCamera = "user";
let trackingMode = "hands";
let handTracker;
let poseTracker;
let trackersReady = false;
let lastFrameTime = -1;
let isRunning = false;
let isCapturing = false;
let isBursting = false;
let capturedFrames = [];
let capturedVisualFrames = [];
let capturedTimestamps = [];
let captureResolver = null;
let captureStartedAt = null;
let sourceCaptureFrameCount = 0;
let sourceCaptureFps = 0;
let framesWithAnyHand = 0;
let framesWithTwoHands = 0;
let framesWithPose = 0;
let handSlotState = createKslHandSlotState();

const HAND_BONES = [
    [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]
];

const POSE_BONES = [
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
    [11, 23], [12, 24], [23, 24]
];

const trackersReadyPromise = initTrackers();

async function initTrackers() {
    try {
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
        );

        handTracker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            numHands: trackingMode === "hands" ? 1 : 2,
            minHandDetectionConfidence: 0.5,
            minHandPresenceConfidence: 0.5,
            minTrackingConfidence: 0.5
        });

        poseTracker = await PoseLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
                delegate: "GPU"
            },
            runningMode: "VIDEO"
        });

        // A user may switch modes while the asynchronous model files are loading.
        await handTracker.setOptions({ numHands: trackingMode === "hands" ? 1 : 2 });
        trackersReady = true;
        setStatus("Studio ready.", "#8bc34a");
    } catch (error) {
        console.error("Unable to initialize tracking models:", error);
        setStatus("Tracking models failed to load. Check your connection and refresh.", "#ff5252");
        throw error;
    }
}

window.toggleMode = async function () {
    if (isCapturing || isBursting) return;

    trackingMode = trackingMode === "hands" ? "ksl" : "hands";
    const isKsl = trackingMode === "ksl";
    modeToggle.textContent = isKsl
        ? "Mode: KSL Words (Upper Body + 2 Hands)"
        : "Mode: ASL Alphabet (1 Hand)";
    modeToggle.style.backgroundColor = isKsl ? "#e91e63" : "#ff9800";
    schemaEl.textContent = isKsl
        ? `${KSL_SCHEMA_VERSION} (${KSL_SEQUENCE_LENGTH} frames + hand crops)`
        : `ASL hand21_xyz (${ASL_SEQUENCE_LENGTH} frames)`;

    capturedFrames = [];
    capturedVisualFrames = [];
    handSlotState = createKslHandSlotState();
    isCapturing = false;

    if (trackersReady) {
        await handTracker.setOptions({ numHands: isKsl ? 2 : 1 });
    }
    setStatus("Mode switched. Capture buffers cleared.", "yellow");
};

window.startCamera = async function () {
    try {
        cameraToggle.disabled = true;
        cameraToggle.textContent = trackersReady ? "Starting..." : "Loading trackers...";
        await trackersReadyPromise;
        await handTracker.setOptions({ numHands: trackingMode === "ksl" ? 2 : 1 });

        if (cameraStream) window.stopCamera();
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: activeCamera,
                width: { ideal: 1280 },
                height: { ideal: 720 },
                frameRate: { ideal: 30 }
            }
        });
        videoEl.srcObject = cameraStream;
        await videoEl.play();
        isRunning = true;
        lastFrameTime = -1;
        runDetectionLoop();
        cameraToggle.textContent = "Pause Video";
        cameraToggle.onclick = window.pauseCamera;
        setStatus("Camera active.", "#8bc34a");
    } catch (error) {
        console.error("Unable to access camera:", error);
        const message = error.name === "NotAllowedError"
            ? "Camera permission was denied. Allow camera access and try again."
            : "Unable to access the camera. Open this page from localhost or HTTPS.";
        setStatus(message, "#ff5252");
    } finally {
        cameraToggle.disabled = false;
    }
};

window.pauseCamera = function () {
    videoEl.pause();
    isRunning = false;
    cameraToggle.textContent = "Resume Video";
    cameraToggle.onclick = window.resumeCamera;
};

window.resumeCamera = async function () {
    await videoEl.play();
    isRunning = true;
    lastFrameTime = -1;
    runDetectionLoop();
    cameraToggle.textContent = "Pause Video";
    cameraToggle.onclick = window.pauseCamera;
};

window.stopCamera = function () {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    isRunning = false;
};

window.flipCamera = async function () {
    activeCamera = activeCamera === "user" ? "environment" : "user";
    window.stopCamera();
    await window.startCamera();
};

window.recordGesture = async function () {
    if (isCapturing || isBursting) return;
    try {
        await recordOneSample();
    } catch (error) {
        console.error(error);
        setStatus(error.message, "#ff5252");
    }
};

window.recordBurst = async function () {
    if (isCapturing || isBursting) return;
    const count = clampNumber(document.getElementById("burstCount").value, 1, 20, 5);
    isBursting = true;
    setControlsDisabled(true);

    try {
        for (let index = 0; index < count; index++) {
            setStatus(`Burst sample ${index + 1} of ${count}.`, "#ce93d8");
            await recordOneSample({ manageControls: false });
            if (index < count - 1) await delay(500);
        }
        setStatus(`Burst complete: ${count} independent recordings downloaded.`, "lime");
    } catch (error) {
        console.error(error);
        setStatus(error.message, "#ff5252");
    } finally {
        isBursting = false;
        setControlsDisabled(false);
    }
};

async function recordOneSample({ manageControls = true } = {}) {
    const metadata = readRecordingMetadata();
    if (!isRunning || videoEl.paused) throw new Error("Start the camera before recording.");
    if (!metadata.label) throw new Error("Enter a gesture label first.");
    if (!metadata.signerId) throw new Error("Enter a signer ID so evaluation can be signer-independent.");
    if (!metadata.sessionId) throw new Error("Enter a session ID so near-duplicate sessions stay in one split.");

    if (manageControls) setControlsDisabled(true);
    const delaySeconds = clampNumber(document.getElementById("timerDelay").value, 0, 10, 3);

    try {
        await showCountdown(delaySeconds);
        capturedFrames = [];
        capturedVisualFrames = [];
        capturedTimestamps = [];
        captureStartedAt = null;
        sourceCaptureFrameCount = 0;
        sourceCaptureFps = 0;
        framesWithAnyHand = 0;
        framesWithTwoHands = 0;
        framesWithPose = 0;
        handSlotState = createKslHandSlotState();
        isCapturing = true;

        setStatus(
            trackingMode === "ksl"
                ? "Recording a 2-second KSL clip..."
                : `Recording ${ASL_SEQUENCE_LENGTH} fresh frames...`,
            "#ff5252"
        );
        await new Promise(resolve => { captureResolver = resolve; });
        downloadRecording(metadata);
    } finally {
        isCapturing = false;
        captureResolver = null;
        if (manageControls) setControlsDisabled(false);
    }
}

function readRecordingMetadata() {
    return {
        label: normalizeLabel(document.getElementById("gestureName").value),
        signerId: document.getElementById("signerId").value.trim(),
        sessionId: document.getElementById("sessionId").value.trim()
    };
}

function downloadRecording(metadata) {
    const isKsl = trackingMode === "ksl";
    const recordingId = globalThis.crypto?.randomUUID?.()
        || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const elapsedMs = capturedTimestamps.length > 1
        ? capturedTimestamps[capturedTimestamps.length - 1] - capturedTimestamps[0]
        : 0;
    const measuredFps = elapsedMs > 0
        ? (capturedTimestamps.length - 1) * 1000 / elapsedMs
        : 0;

    const payload = {
        schemaVersion: isKsl ? KSL_SCHEMA_VERSION : "asl-hand21-v1",
        layoutId: isKsl ? KSL_LAYOUT_ID : "hand21_xyz",
        label: metadata.label,
        signerId: metadata.signerId,
        sessionId: metadata.sessionId,
        recordingId,
        capturedAt: new Date().toISOString(),
        cameraFacing: activeCamera,
        frameCount: capturedFrames.length,
        fps: Number(measuredFps.toFixed(2)),
        rawFrameWidth: isKsl ? KSL_RAW_FRAME_WIDTH : 63,
        frames: capturedFrames
    };

    if (isKsl) {
        const sourceFrameTotal = sourceCaptureFrameCount || capturedFrames.length;
        payload.sourceFrameCount = sourceCaptureFrameCount;
        payload.sourceFps = Number(sourceCaptureFps.toFixed(2));
        payload.fps = 30;
        payload.visualFrames = capturedVisualFrames;
        payload.visualShape = [VISUAL_HEIGHT, VISUAL_WIDTH, 1];
        payload.quality = {
            anyHandRatio: ratio(framesWithAnyHand, sourceFrameTotal),
            twoHandRatio: ratio(framesWithTwoHands, sourceFrameTotal),
            poseRatio: ratio(framesWithPose, sourceFrameTotal)
        };
    }

    const blob = new Blob([JSON.stringify(payload)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = buildRecordingFilename(metadata, isKsl);
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);

    const quality = isKsl
        ? `two hands ${Math.round(payload.quality.twoHandRatio * 100)}%, pose ${Math.round(payload.quality.poseRatio * 100)}%`
        : `hand visible ${Math.round(ratio(framesWithAnyHand, capturedFrames.length) * 100)}%`;
    setStatus(`Saved ${capturedFrames.length} frames (${quality}).`, "lime");
}

function runDetectionLoop() {
    if (!isRunning) return;
    if (videoEl.videoWidth === 0 || !trackersReady) {
        requestAnimationFrame(runDetectionLoop);
        return;
    }

    if (videoEl.currentTime !== lastFrameTime) {
        lastFrameTime = videoEl.currentTime;
        const now = performance.now();

        if (overlayCanvas.width !== videoEl.videoWidth || overlayCanvas.height !== videoEl.videoHeight) {
            overlayCanvas.width = videoEl.videoWidth;
            overlayCanvas.height = videoEl.videoHeight;
        }

        ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
        const handResults = handTracker.detectForVideo(videoEl, now);
        const poseResults = trackingMode === "ksl"
            ? poseTracker.detectForVideo(videoEl, now)
            : null;
        const assignedHands = assignHands(handResults, poseResults);

        drawResults(poseResults, assignedHands);
        updateLiveQuality(poseResults, assignedHands);

        if (isCapturing) {
            captureFrame(poseResults, assignedHands, now);
        }
    }

    requestAnimationFrame(runDetectionLoop);
}

function captureFrame(poseResults, assignedHands, timestamp) {
    const isKsl = trackingMode === "ksl";
    let frame;

    if (isKsl) {
        frame = createKslCaptureFrame(poseResults, assignedHands);
        capturedVisualFrames.push(extractHandCropFrame(assignedHands));
    } else if (assignedHands.primary) {
        frame = flattenLandmarks(assignedHands.primary);
    } else {
        frame = new Array(63).fill(0);
    }

    capturedFrames.push(frame);
    capturedTimestamps.push(timestamp);
    if (captureStartedAt === null) captureStartedAt = timestamp;
    if (assignedHands.left || assignedHands.right || assignedHands.primary) framesWithAnyHand++;
    if (assignedHands.left && assignedHands.right) framesWithTwoHands++;
    if (poseResults?.landmarks?.[0]) framesWithPose++;

    const kslDurationComplete = isKsl
        && capturedFrames.length >= 2
        && timestamp - captureStartedAt >= KSL_CAPTURE_DURATION_MS;
    const aslFrameCountComplete = !isKsl && capturedFrames.length >= ASL_SEQUENCE_LENGTH;
    if (kslDurationComplete || aslFrameCountComplete) {
        if (isKsl) resampleCapturedKslClip();
        isCapturing = false;
        const resolve = captureResolver;
        captureResolver = null;
        resolve?.();
    }
}

function resampleCapturedKslClip() {
    sourceCaptureFrameCount = capturedFrames.length;
    const sourceElapsedMs = capturedTimestamps[capturedTimestamps.length - 1] - capturedTimestamps[0];
    sourceCaptureFps = sourceElapsedMs > 0
        ? (sourceCaptureFrameCount - 1) * 1000 / sourceElapsedMs
        : 0;
    capturedFrames = resampleKslRawFrames(capturedFrames, KSL_SEQUENCE_LENGTH);
    capturedVisualFrames = resampleVisualFrames(capturedVisualFrames, KSL_SEQUENCE_LENGTH);

    const firstTimestamp = capturedTimestamps[0];
    const lastTimestamp = capturedTimestamps[capturedTimestamps.length - 1];
    capturedTimestamps = Array.from({ length: KSL_SEQUENCE_LENGTH }, (_, index) =>
        firstTimestamp + (index / (KSL_SEQUENCE_LENGTH - 1)) * (lastTimestamp - firstTimestamp)
    );
}

function resampleVisualFrames(frames, targetLength) {
    if (frames.length === targetLength) return frames.map(frame => frame.slice());
    return Array.from({ length: targetLength }, (_, targetIndex) => {
        const sourcePosition = targetIndex * (frames.length - 1) / (targetLength - 1);
        return frames[Math.round(sourcePosition)].slice();
    });
}

function createKslCaptureFrame(poseResults, assignedHands) {
    return buildKslRawFrameFromSlots({
        poseLandmarks: poseResults?.landmarks?.[0] || [],
        leftHandLandmarks: assignedHands.left || [],
        rightHandLandmarks: assignedHands.right || []
    });
}

function flattenLandmarks(landmarks) {
    const output = new Array(63).fill(0);
    if (!landmarks) return output;
    for (let index = 0; index < Math.min(21, landmarks.length); index++) {
        output[index * 3] = landmarks[index].x;
        output[index * 3 + 1] = landmarks[index].y;
        output[index * 3 + 2] = landmarks[index].z;
    }
    return output;
}

function assignHands(handResults, poseResults = null) {
    const landmarks = handResults?.landmarks || [];
    const candidates = landmarks.map((hand, index) => ({
        hand,
        side: handResults?.handednesses?.[index]?.[0]?.categoryName || "Unknown",
        confidence: handResults?.handednesses?.[index]?.[0]?.score || 0
    }));

    if (trackingMode !== "ksl") {
        candidates.sort((a, b) => b.confidence - a.confidence);
        return { primary: candidates[0]?.hand || null, left: null, right: null };
    }

    const slots = assignStableHandSlots({
        handLandmarks: handResults?.landmarks || [],
        handednesses: handResults?.handednesses || [],
        poseLandmarks: poseResults?.landmarks?.[0] || []
    }, handSlotState);
    handSlotState = slots.state;
    return {
        primary: null,
        left: slots.leftHandLandmarks.length ? slots.leftHandLandmarks : null,
        right: slots.rightHandLandmarks.length ? slots.rightHandLandmarks : null
    };
}

function extractHandCropFrame(assignedHands) {
    cropCtx.fillStyle = "black";
    cropCtx.fillRect(0, 0, VISUAL_WIDTH, VISUAL_HEIGHT);
    drawHandCrop(assignedHands.left, 0);
    drawHandCrop(assignedHands.right, VISUAL_HAND_WIDTH);

    const pixels = cropCtx.getImageData(0, 0, VISUAL_WIDTH, VISUAL_HEIGHT).data;
    const grayscale = new Array(VISUAL_HEIGHT * VISUAL_WIDTH);
    for (let pixel = 0, output = 0; pixel < pixels.length; pixel += 4, output++) {
        const value = (0.299 * pixels[pixel] + 0.587 * pixels[pixel + 1] + 0.114 * pixels[pixel + 2]) / 255;
        grayscale[output] = Math.round(value * 1000) / 1000;
    }
    return grayscale;
}

function drawHandCrop(hand, targetX) {
    if (!hand?.length) return;
    const xs = hand.map(point => point.x * videoEl.videoWidth);
    const ys = hand.map(point => point.y * videoEl.videoHeight);
    const centerX = (Math.min(...xs) + Math.max(...xs)) / 2;
    const centerY = (Math.min(...ys) + Math.max(...ys)) / 2;
    const side = Math.max(
        Math.max(...xs) - Math.min(...xs),
        Math.max(...ys) - Math.min(...ys),
        40
    ) * 1.5;
    const sourceX = Math.max(0, Math.min(videoEl.videoWidth - side, centerX - side / 2));
    const sourceY = Math.max(0, Math.min(videoEl.videoHeight - side, centerY - side / 2));
    const sourceSide = Math.min(side, videoEl.videoWidth - sourceX, videoEl.videoHeight - sourceY);

    cropCtx.drawImage(
        videoEl,
        sourceX, sourceY, sourceSide, sourceSide,
        targetX, 0, VISUAL_HAND_WIDTH, VISUAL_HEIGHT
    );
}

function drawResults(poseResults, assignedHands) {
    const pose = poseResults?.landmarks?.[0];
    if (pose) {
        for (const [from, to] of POSE_BONES) drawBone(pose[from], pose[to], "lime");
    }
    for (const hand of [assignedHands.primary, assignedHands.left, assignedHands.right]) {
        if (!hand) continue;
        for (const [from, to] of HAND_BONES) drawBone(hand[from], hand[to], "cyan");
    }
}

function drawBone(from, to, color) {
    if (!from || !to) return;
    ctx.beginPath();
    ctx.moveTo(from.x * overlayCanvas.width, from.y * overlayCanvas.height);
    ctx.lineTo(to.x * overlayCanvas.width, to.y * overlayCanvas.height);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
}

function updateLiveQuality(poseResults, assignedHands) {
    const hands = Number(Boolean(assignedHands.primary || assignedHands.left)) + Number(Boolean(assignedHands.right));
    const poseReady = Boolean(poseResults?.landmarks?.[0]);
    if (trackingMode === "hands") {
        qualityEl.textContent = hands > 0 ? "Quality: hand detected" : "Quality: show one hand";
        qualityEl.style.color = hands > 0 ? "#8bc34a" : "#ffb74d";
    } else {
        qualityEl.textContent = `Quality: ${hands}/2 hands, upper body ${poseReady ? "detected" : "missing"}`;
        qualityEl.style.color = hands > 0 && poseReady ? "#8bc34a" : "#ffb74d";
    }
}

async function showCountdown(seconds) {
    if (seconds <= 0) return;
    countdownEl.style.display = "block";
    for (let remaining = seconds; remaining > 0; remaining--) {
        countdownEl.textContent = String(remaining);
        await delay(1000);
    }
    countdownEl.style.display = "none";
}

function setControlsDisabled(disabled) {
    recordBtn.disabled = disabled;
    burstBtn.disabled = disabled;
    modeToggle.disabled = disabled;
}

function setStatus(message, color) {
    statusEl.textContent = message;
    statusEl.style.color = color;
}

function ratio(value, total) {
    return total > 0 ? Number((value / total).toFixed(4)) : 0;
}

function clampNumber(value, minimum, maximum, fallback) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? Math.min(maximum, Math.max(minimum, parsed)) : fallback;
}

function safeFilename(value) {
    return value.replace(/[^a-z0-9_-]+/gi, "_").replace(/^_+|_+$/g, "") || "sign";
}

function normalizeLabel(value) {
    return value.normalize("NFKC").trim().replace(/\s+/g, " ").toLocaleLowerCase();
}

function buildRecordingFilename(metadata, isKsl, timestamp = Date.now()) {
    const parts = [
        metadata.label,
        metadata.signerId,
        metadata.sessionId,
        isKsl ? "ksl" : "asl",
        String(timestamp)
    ];
    return `${parts.map(safeFilename).join("__")}.json`;
}

function delay(milliseconds) {
    return new Promise(resolve => setTimeout(resolve, milliseconds));
}
