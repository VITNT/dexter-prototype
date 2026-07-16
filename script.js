import { HandLandmarker, PoseLandmarker, FilesetResolver }
    from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";
import {
    KSL_FEATURE_SCHEMA,
    assignStableHandSlots,
    createKslHandSlotState,
    prepareKslSequence
} from "./src/pipeline/ksl_feature_pipeline.mjs";
import {
    normalizeRgbBytes,
    uniformSampleIndices,
    validateWlaslManifest
} from "./src/pipeline/wlasl_video_pipeline.mjs";
import {
    buildHandSignature,
    copyHandSignature,
    handSignatureDistance,
    normalizeHandShapeSignature,
    updatePauseProgress,
    updateReleaseProgress
} from "./src/pipeline/signing_release.mjs";
import {
    predictStaticKsl,
    validateStaticKslModel
} from "./src/pipeline/ksl_static_classifier.mjs";
import {
    buildLegacyKslHandFrame,
    prepareLegacyKslSequence
} from "./src/pipeline/ksl_legacy_word_pipeline.mjs";
import {
    getAslDynamicLetterIntent,
    hasEmergingAslLetterMotion
} from "./src/pipeline/asl_letter_motion.mjs";
import {
    classifyClosedThumbLetter,
    hasCrossedIndexMiddle
} from "./src/pipeline/asl_static_geometry.mjs";
import {
    hasVisibleHandLandmarks,
    shouldSubmitAslWordClip
} from "./src/pipeline/word_hand_presence.mjs";

// skip drawing skeletons on phones — they can't handle it
const isMobileDevice = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
let shouldDrawSkeletons = !isMobileDevice;

// camera stuff
let cameraStream;
let webcamVideo = document.getElementById('webcamFeed');
let webcamContainer = webcamVideo.closest('.video-container');
let cameraToggleBtn = document.getElementById('cameraBtn');
let pauseToggleBtn = document.getElementById('pauseBtn');
let activeCameraFacing = "user";
let isDetecting = false;
let currentTrackingMode = "asl_letter";

// UI elements we update constantly during detection
let rawExtractionDisplay = document.getElementById('liveOutput');
let confidenceProgressBar = document.getElementById('confidenceBar');
let confidencePctDisplay = document.getElementById('confidencePct');
let guessLabelDisplay = document.getElementById('guessLabel');
let sentenceBuilderDisplay = document.getElementById('builtSentence');
let finalTranslationDisplay = document.getElementById('translatedResult');
let speakTranslationButton = document.getElementById('speakTranslationBtn');
let captureGuide = document.getElementById('captureGuide');
let captureStateLabel = document.getElementById('captureStateLabel');
let captureStateDetail = document.getElementById('captureStateDetail');
let captureProgressTrack = document.getElementById('captureProgressTrack');
let captureProgressBar = document.getElementById('captureProgressBar');
let captureCountdown = document.getElementById('captureCountdown');
let captureCountdownNumber = document.getElementById('captureCountdownNumber');

// mediapipe hand + pose trackers
let handTracker;
let poseTracker;
let visionFileset;

// We retain the historical `csl` filenames for compatibility with existing
// Cambodian/Khmer assets while presenting the language as KSL in the UI.
let aslModel, aslWordModel, cslModel;
let kslStaticModel = null;
let latestKslStaticHands = [];
let wlaslModel = null;
let wlaslConfig = null;
let wlaslBackendReady = false;
let wlaslRecorder = null;
let wlaslRecorderChunks = [];
let wlaslCaptureTimer = null;
let wlaslCaptureStartedAt = null;
let wlaslRequestInFlight = false;
let wlaslPendingClip = null;
let wlaslCaptureGeneration = 0;
let wlaslTrackedHandFrames = 0;
let aslModelMetadata = null;
let aslWordModelMetadata = null;
let cslModelMetadata = null;
let aslVocab = [], aslWordVocab = [], cslVocab = [];

// whichever model is selected right now
let activeAIModel;
let activeVocabulary = [];
let activeModelMetadata = null;

// routed word sub-models, keyed by handshape + body zone (for example "fist_face")
let subModelRegistry = {};
let activeSubModel = null;
let activeSubVocabulary = null;
let activeSubMetadata = null;
let wristVelocityBuffer = [];
let routingWristHistory = { Left: [], Right: [] };
let routeHistoryBuffer = [];
let routingDebug = { handshape: "unknown", zone: "neutral", routeKey: null, matched: false, velocity: 0, dynThreshold: 0.90 };

// Legacy models use 30 frames; the face-free KSL graph model uses 60.
let motionBuffer = [];
let legacyKslHandBuffer = [];
let handCropBuffer = [];
let motionTimestampBuffer = [];
let wlaslFrameBuffer = [];
let lastWlaslInferenceAt = 0;
let accumulatedWord = "";
let previousVideoTime = -1;
const LEGACY_REQUIRED_FRAMES = 30;
const KSL_REQUIRED_FRAMES = KSL_FEATURE_SCHEMA.targetFrameCount;
const REQUIRED_FRAMES = LEGACY_REQUIRED_FRAMES;
const CONFIDENCE_THRESHOLD = 0.90;
const KSL_STATIC_CONFIDENCE_THRESHOLD = 0.75;
const WLASL_API_URL = 'http://127.0.0.1:8001';
// Match the 64-frame WLASL training window at typical 25-30 FPS.
const WLASL_CLIP_WINDOW_MS = 2500;
const WLASL_BACKEND_CONFIDENCE = 0.55;
const MIN_CAPTURE_HAND_CONFIDENCE = 0.65;
const STATIC_HOLD_DURATION_MS = 260;
const STATIC_HOLD_MAX_TICK_MS = 100;
const STATIC_HOLD_DECAY_RATE = 0.35;
const STATIC_PREVIEW_REVEAL_PROGRESS = 0.35;
// Give people a consistent half-second to react after recognition before the
// camera is ready to accept another sign.
const RECOGNITION_REACTION_DELAY_MS = 500;
const CAPTURE_COUNTDOWN_START = 3;
const CAPTURE_COUNTDOWN_STEP_MS = 300;
const STATIC_RELEASE_DURATION_MS = 160;
const STATIC_RELEASE_SHAPE_THRESHOLD = 0.12;
const DYNAMIC_LETTER_REQUIRED_HITS = 4;
const DYNAMIC_LETTER_RELEASE_DURATION_MS = 160;
const DYNAMIC_REARM_PAUSE_MS = RECOGNITION_REACTION_DELAY_MS;
const DYNAMIC_REARM_MAX_TICK_MS = 100;
const DYNAMIC_REARM_STILL_VELOCITY = 0.12;

const wlaslCaptureCanvas = document.createElement('canvas');
const wlaslCaptureContext = wlaslCaptureCanvas.getContext('2d', { willReadFrequently: true });

// how many frames in a row the same sign has been detected
let dynamicSignStreak = 0;
let lastDynamicGuess = "";
let staticSignStreak = 0;
let lastStaticGuess = "";
let staticHoldElapsedMs = 0;
let staticHoldLastTickAt = null;
let staticLetterVotes = [];
let staticAwaitingRelease = false;
let acceptedStaticLetter = "";
let acceptedStaticSignature = [];
let staticReleaseStartedAt = null;
let wordAwaitingRelease = false;
let acceptedWordLabel = "";
let dynamicLetterAwaitingRelease = false;
let acceptedDynamicLetter = "";
let acceptedDynamicSignature = [];
let dynamicLetterReleaseElapsedMs = 0;
let dynamicLetterReleaseLastTickAt = null;
let dynamicPauseElapsedMs = 0;
let dynamicPauseLastTickAt = null;
let dynamicPauseLastSignature = [];
let latestHandSignature = [];
let captureIsActive = false;
let captureCountdownTimer = null;
let captureCountdownGeneration = 0;
let latestHandsVisible = false;
let letterSessionCountdownComplete = false;
const VOTE_WINDOW = 5;
let predictionVotes = [];

// the transparent canvas sitting on top of the video for drawing skeletons
const overlayCanvasElement = document.getElementsByClassName('overlayCanvas')[0];
const canvasContext = overlayCanvasElement.getContext('2d');

// Compact grayscale left/right hand crops for optional late-fusion KSL models.
const HAND_CROP_HEIGHT = 24;
const SINGLE_HAND_CROP_WIDTH = 24;
const HAND_CROP_WIDTH = SINGLE_HAND_CROP_WIDTH * 2;
const handCropCanvas = document.createElement('canvas');
handCropCanvas.width = HAND_CROP_WIDTH;
handCropCanvas.height = HAND_CROP_HEIGHT;
const handCropContext = handCropCanvas.getContext('2d', { willReadFrequently: true });

// Missing hands remain zero so KSL preprocessing can attach an honest presence
// mask instead of inventing motion toward the top-left corner.
let runtimeHandSlotState = createKslHandSlotState();

// shorthand for fingerpose's finger names
const { Thumb, Index, Middle, Ring, Pinky } = fp.Finger;
const { NoCurl, HalfCurl, FullCurl } = fp.FingerCurl;

// hand shapes for each letter — fingerpose matches against these
const alphabetGestures = [
    buildSign('A', [Thumb], [], [Index, Middle, Ring, Pinky]),
    buildSign('B', [Index, Middle, Ring, Pinky], [Thumb], []),
    buildSign('C', [], [Thumb, Index, Middle, Ring, Pinky], []),
    buildSign('D', [Index], [Thumb], [Middle, Ring, Pinky]),
    buildSign('E', [], [], [Thumb, Index, Middle, Ring, Pinky]),
    buildSign('F', [Middle, Ring, Pinky], [Thumb, Index], []),
    buildSign('G', [Index], [Thumb], [Middle, Ring, Pinky]),
    buildSign('H', [Index, Middle], [Thumb], [Ring, Pinky]),
    buildSign('I', [Pinky], [], [Thumb, Index, Middle, Ring]),
    buildSign('K', [Index, Middle], [Thumb], [Ring, Pinky]),
    buildSign('L', [Thumb, Index], [], [Middle, Ring, Pinky]),
    buildSign('M', [], [Thumb], [Index, Middle, Ring, Pinky]),
    buildSign('N', [], [Thumb], [Index, Middle, Ring, Pinky]),
    buildSign('O', [], [Thumb, Index, Middle, Ring, Pinky], []),
    buildSign('P', [Index, Middle], [Thumb], [Ring, Pinky]),
    buildSign('Q', [Index], [Thumb], [Middle, Ring, Pinky]),
    buildSign('R', [Index, Middle], [], [Thumb, Ring, Pinky]),
    buildSign('S', [], [], [Thumb, Index, Middle, Ring, Pinky]),
    buildSign('T', [], [], [Thumb, Index, Middle, Ring, Pinky]),
    buildSign('U', [Index, Middle], [], [Thumb, Ring, Pinky]),
    buildSign('V', [Index, Middle], [], [Thumb, Ring, Pinky]),
    buildSign('W', [Index, Middle, Ring], [], [Thumb, Pinky]),
    buildSign('X', [], [Index], [Thumb, Middle, Ring, Pinky]),
    buildSign('Y', [Thumb, Pinky], [], [Index, Middle, Ring])
];
const gestureEstimator = new fp.GestureEstimator(alphabetGestures);

// hand-shape classifier used only for routing to hierarchy sub-models
const routeFist = new fp.GestureDescription('fist');
[Thumb, Index, Middle, Ring, Pinky].forEach(finger => routeFist.addCurl(finger, FullCurl, 1.0));

const routeOpenPalm = new fp.GestureDescription('open_palm');
[Thumb, Index, Middle, Ring, Pinky].forEach(finger => routeOpenPalm.addCurl(finger, NoCurl, 1.0));

const routeIndexPoint = new fp.GestureDescription('index_point');
routeIndexPoint.addCurl(Index, NoCurl, 1.0);
[Thumb, Middle, Ring, Pinky].forEach(finger => routeIndexPoint.addCurl(finger, FullCurl, 1.0));

const routeCurled = new fp.GestureDescription('curled');
[Thumb, Index, Middle, Ring, Pinky].forEach(finger => routeCurled.addCurl(finger, HalfCurl, 1.0));

const routeThumbUp = new fp.GestureDescription('thumb_up');
routeThumbUp.addCurl(Thumb, NoCurl, 1.0);
[Index, Middle, Ring, Pinky].forEach(finger => routeThumbUp.addCurl(finger, FullCurl, 1.0));

const routeGestureEstimator = new fp.GestureEstimator([
    routeFist, routeOpenPalm, routeIndexPoint, routeCurled, routeThumbUp
]);

// which joints to connect when drawing the skeleton overlay
const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]
];
const POSE_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8],
    [9, 10], [11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
    [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
    [11, 23], [12, 24], [23, 24], [23, 25], [24, 26], [25, 27], [26, 28],
    [27, 29], [28, 30], [29, 31], [30, 32], [27, 31], [28, 32]
];

async function initializeMediaPipe() {
    try {
        await tf.setBackend('webgl');
        tf.env().set('WEBGL_FORCE_F16_TEXTURES', true);
        tf.env().set('WEBGL_PACK', true);
        tf.enableProdMode();
    } catch (error) {
        console.warn("WebGL initialization failed, falling back to the default TensorFlow backend.", error);
    }

    visionFileset = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");

    handTracker = await HandLandmarker.createFromOptions(visionFileset, {
        baseOptions: { modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`, delegate: "GPU" },
        runningMode: "VIDEO",
        // A mode can be selected while the asynchronous model setup is running.
        // Read the current mode here instead of always reverting to one hand.
        numHands: currentTrackingMode === "asl_letter" ? 1 : 2,
        minHandDetectionConfidence: MIN_CAPTURE_HAND_CONFIDENCE,
        minHandPresenceConfidence: MIN_CAPTURE_HAND_CONFIDENCE,
        minTrackingConfidence: 0.5
    });

    if (currentTrackingMode !== "asl_letter") await ensurePoseTracker();

    try {
        const aslResponse = await fetch('./asl-labels.json');
        aslVocab = await aslResponse.json();
        aslModel = await tf.loadLayersModel('./asl-model.json');
        aslModelMetadata = await loadOptionalJson('./asl-model_metadata.json');
        validateModelBundle(aslModel, aslVocab, 'ASL');

        activeAIModel = aslModel;
        activeVocabulary = aslVocab;
        activeModelMetadata = aslModelMetadata;
        console.log("ASL model loaded successfully.");
    } catch (error) {
        console.error("Failed to load the ASL model. Check the filenames and paths.", error);
    }

    try {
        const manifestResponse = await fetch('./models/asl/wlasl/manifest.json');
        if (manifestResponse.ok) {
            const manifest = await manifestResponse.json();
            if (manifest.enabled) {
                wlaslConfig = validateWlaslManifest(manifest);
                const labelsResponse = await fetch(`./${wlaslConfig.labelsPath}`);
                if (!labelsResponse.ok) throw new Error(`Unable to load ${wlaslConfig.labelsPath}.`);
                aslWordVocab = await labelsResponse.json();
                wlaslModel = await tf.loadLayersModel(`./${wlaslConfig.modelPath}`);
                validateWlaslModelBundle(wlaslModel, aslWordVocab, wlaslConfig);
                aslWordModel = wlaslModel;
                aslWordModelMetadata = { ...wlaslConfig, adapter: 'wlasl_rgb_video' };
                console.log(`WLASL ${wlaslConfig.subset || ''} video model loaded successfully.`);
            }
        }
    } catch (error) {
        console.error("WLASL bundle is enabled but could not be loaded.", error);
    }

    if (!wlaslModel) {
        try {
            const aslWordResponse = await fetch('./asl_word_labels.json');
            if (aslWordResponse.ok) {
                aslWordVocab = await aslWordResponse.json();
                aslWordModel = await tf.loadLayersModel('./asl_word_model.json');
                aslWordModelMetadata = await loadOptionalJson('./asl_word_model_metadata.json');
                validateModelBundle(aslWordModel, aslWordVocab, 'ASL Words');
                console.log("ASL word model loaded successfully.");
            }
        } catch (error) {
            console.log("No ASL word model installed; ASL Spelling remains available.");
        }
    }

    try {
        const cslResponse = await fetch('./csl_labels.json');
        if (cslResponse.ok) {
            cslVocab = await cslResponse.json();
            cslModel = await tf.loadLayersModel('./csl_model.json');
            cslModelMetadata = await loadOptionalJson('./csl_model_metadata.json');
            validateModelBundle(cslModel, cslVocab, 'KSL');
            console.log("Flat KSL model loaded successfully.");
        } else {
            console.log("No flat KSL model labels found; KSL Words will use routed sub-models when available.");
        }
    } catch (error) {
        console.log("Flat KSL model not available; continuing with routed sub-models only.");
    }

    if (!cslModel) {
        try {
            const [labelsResponse, metadataResponse] = await Promise.all([
                fetch('./models/ksl/legacy-word-v1/labels.json'),
                fetch('./models/ksl/legacy-word-v1/metadata.json')
            ]);
            if (!labelsResponse.ok || !metadataResponse.ok) throw new Error('Legacy KSL manifest is incomplete.');
            cslVocab = await labelsResponse.json();
            cslModelMetadata = await metadataResponse.json();
            cslModel = await tf.loadLayersModel('./models/ksl/legacy-word-v1/model.json');
            validateModelBundle(cslModel, cslVocab, 'KSL words');
            console.log(`Bundled KSL whole-word model loaded with ${cslVocab.length} classes.`);
        } catch (error) {
            console.log("Bundled KSL whole-word model is unavailable.", error);
        }
    }

    try {
        const staticResponse = await fetch('./models/ksl/static-landmark-v1.json');
        if (!staticResponse.ok) throw new Error(`HTTP ${staticResponse.status}`);
        kslStaticModel = validateStaticKslModel(await staticResponse.json());
        console.log(`Static KSL fallback loaded with ${kslStaticModel.classes.length} form classes.`);
    } catch (error) {
        console.log("Static KSL fallback is not installed.", error);
    }

    if (currentTrackingMode === "csl_word") {
        activeAIModel = cslModel;
        activeVocabulary = cslVocab;
        activeModelMetadata = cslModelMetadata;
    }

    try {
        const registryResponse = await fetch('./csl-words/submodel-registry.json');
        if (registryResponse.ok) {
            const registryConfig = await registryResponse.json();
            for (const [routeKey, paths] of Object.entries(registryConfig)) {
                if (routeKey.startsWith("DISABLED_")) continue;

                try {
                    const subLabels = await (await fetch(paths.labels)).json();
                    const subModel = await tf.loadLayersModel(paths.model);
                    const metadataPath = paths.metadata || paths.model.replace(/\.json$/i, '_metadata.json');
                    const metadata = await loadOptionalJson(metadataPath);
                    validateModelBundle(subModel, subLabels, `KSL route ${routeKey}`);
                    subModelRegistry[routeKey] = { model: subModel, labels: subLabels, metadata };
                    console.log(`Sub-model loaded for route "${routeKey}".`);
                } catch (subError) {
                    console.warn(`Sub-model "${routeKey}" failed to load; falling back for that route.`, subError);
                }
            }
        } else {
            console.log("No submodel-registry.json found; running flat-model-only.");
        }
    } catch (error) {
        console.log("Sub-model registry not available; running flat-model-only.");
    }

}

async function ensurePoseTracker() {
    if (poseTracker) return poseTracker;
    if (!visionFileset) throw new Error("MediaPipe is not initialized yet.");
    poseTracker = await PoseLandmarker.createFromOptions(visionFileset, {
        baseOptions: { modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`, delegate: "GPU" },
        runningMode: "VIDEO"
    });
    return poseTracker;
}

async function checkWlaslBackend() {
    try {
        const response = await fetch(`${WLASL_API_URL}/health`);
        const status = response.ok ? await response.json() : null;
        wlaslBackendReady = status?.status === 'ready' && status?.classes === 300;
    } catch (error) {
        wlaslBackendReady = false;
    }
    return wlaslBackendReady;
}

async function loadOptionalJson(path) {
    try {
        const response = await fetch(path);
        return response.ok ? await response.json() : null;
    } catch (error) {
        console.warn(`Optional model metadata was not available at ${path}.`, error);
        return null;
    }
}

function validateModelBundle(model, labels, displayName) {
    const outputCount = model?.outputs?.[0]?.shape?.at(-1);
    if (!Array.isArray(labels) || labels.length === 0) {
        throw new Error(`${displayName} has no labels.`);
    }
    if (Number.isFinite(outputCount) && outputCount !== labels.length) {
        throw new Error(
            `${displayName} model outputs ${outputCount} classes but its label file contains ${labels.length}.`
        );
    }
}

function validateWlaslModelBundle(model, labels, config) {
    validateModelBundle(model, labels, 'WLASL');
    const shape = model?.inputs?.[0]?.shape;
    const expected = [null, config.frameCount, config.imageSize, config.imageSize, 3];
    if (!shape || shape.length !== 5 || expected.slice(1).some((value, index) => shape[index + 1] !== value)) {
        throw new Error(`WLASL model input must be [batch, ${expected.slice(1).join(', ')}].`);
    }
}

// track initialization state so we don't start the camera too early
let isModelsReady = false;
let modelInitPromise = initializeMediaPipe().then(() => {
    isModelsReady = true;
    console.log("System ready for detection.");
});

function buildSign(name, noCurls, halfCurls, fullCurls) {
    const sign = new fp.GestureDescription(name);
    noCurls.forEach(finger => sign.addCurl(finger, NoCurl, 1.0));
    halfCurls.forEach(finger => sign.addCurl(finger, HalfCurl, 1.0));
    fullCurls.forEach(finger => sign.addCurl(finger, FullCurl, 1.0));
    return sign;
}

function evaluateFingerPose(handLandmarks) {
    const keypoints3D = handLandmarks.map(pt => [
        pt.x * overlayCanvasElement.width,
        pt.y * overlayCanvasElement.height,
        pt.z * overlayCanvasElement.width
    ]);
    const estimation = gestureEstimator.estimate(keypoints3D, 7.0);
    if (estimation.gestures.length === 0) return null;
    return estimation.gestures.reduce((best, candidate) => best.score > candidate.score ? best : candidate);
}

// remember dark mode between visits
if (localStorage.getItem('darkMode') === 'true') {
    document.body.classList.add('dark-mode');
}

document.getElementById('darkModeToggle').addEventListener('click', () => {
    document.body.classList.toggle('dark-mode');
    localStorage.setItem('darkMode', document.body.classList.contains('dark-mode'));
});

// mode picker dropdown

window.toggleDropdown = function () {
    document.getElementById('modeMenu').classList.toggle('show');
}

window.selectMode = async function (modeId, modeLabel, element) {
    document.getElementById('currentModeLabel').innerText = modeLabel;

    document.querySelectorAll('.dropdown-item').forEach(item => item.classList.remove('active'));
    element.classList.add('active');

    document.getElementById('modeMenu').classList.remove('show');

    currentTrackingMode = modeId;
    letterSessionCountdownComplete = false;
    window.clearWord();
    staticSignStreak = 0;
    dynamicSignStreak = 0;
    clearHandCache();
    stopWlaslCaptureLoop();

    const guessLabelUI = document.getElementById('guessLabel');
    const rawOutputUI = document.getElementById('liveOutput');
    const outputLangUI = document.getElementById('outputLangLabel');

    if (currentTrackingMode === "csl_word") {
        if (visionFileset) await ensurePoseTracker();
        guessLabelUI.innerText = "Mode: KSL Words";
        rawOutputUI.innerText = `[KSL]`;
        if (outputLangUI) outputLangUI.innerText = "KSL";
        activeAIModel = cslModel;
        activeVocabulary = cslVocab;
        activeModelMetadata = cslModelMetadata;
        if (handTracker) await handTracker.setOptions({ numHands: 2 });

    } else if (currentTrackingMode === "asl_word") {
        if (visionFileset) await ensurePoseTracker();
        await checkWlaslBackend();
        guessLabelUI.innerText = `Mode: ${modeLabel}`;
        rawOutputUI.innerText = (wlaslBackendReady || aslWordModel)
            ? `[ASL WLASL300]`
            : "Start the WLASL service on port 8001";
        if (outputLangUI) outputLangUI.innerText = "English";
        activeAIModel = aslWordModel;
        activeVocabulary = aslWordVocab;
        activeModelMetadata = aslWordModelMetadata;
        if (handTracker) await handTracker.setOptions({ numHands: 2 });
    } else {
        guessLabelUI.innerText = "Mode: ASL Spelling";
        rawOutputUI.innerText = "...";
        if (outputLangUI) outputLangUI.innerText = "English";
        activeAIModel = aslModel;
        activeVocabulary = aslVocab;
        activeModelMetadata = aslModelMetadata;
        if (handTracker) await handTracker.setOptions({ numHands: 1 });
    }
    if (isDetecting) prepareForNextCapture();
}

// clicking anywhere else closes the dropdown
window.addEventListener('click', function (event) {
    // Close the mode picker dropdown if clicking outside
    if (!event.target.closest('.custom-dropdown')) {
        let menu = document.getElementById("modeMenu");
        if (menu && menu.classList.contains('show')) {
            menu.classList.remove('show');
        }
    }

    // Header "Learn" dropdown toggle for mobile
    const navDropdown = event.target.closest('.dex-nav-dropdown');
    const isClickInsideDropdown = event.target.closest('.dex-nav-dropdown-content');

    if (navDropdown && !isClickInsideDropdown) {
        // Toggle the dropdown if clicking the 'Learn' item itself
        event.preventDefault();
        navDropdown.classList.toggle('show');
    } else if (!isClickInsideDropdown) {
        // Close all dropdowns if clicking outside
        document.querySelectorAll('.dex-nav-dropdown.show').forEach(d => d.classList.remove('show'));
    }
});



window.changeMode = async function () {
    const selector = document.getElementById('modeSelector');
    currentTrackingMode = selector.value;
    letterSessionCountdownComplete = false;
    window.clearWord();
    staticSignStreak = 0; dynamicSignStreak = 0;
    clearHandCache();
    stopWlaslCaptureLoop();


    const outputLangUI = document.getElementById('outputLangLabel');

    if (currentTrackingMode === "csl_word") {
        if (visionFileset) await ensurePoseTracker();
        guessLabelDisplay.innerText = "Mode: KSL Words";
        rawExtractionDisplay.innerText = "[KSL]";
        if (outputLangUI) outputLangUI.innerText = "KSL";
        activeAIModel = cslModel;
        activeVocabulary = cslVocab;
        activeModelMetadata = cslModelMetadata;
        if (handTracker) await handTracker.setOptions({ numHands: 2 });

    } else if (currentTrackingMode === "asl_word") {
        if (visionFileset) await ensurePoseTracker();
        guessLabelDisplay.innerText = "Mode: ASL Words";
        await checkWlaslBackend();
        rawExtractionDisplay.innerText = (wlaslBackendReady || aslWordModel)
            ? "[ASL WLASL300]"
            : "Start the WLASL service on port 8001";
        if (outputLangUI) outputLangUI.innerText = "English";
        activeAIModel = aslWordModel;
        activeVocabulary = aslWordVocab;
        activeModelMetadata = aslWordModelMetadata;
        if (handTracker) await handTracker.setOptions({ numHands: 2 });
    } else {
        guessLabelDisplay.innerText = "Mode: ASL Spelling";
        rawExtractionDisplay.innerText = "...";
        if (outputLangUI) outputLangUI.innerText = "English";
        activeAIModel = aslModel;
        activeVocabulary = aslVocab;
        activeModelMetadata = aslModelMetadata;
        if (handTracker) await handTracker.setOptions({ numHands: 1 });
    }
    if (isDetecting) prepareForNextCapture();
}


window.startCam = async function () {
    try {
        if (!isModelsReady) {
            const originalText = cameraToggleBtn.textContent;
            cameraToggleBtn.textContent = "Loading AI Models...";
            cameraToggleBtn.disabled = true;
            setCaptureGuide("analyzing", "Loading recognition", "The camera will be ready shortly.", 0);
            await modelInitPromise;
            cameraToggleBtn.disabled = false;
            cameraToggleBtn.textContent = originalText;
        }

        // Enforce the selected mode after initialization as well. This closes a
        // race where choosing a word mode during startup left the tracker at 1 hand.
        await handTracker.setOptions({
            numHands: currentTrackingMode === "asl_letter" ? 1 : 2
        });
        if (currentTrackingMode !== "asl_letter") await ensurePoseTracker();

        if (cameraStream) window.stopCam();

        // phones get a lighter 640x480 feed, desktops get full HD
        const cameraConfig = isMobileDevice
            ? { facingMode: activeCameraFacing, width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 15 } }
            : { facingMode: activeCameraFacing, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } };

        cameraStream = await navigator.mediaDevices.getUserMedia({ video: cameraConfig });

        webcamVideo.srcObject = cameraStream;
        await webcamVideo.play();

        if (currentTrackingMode === "asl_word" || currentTrackingMode === "asl_letter") {
            await checkWlaslBackend();
        }

        // only mirror the video if we're using the front-facing "selfie" camera
        const videoContainer = webcamVideo.closest('.video-container');
        if (videoContainer) {
            if (activeCameraFacing === 'user') {
                videoContainer.classList.add('mirror-mode');
            } else {
                videoContainer.classList.remove('mirror-mode');
            }
        }

        letterSessionCountdownComplete = false;
        isDetecting = true;
        prepareForNextCapture();
        processVideoFrames();

        cameraToggleBtn.textContent = "Stop Camera";
        cameraToggleBtn.onclick = window.stopCam;

        if (pauseToggleBtn) {
            pauseToggleBtn.style.display = "inline-block";
            pauseToggleBtn.textContent = "Pause Camera";
            pauseToggleBtn.onclick = window.pauseCam;
        }
    } catch (err) {
        setCaptureGuide("off", "Camera unavailable", "Check camera permission, then try again.", 0);
        alert("Unable to access camera. Please check your permissions.");
        console.error(err);
    }
}

window.pauseCam = function () {
    webcamVideo.pause();
    isDetecting = false;
    cancelCaptureCountdown();
    setCaptureActive(false);
    clearHandCache();
    stopWlaslCaptureLoop();
    setCaptureGuide("off", "Camera paused", "Resume when you are ready to continue.", 0);

    if (pauseToggleBtn) {
        pauseToggleBtn.textContent = "Resume Camera";
        pauseToggleBtn.onclick = window.resumeCam;
    }
}

window.resumeCam = async function () {
    clearHandCache();
    await webcamVideo.play();
    if (currentTrackingMode === "asl_word" || currentTrackingMode === "asl_letter") {
        await checkWlaslBackend();
    }
    isDetecting = true;
    prepareForNextCapture();
    processVideoFrames();

    if (pauseToggleBtn) {
        pauseToggleBtn.textContent = "Pause Camera";
        pauseToggleBtn.onclick = window.pauseCam;
    }
}

window.stopCam = function () {
    cancelCaptureCountdown();
    setCaptureActive(false);
    stopWlaslCaptureLoop();
    letterSessionCountdownComplete = false;
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    if (webcamVideo) {
        webcamVideo.pause();
        webcamVideo.removeAttribute('src');
        webcamVideo.removeAttribute('srcObject');
        webcamVideo.srcObject = null;
        webcamVideo.load();
    }
    isDetecting = false;

    cameraToggleBtn.textContent = "Start Camera";
    cameraToggleBtn.onclick = window.startCam;

    if (pauseToggleBtn) {
        pauseToggleBtn.style.display = "none";
        pauseToggleBtn.textContent = "Pause Camera";
        pauseToggleBtn.onclick = window.pauseCam;
    }

    // wipe the skeleton overlay and forget cached hand positions
    canvasContext.clearRect(0, 0, overlayCanvasElement.width, overlayCanvasElement.height);
    clearHandCache();
    setCaptureGuide("off", "Camera off", "Start the camera when you are ready.", 0);
}

function startWlaslCaptureLoop() {
    if (!wlaslBackendReady || !isDetecting || currentTrackingMode !== "asl_word" || !cameraStream) return;
    if (wlaslRecorder && wlaslRecorder.state !== 'inactive') return;

    const generation = wlaslCaptureGeneration;
    const mimeType = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm']
        .find(type => MediaRecorder.isTypeSupported(type));
    try {
        wlaslRecorderChunks = [];
        wlaslRecorder = mimeType
            ? new MediaRecorder(cameraStream, { mimeType })
            : new MediaRecorder(cameraStream);
    } catch (error) {
        rawExtractionDisplay.innerText = "This browser cannot record a WLASL clip";
        setCaptureGuide("waiting", "Capture unavailable", "Try a different browser or ASL mode.", 0);
        console.error("Unable to create WLASL recorder:", error);
        return;
    }

    wlaslRecorder.ondataavailable = event => {
        if (event.data?.size) wlaslRecorderChunks.push(event.data);
    };
    wlaslRecorder.onstop = () => {
        clearTimeout(wlaslCaptureTimer);
        if (generation !== wlaslCaptureGeneration) return;
        const trackedHandFrames = wlaslTrackedHandFrames;
        const clip = new Blob(wlaslRecorderChunks, { type: wlaslRecorder.mimeType || 'video/webm' });
        wlaslRecorder = null;
        wlaslRecorderChunks = [];
        // Keep recording while the previous window is classified. Hand
        // tracking gates inference below, while the recorder stays gap-free.
        if (isDetecting && currentTrackingMode === "asl_word") startWlaslCaptureLoop();
        if (shouldSubmitAslWordClip({ clipSize: clip.size, trackedHandFrames })) {
            if (wlaslRequestInFlight) {
                wlaslPendingClip = { clip, generation, trackedHandFrames };
            } else {
                submitWlaslClip(clip, generation, trackedHandFrames);
            }
        }
    };
    wlaslTrackedHandFrames = 0;
    wlaslRecorder.start();
    setCaptureActive(true);
    wlaslCaptureStartedAt = performance.now();
    if (latestHandsVisible) {
        showWordCaptureProgress(0, WLASL_CLIP_WINDOW_MS);
        startCaptureCountdown();
    } else {
        showWaitingForInputGuide();
    }
    rawExtractionDisplay.innerText = latestHandsVisible
        ? "Watching for one ASL word…"
        : "Waiting for hands…";
    wlaslCaptureTimer = setTimeout(() => {
        if (wlaslRecorder?.state === 'recording') wlaslRecorder.stop();
    }, WLASL_CLIP_WINDOW_MS);
}

function stopWlaslCaptureLoop() {
    wlaslCaptureGeneration++;
    clearTimeout(wlaslCaptureTimer);
    wlaslCaptureTimer = null;
    if (wlaslRecorder && wlaslRecorder.state !== 'inactive') {
        wlaslRecorder.onstop = null;
        wlaslRecorder.stop();
    }
    wlaslRecorder = null;
    wlaslRecorderChunks = [];
    wlaslCaptureStartedAt = null;
    wlaslPendingClip = null;
    wlaslTrackedHandFrames = 0;
}

async function submitWlaslClip(clip, generation, trackedHandFrames) {
    if (!shouldSubmitAslWordClip({ clipSize: clip?.size, trackedHandFrames })) return;
    wlaslRequestInFlight = true;
    rawExtractionDisplay.innerText = "Recognizing…";
    setCaptureGuide("analyzing", "Checking your sign", "Please wait for the result.", 0);
    const form = new FormData();
    form.append('video', clip, 'wlasl-window.webm');
    try {
        const response = await fetch(`${WLASL_API_URL}/predict`, { method: 'POST', body: form });
        const result = await response.json();
        if (!response.ok) throw new Error(result?.detail || `WLASL API returned ${response.status}`);
        if (generation !== wlaslCaptureGeneration || currentTrackingMode !== "asl_word") return;

        const predictions = Array.isArray(result.predictions) ? result.predictions : [];
        // New services make an explicit closed-set/no-sign decision. Keep the
        // confidence fallback so the browser remains compatible with an older
        // service while it is being restarted.
        const detected = result.detected === true
            || (result.detected == null && result.confidence >= WLASL_BACKEND_CONFIDENCE);
        if (detected && predictions.length) {
            showDynamicMatch(result.word, result.confidence);
            acceptDynamicWord(predictions, undefined, { lockRecognition: false });
            rawExtractionDisplay.innerText = result.word;
            setCaptureGuide("accepted", `${result.word} accepted`, "Recognition is still running.", 1);
        } else {
            showDynamicIdle(result.confidence || 0, true);
            rawExtractionDisplay.innerText = result.rejectionReason === 'no_motion'
                ? "Waiting for a complete sign…"
                : "No confident ASL word yet";
            setCaptureGuide(
                "ready",
                "Try one complete sign",
                "Recognition is still running; begin the next sign naturally.",
                0
            );
        }
    } catch (error) {
        console.error("WLASL prediction failed:", error);
        rawExtractionDisplay.innerText = "WLASL service unavailable";
        setCaptureGuide("waiting", "Recognition unavailable", "Check the WLASL service, then try again.", 0);
    } finally {
        wlaslRequestInFlight = false;
        const pending = wlaslPendingClip;
        wlaslPendingClip = null;
        if (pending && isDetecting && currentTrackingMode === "asl_word") {
            submitWlaslClip(pending.clip, pending.generation, pending.trackedHandFrames);
        }
    }
}

window.flipCam = async function () {
    activeCameraFacing = activeCameraFacing === "user" ? "environment" : "user";
    window.stopCam();
    await window.startCam();
}



window.clearWord = function () {
    accumulatedWord = "";
    sentenceBuilderDisplay.innerHTML = "";

    setFinalTranslation("");
    rawExtractionDisplay.innerText = "...";
}

window.backspaceWord = function () {
    if (!sentenceBuilderDisplay) return;

    // Recognized words add a trailing space node. Remove that separator first,
    // then remove the latest complete word dropdown or letter span.
    while (sentenceBuilderDisplay.lastChild?.nodeType === Node.TEXT_NODE
        && !sentenceBuilderDisplay.lastChild.textContent.trim()) {
        sentenceBuilderDisplay.lastChild.remove();
    }

    sentenceBuilderDisplay.lastChild?.remove();
    syncSentenceState();
    setFinalTranslation("");
    rawExtractionDisplay.innerText = "...";
}

window.translateSentence = async function () {
    if (!accumulatedWord) return;

    setFinalTranslation("Dexter is thinking...");

    try {
        const response = await fetch('/api/translate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ aslString: accumulatedWord })
        });

        if (!response.ok) {
            console.error("Backend error:", await response.text());
            setFinalTranslation("Error: Translation service failed.");
            return;
        }

        const data = await response.json();
        const translation = typeof data.translation === "string" ? data.translation.trim() : "";
        setFinalTranslation(translation || "Error: Empty response.", { speakable: Boolean(translation) });
    } catch (err) {
        console.error("Network issue:", err);
        setFinalTranslation("Network Error. Please try again.");
    }
}

const speechSynthesisSupported = "speechSynthesis" in window && "SpeechSynthesisUtterance" in window;
let speakableTranslation = "";
let activeTranslationUtterance = null;

function updateSpeakTranslationButton(isSpeaking = false) {
    if (!speakTranslationButton) return;

    const canSpeak = speechSynthesisSupported && Boolean(speakableTranslation);
    speakTranslationButton.disabled = !canSpeak;
    speakTranslationButton.classList.toggle("is-speaking", isSpeaking);
    speakTranslationButton.setAttribute("aria-label", isSpeaking ? "Stop speaking" : "Speak translation");
    speakTranslationButton.title = speechSynthesisSupported
        ? (isSpeaking ? "Stop speaking" : "Speak translation")
        : "Text to speech is not supported by this browser";
}

function cancelTranslationSpeech() {
    if (speechSynthesisSupported) window.speechSynthesis.cancel();
    activeTranslationUtterance = null;
    updateSpeakTranslationButton(false);
}

function setFinalTranslation(text, { speakable = false } = {}) {
    cancelTranslationSpeech();
    speakableTranslation = speakable ? text.trim() : "";
    if (finalTranslationDisplay) finalTranslationDisplay.innerText = text;
    updateSpeakTranslationButton(false);
}

window.speakTranslation = function () {
    if (!speechSynthesisSupported || !speakableTranslation) return;

    if (activeTranslationUtterance) {
        cancelTranslationSpeech();
        return;
    }

    const utterance = new SpeechSynthesisUtterance(speakableTranslation);
    utterance.lang = "en-US";
    utterance.onstart = () => {
        if (activeTranslationUtterance === utterance) updateSpeakTranslationButton(true);
    };
    utterance.onend = () => {
        if (activeTranslationUtterance !== utterance) return;
        activeTranslationUtterance = null;
        updateSpeakTranslationButton(false);
    };
    utterance.onerror = utterance.onend;

    activeTranslationUtterance = utterance;
    updateSpeakTranslationButton(true);
    window.speechSynthesis.speak(utterance);
}

updateSpeakTranslationButton(false);



const calculateDistance = (p1, p2) => Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));

// stitch together pose + both hands into one flat array the model expects
function buildHolisticFrame(poseData, handData, assignedHands = null) {
    let rawPose = new Array(33 * 3).fill(0);

    if (poseData?.landmarks?.length > 0) {
        const poseLandmarks = poseData.landmarks[0];
        for (let i = 0; i < poseLandmarks.length; i++) {
            rawPose[i * 3] = poseLandmarks[i].x;
            rawPose[i * 3 + 1] = poseLandmarks[i].y;
            rawPose[i * 3 + 2] = poseLandmarks[i].z;
        }
    }

    const assigned = assignedHands || assignDetectedHands(handData, poseData);
    const leftHand = flattenHand(assigned.left);
    const rightHand = flattenHand(assigned.right);

    return [...rawPose, ...leftHand, ...rightHand];
}

function flattenHand(hand) {
    const output = new Array(63).fill(0);
    if (!hand) return output;
    for (let index = 0; index < Math.min(21, hand.length); index++) {
        output[index * 3] = hand[index].x;
        output[index * 3 + 1] = hand[index].y;
        output[index * 3 + 2] = hand[index].z;
    }
    return output;
}

function assignDetectedHands(handData, poseData = null) {
    const slots = assignStableHandSlots({
        handLandmarks: handData?.landmarks || [],
        handednesses: handData?.handednesses || [],
        poseLandmarks: poseData?.landmarks?.[0] || []
    }, runtimeHandSlotState);
    runtimeHandSlotState = slots.state;
    return {
        left: slots.leftHandLandmarks.length ? slots.leftHandLandmarks : null,
        right: slots.rightHandLandmarks.length ? slots.rightHandLandmarks : null
    };
}

function extractTwoHandCropFrame(handData, assignedHands = null) {
    const assigned = assignedHands || assignDetectedHands(handData);
    handCropContext.fillStyle = 'black';
    handCropContext.fillRect(0, 0, HAND_CROP_WIDTH, HAND_CROP_HEIGHT);
    drawHandCrop(assigned.left, 0);
    drawHandCrop(assigned.right, SINGLE_HAND_CROP_WIDTH);

    const pixels = handCropContext.getImageData(0, 0, HAND_CROP_WIDTH, HAND_CROP_HEIGHT).data;
    const grayscale = new Array(HAND_CROP_HEIGHT * HAND_CROP_WIDTH);
    for (let pixel = 0, output = 0; pixel < pixels.length; pixel += 4, output++) {
        const value = (
            0.299 * pixels[pixel] +
            0.587 * pixels[pixel + 1] +
            0.114 * pixels[pixel + 2]
        ) / 255;
        grayscale[output] = Math.round(value * 1000) / 1000;
    }
    return grayscale;
}

function drawHandCrop(hand, targetX) {
    if (!hand?.length || !webcamVideo.videoWidth || !webcamVideo.videoHeight) return;
    const xs = hand.map(point => point.x * webcamVideo.videoWidth);
    const ys = hand.map(point => point.y * webcamVideo.videoHeight);
    const centerX = (Math.min(...xs) + Math.max(...xs)) / 2;
    const centerY = (Math.min(...ys) + Math.max(...ys)) / 2;
    const requestedSide = Math.max(
        Math.max(...xs) - Math.min(...xs),
        Math.max(...ys) - Math.min(...ys),
        40
    ) * 1.5;
    const sourceX = Math.max(0, Math.min(webcamVideo.videoWidth - requestedSide, centerX - requestedSide / 2));
    const sourceY = Math.max(0, Math.min(webcamVideo.videoHeight - requestedSide, centerY - requestedSide / 2));
    const sourceSide = Math.min(
        requestedSide,
        webcamVideo.videoWidth - sourceX,
        webcamVideo.videoHeight - sourceY
    );
    if (sourceSide <= 0) return;

    handCropContext.drawImage(
        webcamVideo,
        sourceX, sourceY, sourceSide, sourceSide,
        targetX, 0, SINGLE_HAND_CROP_WIDTH, HAND_CROP_HEIGHT
    );
}

// center everything relative to the body so the model isn't thrown off
// by where the person is standing in the frame
function centerFrameData(rawFrame) {
    // try nose first, then fall back to wrists
    let anchorX = rawFrame[0];
    let anchorY = rawFrame[1];

    if (anchorX === 0 && anchorY === 0) {
        anchorX = rawFrame[99];
        anchorY = rawFrame[100];
    }
    if (anchorX === 0 && anchorY === 0) {
        anchorX = rawFrame[162];
        anchorY = rawFrame[163];
    }

    if (anchorX === 0 && anchorY === 0) return rawFrame;

    let centered = [];
    for (let i = 0; i < rawFrame.length; i += 3) {
        let x = rawFrame[i];
        let y = rawFrame[i + 1];
        let z = rawFrame[i + 2];

        if (x === 0 && y === 0) {
            centered.push(0, 0, 0);
        } else {
            centered.push(x - anchorX, y - anchorY, z);
        }
    }
    return centered;
}

function clearHandCache({ preserveReleaseGate = false } = {}) {
    runtimeHandSlotState = createKslHandSlotState();
    predictionVotes = [];
    motionBuffer = [];
    legacyKslHandBuffer = [];
    handCropBuffer = [];
    motionTimestampBuffer = [];
    wlaslFrameBuffer = [];
    lastWlaslInferenceAt = 0;
    wristVelocityBuffer = [];
    routingWristHistory = { Left: [], Right: [] };
    routeHistoryBuffer = [];
    activeSubModel = null;
    activeSubVocabulary = null;
    activeSubMetadata = null;
    resetStaticRecognitionState();
    if (!preserveReleaseGate) {
        resetWordRecognitionState();
        resetDynamicLetterRecognitionState();
    }
    routingDebug = { handshape: "unknown", zone: "neutral", routeKey: null, matched: false, velocity: 0, dynThreshold: CONFIDENCE_THRESHOLD };
}

function setCaptureGuide(state, label, detail, progress = 0) {
    if (!captureGuide) return;
    const normalizedProgress = Math.max(0, Math.min(1, Number(progress) || 0));
    const percent = Math.round(normalizedProgress * 100);

    captureGuide.dataset.state = state;
    syncCameraCaptureAppearance();
    if (captureStateLabel.textContent !== label) captureStateLabel.textContent = label;
    if (captureStateDetail.textContent !== detail) captureStateDetail.textContent = detail;
    captureProgressBar.style.width = `${percent}%`;
    captureProgressTrack.setAttribute('aria-valuenow', String(percent));
}

function syncCameraCaptureAppearance() {
    const wordModeIsIdle = currentTrackingMode === "asl_word" && isDetecting && !latestHandsVisible;
    const shouldDim = Boolean(cameraStream) && (wordModeIsIdle || !captureIsActive);
    webcamContainer?.classList.toggle('is-dimmed', shouldDim);
}

function setCaptureActive(active) {
    captureIsActive = Boolean(active);
    syncCameraCaptureAppearance();
}

function hideCaptureCountdown() {
    if (captureCountdown) captureCountdown.hidden = true;
}

function cancelCaptureCountdown() {
    captureCountdownGeneration++;
    clearTimeout(captureCountdownTimer);
    captureCountdownTimer = null;
    hideCaptureCountdown();
}

function prepareForNextCapture() {
    cancelCaptureCountdown();
    latestHandsVisible = false;
    if (!isDetecting) return;
    if (currentTrackingMode === "asl_word" && wlaslBackendReady) {
        setCaptureActive(true);
        showReadyCaptureGuide();
        startWlaslCaptureLoop();
        return;
    }
    setCaptureActive(false);
    showWaitingForInputGuide();
    if (currentTrackingMode === "asl_letter" && letterSessionCountdownComplete) {
        if (latestHandsVisible) {
            setCaptureActive(true);
            showReadyCaptureGuide();
        }
        return;
    }
}

function showCaptureCountdown(value) {
    if (captureCountdownNumber) captureCountdownNumber.textContent = String(value);
    if (captureCountdown) captureCountdown.hidden = false;
    setCaptureGuide(
        "countdown",
        currentTrackingMode === "asl_word" ? `Recognition active · ${value}` : `Capture starts in ${value}`,
        currentTrackingMode === "asl_word"
            ? "Visual timing only — every frame is still being recorded and analyzed."
            : "Hold your starting position. Frames are not being saved yet.",
        (CAPTURE_COUNTDOWN_START - value) / CAPTURE_COUNTDOWN_START
    );
}

function startCaptureCountdown() {
    const visualOnly = currentTrackingMode === "asl_word";
    if (
        !isDetecting
        || (!visualOnly && captureIsActive)
        || captureCountdownTimer
        || !latestHandsVisible
    ) return;

    const generation = ++captureCountdownGeneration;
    let value = CAPTURE_COUNTDOWN_START;
    showCaptureCountdown(value);

    const advance = () => {
        if (generation !== captureCountdownGeneration || !isDetecting) return;
        if (!latestHandsVisible) {
            cancelCaptureCountdown();
            if (!visualOnly) setCaptureActive(false);
            showWaitingForInputGuide();
            return;
        }

        value--;
        if (value > 0) {
            showCaptureCountdown(value);
            captureCountdownTimer = setTimeout(advance, CAPTURE_COUNTDOWN_STEP_MS);
            return;
        }

        captureCountdownTimer = null;
        hideCaptureCountdown();
        if (visualOnly) return;
        clearHandCache();
        if (currentTrackingMode === "asl_letter") letterSessionCountdownComplete = true;
        setCaptureActive(true);
        showReadyCaptureGuide();
    };

    captureCountdownTimer = setTimeout(advance, CAPTURE_COUNTDOWN_STEP_MS);
}

function showWaitingForInputGuide() {
    setCaptureGuide(
        "waiting",
        currentTrackingMode === "asl_word" ? "Get ready to sign" : "Finding your hands",
        currentTrackingMode === "asl_letter"
            ? "Place one hand fully inside the camera frame."
            : (currentTrackingMode === "asl_word"
                ? "Show your hands to start recognition and the timing cue."
                : "Keep your hands and upper body visible."),
        0
    );
}

function showReadyCaptureGuide() {
    if (!isDetecting) return;
    setCaptureGuide(
        "ready",
        currentTrackingMode === "asl_letter" ? "Ready for a letter" : "Ready for one word",
        currentTrackingMode === "asl_letter"
            ? "Make the handshape and hold it steady."
            : "Sign at a comfortable pace and keep your hands visible.",
        0
    );
}

function showWordCaptureProgress(elapsedMs, durationMs) {
    const progress = Math.max(0, Math.min(1, elapsedMs / durationMs));
    setCaptureGuide(
        progress < 1 ? "capturing" : "ready",
        progress < 1 ? "Capturing your sign" : "Checking your sign",
        progress < 1
            ? "This bar shows the saved clip window, not model confidence."
            : "Finish the sign naturally while Dexter checks the capture.",
        progress
    );
}

function resetStaticHoldTimer({ clearCandidate = true } = {}) {
    staticHoldElapsedMs = 0;
    staticHoldLastTickAt = null;
    staticSignStreak = 0;
    if (clearCandidate) {
        lastStaticGuess = "";
        staticLetterVotes = [];
    }
}

function resetStaticRecognitionState() {
    resetStaticHoldTimer();
    staticAwaitingRelease = false;
    acceptedStaticLetter = "";
    acceptedStaticSignature = [];
    staticReleaseStartedAt = null;
}

function timerDelta(timestamp) {
    if (!Number.isFinite(staticHoldLastTickAt)) {
        staticHoldLastTickAt = timestamp;
        return 0;
    }
    const delta = Math.max(0, Math.min(STATIC_HOLD_MAX_TICK_MS, timestamp - staticHoldLastTickAt));
    staticHoldLastTickAt = timestamp;
    return delta;
}

function decayStaticHold(timestamp) {
    staticHoldElapsedMs = Math.max(
        0,
        staticHoldElapsedMs - timerDelta(timestamp) * STATIC_HOLD_DECAY_RATE
    );
    if (staticHoldElapsedMs === 0) lastStaticGuess = "";
    return staticHoldElapsedMs / STATIC_HOLD_DURATION_MS;
}

function updateStaticRelease(handData, timestamp) {
    if (!staticAwaitingRelease) return false;
    const currentSignature = normalizeHandShapeSignature(buildHandSignature(handData));
    const released = currentSignature.length === 0
        || handSignatureDistance(acceptedStaticSignature, currentSignature)
            >= STATIC_RELEASE_SHAPE_THRESHOLD;

    if (!released) {
        staticReleaseStartedAt = null;
        setCaptureGuide(
            "accepted",
            `${acceptedStaticLetter} accepted`,
            "Release or make a clearly different handshape to continue.",
            1
        );
        return true;
    }

    if (staticReleaseStartedAt === null) staticReleaseStartedAt = timestamp;
    const progress = Math.min(1, (timestamp - staticReleaseStartedAt) / STATIC_RELEASE_DURATION_MS);
    setCaptureGuide(
        "accepted",
        "Release detected",
        "Keep your hand relaxed until the bar is full.",
        progress
    );

    if (progress >= 1) {
        resetStaticRecognitionState();
        clearHandCache();
        prepareForNextCapture();
    }
    return true;
}

function resetWordRecognitionState() {
    wordAwaitingRelease = false;
    acceptedWordLabel = "";
    dynamicPauseElapsedMs = 0;
    dynamicPauseLastTickAt = null;
    dynamicPauseLastSignature = [];
    latestHandSignature = [];
}

function resetDynamicLetterRecognitionState() {
    dynamicLetterAwaitingRelease = false;
    acceptedDynamicLetter = "";
    acceptedDynamicSignature = [];
    dynamicLetterReleaseElapsedMs = 0;
    dynamicLetterReleaseLastTickAt = null;
}

function lockDynamicLetterRecognition(letter, signature = latestHandSignature) {
    setCaptureActive(false);
    dynamicLetterAwaitingRelease = true;
    acceptedDynamicLetter = letter;
    acceptedDynamicSignature = normalizeHandShapeSignature(signature);
    dynamicLetterReleaseElapsedMs = 0;
    dynamicLetterReleaseLastTickAt = performance.now();
    setCaptureGuide(
        "accepted",
        `${letter} accepted`,
        "Relax or change your handshape before signing the next letter.",
        0
    );
}

function updateDynamicLetterRelease(handData, timestamp) {
    if (!dynamicLetterAwaitingRelease) return false;
    const rawDelta = Number.isFinite(dynamicLetterReleaseLastTickAt)
        ? timestamp - dynamicLetterReleaseLastTickAt
        : 0;
    const currentSignature = normalizeHandShapeSignature(buildHandSignature(handData));
    const shapeChanged = acceptedDynamicSignature.length > 0
        && currentSignature.length > 0
        && handSignatureDistance(acceptedDynamicSignature, currentSignature)
            >= STATIC_RELEASE_SHAPE_THRESHOLD;
    const isReleased = currentSignature.length === 0 || shapeChanged;

    dynamicLetterReleaseElapsedMs = updateReleaseProgress({
        elapsedMs: dynamicLetterReleaseElapsedMs,
        deltaMs: rawDelta,
        isReleased,
        durationMs: DYNAMIC_LETTER_RELEASE_DURATION_MS
    });
    dynamicLetterReleaseLastTickAt = timestamp;

    const progress = dynamicLetterReleaseElapsedMs / DYNAMIC_LETTER_RELEASE_DURATION_MS;
    setCaptureGuide(
        "accepted",
        isReleased ? "Release detected" : `${acceptedDynamicLetter} accepted`,
        isReleased
            ? "Keep the relaxed handshape until the bar is full."
            : "Relax or change your handshape before signing the next letter.",
        progress
    );

    if (progress < 1) return true;

    resetDynamicLetterRecognitionState();
    clearHandCache();
    prepareForNextCapture();
    return true;
}

function lockWordRecognition(label, signature = latestHandSignature) {
    setCaptureActive(false);
    wordAwaitingRelease = true;
    acceptedWordLabel = label;
    dynamicPauseElapsedMs = 0;
    dynamicPauseLastTickAt = performance.now();
    dynamicPauseLastSignature = copyHandSignature(signature);
    setCaptureGuide(
        "accepted",
        `${label} accepted`,
        "Finish the movement, then pause briefly to continue.",
        0
    );
}

function updateWordRelease(handData, timestamp) {
    const currentSignature = buildHandSignature(handData);
    latestHandSignature = currentSignature;
    const rawDelta = Number.isFinite(dynamicPauseLastTickAt)
        ? timestamp - dynamicPauseLastTickAt
        : 0;
    const delta = Math.max(0, Math.min(DYNAMIC_REARM_MAX_TICK_MS, rawDelta));
    const movement = handSignatureDistance(dynamicPauseLastSignature, currentSignature);
    const velocity = rawDelta > 0 && Number.isFinite(movement)
        ? movement * 1000 / rawDelta
        : Infinity;
    const isPaused = currentSignature.length === 0
        || (dynamicPauseLastSignature.length > 0 && velocity <= DYNAMIC_REARM_STILL_VELOCITY);

    dynamicPauseElapsedMs = updatePauseProgress({
        elapsedMs: dynamicPauseElapsedMs,
        deltaMs: delta,
        isPaused,
        durationMs: DYNAMIC_REARM_PAUSE_MS,
        maxTickMs: DYNAMIC_REARM_MAX_TICK_MS
    });
    dynamicPauseLastTickAt = timestamp;
    dynamicPauseLastSignature = copyHandSignature(currentSignature);

    const progress = dynamicPauseElapsedMs / DYNAMIC_REARM_PAUSE_MS;
    setCaptureGuide(
        "accepted",
        isPaused ? "Pause to continue" : `${acceptedWordLabel} accepted`,
        isPaused
            ? "Hold still briefly until the rearm bar is full."
            : "Finish the movement, then pause briefly to continue.",
        progress
    );

    if (progress < 1) return;

    resetWordRecognitionState();
    clearHandCache();
    prepareForNextCapture();
}

function updateWlaslCaptureProgress(timestamp) {
    if (wlaslRequestInFlight) {
        setCaptureGuide("analyzing", "Checking your sign", "Please wait for the result.", 0);
        return;
    }
    if (wlaslRecorder?.state !== 'recording' || !Number.isFinite(wlaslCaptureStartedAt)) return;
    showWordCaptureProgress(timestamp - wlaslCaptureStartedAt, WLASL_CLIP_WINDOW_MS);
}

function classifyHandshape(handLandmarks) {
    if (!handLandmarks?.length) return "unknown";

    const keypoints3D = handLandmarks.map(point => [
        point.x * overlayCanvasElement.width,
        point.y * overlayCanvasElement.height,
        point.z * overlayCanvasElement.width
    ]);
    const estimation = routeGestureEstimator.estimate(keypoints3D, 7.0);

    if (estimation.gestures.length > 0) {
        return estimation.gestures.reduce((prev, current) => (prev.score > current.score) ? prev : current).name;
    }

    return "unknown";
}

function classifyZone(poseData, handLandmarks) {
    if (!handLandmarks?.length || !poseData?.landmarks?.length) return "neutral";

    const pose = poseData.landmarks[0];
    if (pose.length < 15) return "neutral";

    const knuckleY = handLandmarks[9].y;
    const shoulderLineY = (pose[11].y + pose[12].y) / 2;
    const elbowLineY = (pose[13].y + pose[14].y) / 2;

    if (knuckleY < shoulderLineY) return "face";
    if (knuckleY < elbowLineY) return "chest";
    return "low";
}

// Word models receive both hands. The lightweight sub-model router still needs
// one representative handshape, so use whichever visible hand is moving more.
// This handles asymmetric two-hand signs while naturally falling back to the
// only visible hand for one-hand signs.
function selectRoutingHand(handData) {
    const landmarks = handData?.landmarks || [];
    if (landmarks.length === 0) return null;

    const now = performance.now();
    const candidates = landmarks.map((hand, index) => {
        const classification = handData?.handednesses?.[index]?.[0];
        const side = classification?.categoryName || `Hand${index}`;

        if (!routingWristHistory[side]) routingWristHistory[side] = [];
        const history = routingWristHistory[side];
        history.push({ x: hand[0].x, y: hand[0].y, time: now });
        while (history.length > 10 || (history[0] && now - history[0].time > 750)) {
            history.shift();
        }

        let movement = 0;
        for (let i = 1; i < history.length; i++) {
            const dx = history[i].x - history[i - 1].x;
            const dy = history[i].y - history[i - 1].y;
            movement += Math.sqrt(dx * dx + dy * dy);
        }

        return {
            hand,
            side,
            movement,
            confidence: classification?.score || 0
        };
    });

    candidates.sort((a, b) =>
        (b.movement - a.movement) || (b.confidence - a.confidence)
    );
    return candidates[0];
}

function routeToSubModel(handshape, zone) {
    const routeKey = `${handshape}_${zone}`;
    const entry = subModelRegistry[routeKey];

    if (entry) {
        activeSubModel = entry.model;
        activeSubVocabulary = entry.labels;
        activeSubMetadata = entry.metadata || null;
        routingDebug = { ...routingDebug, handshape, zone, routeKey, matched: true };
        return routeKey;
    }

    activeSubModel = null;
    activeSubVocabulary = null;
    activeSubMetadata = null;
    routingDebug = { ...routingDebug, handshape, zone, routeKey, matched: false };
    return null;
}

function getDynamicThreshold() {
    if (wristVelocityBuffer.length < 2) return CONFIDENCE_THRESHOLD;

    const recent = wristVelocityBuffer.slice(-10);
    let totalDelta = 0;
    let deltaCount = 0;
    for (let i = 1; i < recent.length; i++) {
        // Do not interpret a switch between left and right routing hands as motion.
        if (recent[i].side !== recent[i - 1].side) continue;
        const dx = recent[i].x - recent[i - 1].x;
        const dy = recent[i].y - recent[i - 1].y;
        totalDelta += Math.sqrt(dx * dx + dy * dy);
        deltaCount++;
    }

    const avgVelocity = deltaCount > 0 ? totalDelta / deltaCount : 0;
    const minVelocity = 0.005;
    const maxVelocity = 0.015;
    const minThreshold = 0.80;
    const maxThreshold = 0.95;
    const clampedVelocity = Math.max(minVelocity, Math.min(maxVelocity, avgVelocity));
    const velocityRatio = (clampedVelocity - minVelocity) / (maxVelocity - minVelocity);
    const dynThreshold = minThreshold + velocityRatio * (maxThreshold - minThreshold);

    routingDebug = { ...routingDebug, velocity: avgVelocity, dynThreshold };
    return dynThreshold;
}

function updateRoutingDebugUI() {
    const debugEl = document.getElementById('routingDebugText');
    if (!debugEl) return;

    debugEl.innerText = `Route: ${routingDebug.handshape}/${routingDebug.zone} -> ` +
        `${routingDebug.matched ? routingDebug.routeKey + " (sub-model)" : "flat model"} | ` +
        `v=${routingDebug.velocity.toFixed(4)} | thr=${routingDebug.dynThreshold.toFixed(2)}`;
}

function resolveDominantRoute() {
    const zoneTally = {};
    let dominantZone = "neutral";
    let dominantZoneCount = 0;

    for (const key of routeHistoryBuffer) {
        const sepIdx = key.lastIndexOf("_");
        const zone = sepIdx === -1 ? "neutral" : key.slice(sepIdx + 1);
        zoneTally[zone] = (zoneTally[zone] || 0) + 1;

        if (zoneTally[zone] > dominantZoneCount) {
            dominantZoneCount = zoneTally[zone];
            dominantZone = zone;
        }
    }

    const handshapeTally = {};
    let dominantHandshape = "unknown";
    let dominantHandshapeCount = 0;

    for (const key of routeHistoryBuffer) {
        const sepIdx = key.lastIndexOf("_");
        const zone = sepIdx === -1 ? "neutral" : key.slice(sepIdx + 1);
        if (zone !== dominantZone) continue;

        const handshape = sepIdx === -1 ? key : key.slice(0, sepIdx);
        handshapeTally[handshape] = (handshapeTally[handshape] || 0) + 1;

        if (handshapeTally[handshape] > dominantHandshapeCount) {
            dominantHandshapeCount = handshapeTally[handshape];
            dominantHandshape = handshape;
        }
    }

    routeToSubModel(dominantHandshape, dominantZone);
}

function modelAcceptsCurrentBuffer(model, metadata = null) {
    if (metadata?.adapter === 'wlasl_rgb_video') {
        return model === wlaslModel && wlaslFrameBuffer.length > 0;
    }
    if (metadata?.adapter === 'ksl_legacy_hands_v1') {
        return legacyKslHandBuffer.length >= (metadata.frameCount || LEGACY_REQUIRED_FRAMES);
    }
    const expectedFrameSize = model?.inputs?.[0]?.shape?.[2];
    const actualFrameSize = motionBuffer[0]?.length;
    if (!expectedFrameSize || !actualFrameSize) return false;
    if (expectedFrameSize === actualFrameSize) return true;

    const isKslGraphModel = expectedFrameSize === KSL_FEATURE_SCHEMA.modelFrameWidth
        || metadata?.schemaVersion === KSL_FEATURE_SCHEMA.id;
    return isKslGraphModel
        && (currentTrackingMode === "csl_word" || currentTrackingMode === "asl_word")
        && actualFrameSize === 225;
}

// the main loop — runs every animation frame while the camera is on
async function processVideoFrames() {
    if (!isDetecting) return;

    if (webcamVideo.videoWidth === 0) {
        window.requestAnimationFrame(processVideoFrames);
        return;
    }

    let currentTimeMs = performance.now();

    if (webcamVideo.currentTime !== previousVideoTime) {
        previousVideoTime = webcamVideo.currentTime;
        resizeCanvas();

        if (!handTracker) {
            window.requestAnimationFrame(processVideoFrames);
            return;
        }

        const detectedHands = handTracker.detectForVideo(webcamVideo, currentTimeMs);
        latestKslStaticHands = detectedHands.landmarks || [];
        let detectedPose = null;

        if ((currentTrackingMode === "asl_word" || currentTrackingMode === "csl_word") && poseTracker) {
            detectedPose = poseTracker.detectForVideo(webcamVideo, currentTimeMs);
        }

        renderLandmarks(detectedPose, detectedHands);
        latestHandSignature = buildHandSignature(detectedHands);
        const handsWereVisible = latestHandsVisible;
        latestHandsVisible = hasVisibleHandLandmarks(detectedHands);

        if (currentTrackingMode === "asl_word" && wlaslBackendReady) {
            // The backend consumes the camera clip directly; do not also run
            // the legacy landmark word model or let its idle state overwrite UI.
            // Keep the recorder continuous, but discard clips that do not
            // contain enough hand-tracked frames before calling WLASL.
            syncCameraCaptureAppearance();
            if (!wlaslRecorder) startWlaslCaptureLoop();
            if (latestHandsVisible && wlaslRecorder?.state === 'recording') {
                wlaslTrackedHandFrames++;
            }
            if (!latestHandsVisible) {
                if (captureCountdownTimer) cancelCaptureCountdown();
                rawExtractionDisplay.innerText = "Waiting for hands…";
                showWaitingForInputGuide();
            } else {
                if (!handsWereVisible) startCaptureCountdown();
                if (!captureCountdownTimer) updateWlaslCaptureProgress(currentTimeMs);
            }
        } else if (staticAwaitingRelease) {
            updateStaticRelease(detectedHands, currentTimeMs);
        } else if (dynamicLetterAwaitingRelease) {
            updateDynamicLetterRelease(detectedHands, currentTimeMs);
        } else if (wordAwaitingRelease) {
            updateWordRelease(detectedHands, currentTimeMs);
        } else if (!captureIsActive) {
            if (wlaslRequestInFlight) {
                updateWlaslCaptureProgress(currentTimeMs);
            } else if (currentTrackingMode === "asl_word" && wlaslBackendReady) {
                startCaptureCountdown();
            } else if (currentTrackingMode === "asl_letter" && letterSessionCountdownComplete) {
                if (latestHandsVisible) {
                    setCaptureActive(true);
                    showReadyCaptureGuide();
                } else {
                    showWaitingForInputGuide();
                }
            } else if (latestHandsVisible) {
                startCaptureCountdown();
            } else {
                if (captureCountdownTimer) cancelCaptureCountdown();
                if (currentTrackingMode !== "asl_letter") {
                    rawExtractionDisplay.innerText = "Waiting for hands…";
                }
                showWaitingForInputGuide();
            }
        } else if (latestHandsVisible) {
            analyzeMovement(detectedHands, detectedPose, currentTimeMs);
        } else {
            resetInactivity();
        }
    }

    window.requestAnimationFrame(processVideoFrames);
}

function resizeCanvas() {
    if (overlayCanvasElement.width !== webcamVideo.videoWidth) {
        overlayCanvasElement.width = webcamVideo.videoWidth;
        overlayCanvasElement.height = webcamVideo.videoHeight;
    }
    canvasContext.clearRect(0, 0, overlayCanvasElement.width, overlayCanvasElement.height);
}

// draw the skeleton lines on the canvas overlay
function renderLandmarks(poseData, handData) {
    if (!shouldDrawSkeletons) return;


    if (poseData?.landmarks?.length > 0) {
        const body = poseData.landmarks[0];
        canvasContext.strokeStyle = "lime";
        canvasContext.lineWidth = 2;

        for (const link of POSE_CONNECTIONS) {
            const x1 = body[link[0]].x * overlayCanvasElement.width;
            const y1 = body[link[0]].y * overlayCanvasElement.height;
            const x2 = body[link[1]].x * overlayCanvasElement.width;
            const y2 = body[link[1]].y * overlayCanvasElement.height;

            canvasContext.beginPath();
            canvasContext.moveTo(x1, y1);
            canvasContext.lineTo(x2, y2);
            canvasContext.stroke();
        }
    }


    if (handData?.landmarks) {
        canvasContext.strokeStyle = "cyan";
        canvasContext.lineWidth = 2;

        for (const hand of handData.landmarks) {
            for (const link of HAND_CONNECTIONS) {
                const x1 = hand[link[0]].x * overlayCanvasElement.width;
                const y1 = hand[link[0]].y * overlayCanvasElement.height;
                const x2 = hand[link[1]].x * overlayCanvasElement.width;
                const y2 = hand[link[1]].y * overlayCanvasElement.height;

                canvasContext.beginPath();
                canvasContext.moveTo(x1, y1);
                canvasContext.lineTo(x2, y2);
                canvasContext.stroke();
            }
        }
    }
}

// figure out what kind of sign is happening and route to the right detector
function analyzeMovement(handData, poseData, timestamp = performance.now()) {
    let currentFrame = [];
    const primaryHand = handData.landmarks?.length > 0 ? handData.landmarks[0] : null;
    const assignedWordHands = currentTrackingMode === "asl_letter"
        ? null
        : assignDetectedHands(handData, poseData);
    const routingHand = (currentTrackingMode === "asl_word" || currentTrackingMode === "csl_word")
        ? selectRoutingHand(handData)
        : null;

    if (currentTrackingMode === "asl_letter") {
        if (primaryHand) {
            for (let i = 0; i < primaryHand.length; i++) {
                currentFrame.push(primaryHand[i].x, primaryHand[i].y, primaryHand[i].z);
            }
        }
    } else {
        // Keep raw coordinates in the live buffer. Each model adapter applies
        // its own matching legacy or KSL-v2 preprocessing immediately before inference.
        currentFrame = buildHolisticFrame(poseData, handData, assignedWordHands);
    }

    if (currentFrame.length > 0) {
        motionBuffer.push(currentFrame);
        motionTimestampBuffer.push(timestamp);
        if (currentTrackingMode !== "asl_letter") {
            handCropBuffer.push(extractTwoHandCropFrame(handData, assignedWordHands));
            while (
                motionBuffer.length > 2
                && timestamp - motionTimestampBuffer[0] > 2500
            ) {
                motionBuffer.shift();
                motionTimestampBuffer.shift();
                handCropBuffer.shift();
            }
        } else if (motionBuffer.length > LEGACY_REQUIRED_FRAMES) {
            motionBuffer.shift();
            motionTimestampBuffer.shift();
        }
    }

    if (currentTrackingMode === "csl_word") {
        legacyKslHandBuffer.push(buildLegacyKslHandFrame(handData));
        while (legacyKslHandBuffer.length > LEGACY_REQUIRED_FRAMES) legacyKslHandBuffer.shift();
    }

    if (currentTrackingMode === "asl_word" || currentTrackingMode === "csl_word") {
        if (routingHand) {
            const handshape = classifyHandshape(routingHand.hand);
            const zone = classifyZone(poseData, routingHand.hand);
            routeHistoryBuffer.push(`${handshape}_${zone}`);

            wristVelocityBuffer.push({
                x: routingHand.hand[0].x,
                y: routingHand.hand[0].y,
                side: routingHand.side
            });
            if (wristVelocityBuffer.length > 10) wristVelocityBuffer.shift();
        } else {
            routeHistoryBuffer.push("unknown_neutral");
        }

        if (routeHistoryBuffer.length > KSL_REQUIRED_FRAMES) routeHistoryBuffer.shift();
        updateRoutingDebugUI();
    }

    if (currentTrackingMode !== "asl_letter" && motionTimestampBuffer.length > 1) {
        const elapsedMs = motionTimestampBuffer.at(-1) - motionTimestampBuffer[0];
        showWordCaptureProgress(elapsedMs, 1900);
    }

    // Static alphabet inference needs only the current hand. Keep collecting a
    // 30-frame window in parallel solely for the dynamic J/Z recognizer.
    if ((currentTrackingMode === "asl_letter" && primaryHand)
        || (currentTrackingMode !== "asl_letter" && motionBuffer.length >= 2)) {
        if (currentTrackingMode === "asl_letter") {
            const hand = primaryHand;
            const hasMotionWindow = motionBuffer.length >= LEGACY_REQUIRED_FRAMES;
            const recentStart = motionBuffer[Math.max(0, motionBuffer.length - 8)];
            const recentEnd = motionBuffer.at(-1);
            const recentMovement = Math.hypot(
                recentEnd[0] - recentStart[0],
                recentEnd[1] - recentStart[1]
            );
            const recentFrames = motionBuffer.slice(-8);
            const emergingLetterMotion = hasEmergingAslLetterMotion(recentFrames);
            const dynamicLetterIntent = getAslDynamicLetterIntent(motionBuffer);
            const intentionalLetterMotion = hasMotionWindow
                && (dynamicLetterIntent.J || dynamicLetterIntent.Z);
            if (recentMovement > 0.035 || emergingLetterMotion || intentionalLetterMotion) {
                resetStaticHoldTimer();
                showDynamicLetterPending();
                if (intentionalLetterMotion) inferDynamicGesture();
            } else if (hand) {
                inferStaticGesture(hand, timestamp);
            }
        } else {
            inferDynamicGesture();
        }
    }

    if (currentTrackingMode === "asl_word" && wlaslModel) {
        captureWlaslFrame(timestamp);
    }
}

function captureWlaslFrame(timestamp) {
    const size = wlaslConfig.imageSize;
    wlaslCaptureCanvas.width = size;
    wlaslCaptureCanvas.height = size;
    const sourceSize = Math.min(webcamVideo.videoWidth, webcamVideo.videoHeight);
    const sourceX = (webcamVideo.videoWidth - sourceSize) / 2;
    const sourceY = (webcamVideo.videoHeight - sourceSize) / 2;
    wlaslCaptureContext.drawImage(webcamVideo, sourceX, sourceY, sourceSize, sourceSize, 0, 0, size, size);
    wlaslFrameBuffer.push({ timestamp, rgba: wlaslCaptureContext.getImageData(0, 0, size, size).data });
    const cutoff = timestamp - wlaslConfig.clipDurationMs;
    while (wlaslFrameBuffer.length > 1 && wlaslFrameBuffer[0].timestamp < cutoff) wlaslFrameBuffer.shift();
}

function legacyHolisticSequence(frames, targetLength) {
    const prepared = frames.slice(-targetLength).map(frame => frame.slice());
    for (let frameIndex = 1; frameIndex < prepared.length; frameIndex++) {
        for (const [start, end] of [[99, 162], [162, 225]]) {
            if (prepared[frameIndex][start] === 0 && prepared[frameIndex][start + 1] === 0) {
                for (let valueIndex = start; valueIndex < end; valueIndex++) {
                    prepared[frameIndex][valueIndex] = prepared[frameIndex - 1][valueIndex];
                }
            }
        }
    }
    return prepared.map(centerFrameData);
}

function reduceHolisticToKslRaw(frame) {
    const reduced = [];
    for (const poseIndex of KSL_FEATURE_SCHEMA.poseIndices) {
        const offset = poseIndex * 3;
        reduced.push(frame[offset], frame[offset + 1], frame[offset + 2]);
    }
    reduced.push(...frame.slice(99, 162), ...frame.slice(162, 225));
    return reduced;
}

function prepareInferenceInputs(model, metadata) {
    if (metadata?.adapter === 'wlasl_rgb_video') return prepareWlaslInferenceInputs(metadata);
    const sequenceLength = model?.inputs?.[0]?.shape?.[1] || metadata?.frameCount || LEGACY_REQUIRED_FRAMES;
    const frameWidth = model?.inputs?.[0]?.shape?.[2];

    let landmarkFrames;
    let visualFrames = null;
    if (metadata?.adapter === 'ksl_legacy_hands_v1' && frameWidth === 126) {
        landmarkFrames = prepareLegacyKslSequence(legacyKslHandBuffer, sequenceLength);
        if (!landmarkFrames) return null;
    } else if (frameWidth === KSL_FEATURE_SCHEMA.modelFrameWidth) {
        const latestTimestamp = motionTimestampBuffer.at(-1);
        const earliestTimestamp = motionTimestampBuffer[0];
        if (!Number.isFinite(latestTimestamp) || latestTimestamp - earliestTimestamp < 1900) return null;
        const windowStart = latestTimestamp - 2000;
        let startIndex = motionTimestampBuffer.findIndex(timestamp => timestamp >= windowStart);
        if (startIndex < 0) startIndex = 0;
        const rawFrames = motionBuffer.slice(startIndex).map(reduceHolisticToKslRaw);
        landmarkFrames = prepareKslSequence(rawFrames, { targetFrameCount: sequenceLength });
        visualFrames = resampleFlatFrames(handCropBuffer.slice(startIndex), sequenceLength);
    } else if (frameWidth === 225) {
        if (motionBuffer.length < sequenceLength) return null;
        landmarkFrames = legacyHolisticSequence(motionBuffer, sequenceLength);
    } else if (frameWidth === 63) {
        if (motionBuffer.length < sequenceLength) return null;
        landmarkFrames = motionBuffer.slice(-sequenceLength);
    } else {
        console.error(`Unsupported model input width: ${frameWidth}.`);
        return null;
    }

    const landmarkTensor = tf.tensor3d([landmarkFrames]);
    const tensors = [landmarkTensor];
    if ((model.inputs?.length || 0) > 1) {
        if (!visualFrames?.length) {
            landmarkTensor.dispose();
            return null;
        }
        const cropValues = visualFrames.flat();
        const cropTensor = tf.tensor5d(cropValues, [
            1,
            sequenceLength,
            HAND_CROP_HEIGHT,
            HAND_CROP_WIDTH,
            1
        ]);
        tensors.push(cropTensor);
    }

    return {
        inputs: tensors.length === 1 ? tensors[0] : tensors,
        tensors
    };
}

function prepareWlaslInferenceInputs(config) {
    if (wlaslFrameBuffer.length < 2) return null;
    const duration = wlaslFrameBuffer.at(-1).timestamp - wlaslFrameBuffer[0].timestamp;
    if (duration < config.clipDurationMs * 0.8) return null;
    const indices = uniformSampleIndices(wlaslFrameBuffer.length, config.frameCount);
    const rgb = new Uint8Array(config.frameCount * config.imageSize * config.imageSize * 3);
    let target = 0;
    for (const index of indices) {
        const rgba = wlaslFrameBuffer[index].rgba;
        for (let source = 0; source < rgba.length; source += 4) {
            rgb[target++] = rgba[source];
            rgb[target++] = rgba[source + 1];
            rgb[target++] = rgba[source + 2];
        }
    }
    const values = normalizeRgbBytes(rgb, config.normalization);
    const tensor = tf.tensor5d(values, [1, config.frameCount, config.imageSize, config.imageSize, 3]);
    return { inputs: tensor, tensors: [tensor] };
}

function resampleFlatFrames(frames, targetLength) {
    if (!frames.length) return [];
    if (frames.length === targetLength) return frames.map(frame => frame.slice());
    if (frames.length === 1) {
        return Array.from({ length: targetLength }, () => frames[0].slice());
    }
    return Array.from({ length: targetLength }, (_, targetIndex) => {
        const sourcePosition = targetIndex * (frames.length - 1) / (targetLength - 1);
        return frames[Math.round(sourcePosition)].slice();
    });
}

// run the tensorflow model on the buffered frames to recognize a motion sign
function inferDynamicGesture() {
    resetStaticHoldTimer();

    if (currentTrackingMode === "asl_word" && wlaslModel) {
        const now = performance.now();
        if (now - lastWlaslInferenceAt < 500) return;
        lastWlaslInferenceAt = now;
    }

    if (currentTrackingMode === "csl_word") {
        resolveDominantRoute();
    }

    // Routed sub-models belong to the Cambodian/Khmer vocabulary only. Never
    // allow them to leak Khmer labels into ASL Words mode.
    const baseUsesKslGraph = activeAIModel?.inputs?.[0]?.shape?.[2]
        === KSL_FEATURE_SCHEMA.modelFrameWidth;
    // A new unified graph model scales across the whole KSL vocabulary. Keep
    // the old handshape-zone sub-models only as a fallback when that model is absent.
    const useRoutedModel = currentTrackingMode === "csl_word"
        && activeSubModel
        && !baseUsesKslGraph;
    const modelToUse = useRoutedModel
        ? activeSubModel
        : (modelAcceptsCurrentBuffer(activeAIModel, activeModelMetadata) ? activeAIModel : null);
    const vocabularyToUse = useRoutedModel ? activeSubVocabulary : activeVocabulary;
    const metadataToUse = useRoutedModel ? activeSubMetadata : activeModelMetadata;

    if (!modelToUse || !vocabularyToUse?.length) {
        if (currentTrackingMode === "csl_word" && !cslModel && kslStaticModel && latestKslStaticHands.length) {
            inferStaticKslFallback();
            return;
        }
        if (currentTrackingMode === "asl_word" && !aslWordModel && !wlaslBackendReady) {
            rawExtractionDisplay.innerText = "Start the WLASL service on port 8001";
        }
        showDynamicIdle(0, true);
        return;
    }

    const preparedInputs = prepareInferenceInputs(modelToUse, metadataToUse);
    if (!preparedInputs) {
        showDynamicIdle(0, true);
        return;
    }
    let aiPrediction;

    try {
        aiPrediction = modelToUse.predict(preparedInputs.inputs);
    } catch (err) {
        preparedInputs.tensors.forEach(tensor => tensor.dispose());
        console.error("Model inference failed:", err);
        return;
    }

    const probabilities = Array.from(aiPrediction.dataSync());
    let mappedResults = probabilities.map((conf, idx) => ({ word: vocabularyToUse[idx], conf: conf }));
    mappedResults.sort((a, b) => b.conf - a.conf);

    const bestMatch = mappedResults[0];
    const predictedSign = bestMatch.word;
    const confidenceLevel = bestMatch.conf;
    const cleanedSign = predictedSign.trim().toLowerCase();

    let isUserIdle = ["idle", "none", "", "still"].includes(cleanedSign);
    let aslLetterMotionIntent = null;
    let predictedMotionIsSupported = true;

    if (currentTrackingMode === "asl_letter") {
        aslLetterMotionIntent = getAslDynamicLetterIntent(motionBuffer);
        predictedMotionIsSupported = cleanedSign === "j"
            ? aslLetterMotionIntent.J
            : (cleanedSign === "z" ? aslLetterMotionIntent.Z : false);
        if (!aslLetterMotionIntent.J && !aslLetterMotionIntent.Z) isUserIdle = true;
    } else if (metadataToUse?.adapter !== 'ksl_legacy_hands_v1') {
        let totalMovement = 0;
        let jointCount = 0;
        const firstFrame = motionBuffer[0];
        const lastFrame = motionBuffer[motionBuffer.length - 1];

        for (const region of [[99, 162], [162, 225]]) {
            for (let i = region[0]; i < region[1]; i += 3) {
                const oldX = firstFrame[i];
                const oldY = firstFrame[i + 1];
                const newX = lastFrame[i];
                const newY = lastFrame[i + 1];

                if ((oldX !== 0 || oldY !== 0) && (newX !== 0 || newY !== 0)) {
                    totalMovement += Math.sqrt(Math.pow(newX - oldX, 2) + Math.pow(newY - oldY, 2));
                    jointCount++;
                }
            }
        }

        const avgJointMovement = jointCount > 0 ? totalMovement / jointCount : 0;
        if (avgJointMovement < 0.003) {
            isUserIdle = true;
        }
    }

    const voteWord = currentTrackingMode === "asl_letter" && !predictedMotionIsSupported
        ? "idle"
        : predictedSign;
    predictionVotes.push({ word: voteWord, conf: confidenceLevel });
    if (predictionVotes.length > VOTE_WINDOW) predictionVotes.shift();

    const voteTally = {};
    let votedSign = predictedSign;
    let bestVoteScore = 0;

    for (const vote of predictionVotes) {
        if (!voteTally[vote.word]) voteTally[vote.word] = { count: 0, confSum: 0 };
        voteTally[vote.word].count++;
        voteTally[vote.word].confSum += vote.conf;
    }

    for (const [word, data] of Object.entries(voteTally)) {
        const score = data.count * (data.confSum / data.count);
        if (score > bestVoteScore) {
            bestVoteScore = score;
            votedSign = word;
        }
    }

    const idleWords = ["idle", "none", "", "still", "មិនកម្រើក"];
    const votedLetter = votedSign.trim().toUpperCase();
    const votedMotionIsSupported = currentTrackingMode !== "asl_letter"
        || (votedLetter === "J" && aslLetterMotionIntent?.J)
        || (votedLetter === "Z" && aslLetterMotionIntent?.Z);
    const isVotedIdle = isUserIdle
        || !votedMotionIsSupported
        || idleWords.includes(votedSign.trim().toLowerCase());
    const threshold = currentTrackingMode === "asl_word" && wlaslModel
        ? wlaslConfig.confidenceThreshold
        : (Number.isFinite(metadataToUse?.confidenceThreshold)
            ? metadataToUse.confidenceThreshold
            : ((currentTrackingMode === "asl_word" || currentTrackingMode === "csl_word")
                ? getDynamicThreshold()
                : CONFIDENCE_THRESHOLD));

    if (!isVotedIdle && confidenceLevel > threshold) {
        showDynamicMatch(votedSign, confidenceLevel);

        if (votedSign === lastDynamicGuess) {
            dynamicSignStreak++;
        } else {
            lastDynamicGuess = votedSign;
            dynamicSignStreak = 1;
        }

        const requiredHits = (currentTrackingMode === "asl_letter")
            ? DYNAMIC_LETTER_REQUIRED_HITS
            : 4;

        if (dynamicSignStreak >= requiredHits) {
            const topChoices = mappedResults
                .filter(res => !idleWords.includes(res.word.trim().toLowerCase()))
                .filter(res => currentTrackingMode !== "asl_letter"
                    || Boolean(aslLetterMotionIntent?.[res.word.trim().toUpperCase()]))
                .slice(0, 3);

            if (topChoices.length > 0) {
                acceptDynamicWord(topChoices);
            }

            dynamicSignStreak = 0;
            motionBuffer = [];
            clearHandCache({ preserveReleaseGate: true });
            if (!dynamicLetterAwaitingRelease) showDynamicIdle(0, true);
        }
    } else {
        showDynamicIdle(confidenceLevel, isVotedIdle);
        if (currentTrackingMode === "asl_letter") showDynamicLetterPending();
    }

    preparedInputs.tensors.forEach(tensor => tensor.dispose());
    aiPrediction.dispose();
}

function inferStaticKslFallback() {
    const ranked = predictStaticKsl(kslStaticModel, latestKslStaticHands);
    const bestMatch = ranked[0];
    if (!bestMatch || bestMatch.confidence < KSL_STATIC_CONFIDENCE_THRESHOLD) {
        showDynamicIdle(bestMatch?.confidence || 0, false);
        return;
    }

    showDynamicMatch(bestMatch.word, bestMatch.confidence);
    if (bestMatch.word === lastDynamicGuess) {
        dynamicSignStreak++;
    } else {
        lastDynamicGuess = bestMatch.word;
        dynamicSignStreak = 1;
    }

    if (dynamicSignStreak < 4) return;
    acceptDynamicWord(ranked.slice(0, 3).map(({ word, confidence }) => ({
        word,
        conf: confidence
    })));
    dynamicSignStreak = 0;
    motionBuffer = [];
    clearHandCache({ preserveReleaseGate: true });
    showDynamicIdle(0, true);
}

function updateLetterConsensus(history, letter, score, maxFrames = 5) {
    const next = Array.isArray(history) ? history.slice() : [];
    if (letter) next.push({ letter, score: Number(score) || 0 });
    while (next.length > maxFrames) next.shift();
    const totals = new Map();
    for (const vote of next) {
        const current = totals.get(vote.letter) || { count: 0, score: 0 };
        current.count++;
        current.score += vote.score;
        totals.set(vote.letter, current);
    }
    const ranked = [...totals.entries()].sort((a, b) =>
        b[1].count - a[1].count || b[1].score - a[1].score
    );
    const [winner, winnerData] = ranked[0] || [null, { count: 0, score: 0 }];
    return {
        history: next,
        letter: winner,
        score: winnerData.count ? winnerData.score / winnerData.count : 0,
        stable: next.length >= 3 && winnerData.count >= 3
    };
}

// Fingerpose runs locally in the browser. A short majority vote prevents one
// noisy MediaPipe frame from changing the displayed/accepted letter.
function inferStaticGesture(hand, timestamp = performance.now()) {
    dynamicSignStreak = 0;
    const fpGuess = evaluateFingerPose(hand);
    const predictedLetter = fpGuess ? refineGeometricEdges(fpGuess.name, hand) : null;

    if (fpGuess && predictedLetter) {
        const consensus = updateLetterConsensus(staticLetterVotes, predictedLetter, fpGuess.score);
        staticLetterVotes = consensus.history;
        const adjustedLetter = consensus.letter || predictedLetter;
        if (!consensus.stable) {
            staticHoldElapsedMs = 0;
            staticHoldLastTickAt = timestamp;
            showStaticCandidatePending(consensus.score || fpGuess.score, 0);
            return;
        }

        if (adjustedLetter === lastStaticGuess) {
            staticSignStreak++;
            staticHoldElapsedMs = Math.min(
                STATIC_HOLD_DURATION_MS,
                staticHoldElapsedMs + timerDelta(timestamp)
            );
        } else {
            lastStaticGuess = adjustedLetter;
            staticSignStreak = 1;
            staticHoldElapsedMs = 0;
            staticHoldLastTickAt = timestamp;
        }

        const holdProgress = staticHoldElapsedMs / STATIC_HOLD_DURATION_MS;
        if (holdProgress < STATIC_PREVIEW_REVEAL_PROGRESS) {
            showStaticCandidatePending(consensus.score || fpGuess.score, holdProgress);
        } else {
            showStaticMatch(adjustedLetter, consensus.score || fpGuess.score, holdProgress);
        }

        if (holdProgress >= 1) {
            acceptStaticLetter(adjustedLetter);
        }
    } else {
        staticLetterVotes = [];
        resetStaticUI(timestamp);
    }
}

// Fingerpose gets confused between similar hand shapes, so retain the original
// browser-only landmark geometry corrections used before the RGB model.
function refineGeometricEdges(currentGuess, hand) {
    let corrected = currentGuess;

    if (['Y', 'I'].includes(corrected)) {
        const leftHanded = hand[5].x > hand[17].x;
        const thumbSideOffset = leftHanded ? (hand[4].x - hand[5].x) : (hand[5].x - hand[4].x);
        corrected = thumbSideOffset > 0.045 ? 'Y' : 'I';
    }

    if (['U', 'V', 'R', 'P', 'H', 'K'].includes(corrected)) {
        const dx = Math.abs(hand[8].x - hand[5].x);
        const dy = Math.abs(hand[8].y - hand[5].y);
        if (hasCrossedIndexMiddle(hand)) {
            corrected = 'R';
        } else if (dx > dy * 1.2) {
            corrected = 'H';
        } else if (hand[8].y > hand[5].y) {
            corrected = 'P';
        } else {
            const fingerGap = calculateDistance(hand[8], hand[12]);
            const baseGap = calculateDistance(hand[5], hand[9]);
            if (fingerGap < baseGap * 1.4) {
                corrected = 'U';
            } else {
                const leftHanded = hand[5].x > hand[17].x;
                let tX = hand[4].x, iX = hand[5].x, mX = hand[9].x;
                if (leftHanded) { tX = -tX; iX = -iX; mX = -mX; }
                corrected = hand[4].y < hand[9].y - 0.01 && tX > iX && tX < mX ? 'K' : 'V';
            }
        }
    }

    if (['D', 'Q', 'L', 'G'].includes(corrected)) {
        const dx = Math.abs(hand[8].x - hand[5].x);
        const dy = Math.abs(hand[8].y - hand[5].y);
        if (dx > dy * 1.2) {
            corrected = 'G';
        } else if (hand[8].y > hand[5].y) {
            corrected = 'Q';
        } else {
            const width = calculateDistance(hand[5], hand[17]);
            const thumbExtension = calculateDistance(hand[4], hand[5]);
            corrected = thumbExtension > width * 0.7 ? 'L' : 'D';
        }
    }

    if (['C', 'O', 'F'].includes(corrected)) {
        const middleIsUp = hand[12].y < hand[10].y;
        if (middleIsUp) {
            corrected = 'F';
        } else {
            const pinchGap = calculateDistance(hand[8], hand[4]);
            const knuckleWidth = calculateDistance(hand[5], hand[6]);
            corrected = pinchGap < knuckleWidth ? 'O' : 'C';
        }
    }

    if (['A', 'E', 'S', 'T', 'M', 'N'].includes(corrected)) {
        corrected = classifyClosedThumbLetter(hand);
    }

    return corrected;
}

function showDynamicMatch(word, confidence) {
    const percent = Math.round(confidence * 100);
    confidenceProgressBar.style.width = percent + "%";
    confidenceProgressBar.style.background = "lime";
    confidencePctDisplay.innerText = percent + "%";
    guessLabelDisplay.innerText = "Motion: " + word;
}

function showDynamicLetterPending() {
    confidenceProgressBar.style.width = "0%";
    confidenceProgressBar.style.background = "gray";
    confidencePctDisplay.innerText = "—";
    guessLabelDisplay.innerText = "Motion: Detecting…";
    setCaptureGuide(
        "capturing",
        "Following the movement",
        "Complete the motion, then relax your hand.",
        0
    );
}

function showDynamicIdle(confidence, isUserIdle) {
    dynamicSignStreak = 0;
    lastDynamicGuess = "";

    const percent = Math.round(confidence * 100);
    confidenceProgressBar.style.width = isUserIdle ? percent + "%" : "0%";
    confidenceProgressBar.style.background = "gray";
    guessLabelDisplay.innerText = isUserIdle ? "Motion: Idle" : "Swaying...";
    confidencePctDisplay.innerText = isUserIdle ? percent + "%" : "0%";
}

function showStaticMatch(letter, score, holdProgress) {
    // Score out of 10 from fingerpose
    const percent = Math.round((score / 10) * 100);
    const holdPercent = Math.round(holdProgress * 100);
    confidenceProgressBar.style.width = percent + "%";
    confidenceProgressBar.style.background = "cyan";
    guessLabelDisplay.innerText = `Pose: ${letter} (Hold: ${holdPercent}%)`;
    confidencePctDisplay.innerText = percent + "%";
    setCaptureGuide(
        "holding",
        `Hold ${letter} steady`,
        "Keep the handshape still until the bar is full.",
        holdProgress
    );
}

function showStaticCandidatePending(score, holdProgress) {
    const percent = Math.round((score / 10) * 100);
    confidenceProgressBar.style.width = percent + "%";
    confidenceProgressBar.style.background = "cyan";
    confidencePctDisplay.innerText = percent + "%";
    guessLabelDisplay.innerText = "Pose: Checking…";
    setCaptureGuide(
        "holding",
        "Checking handshape",
        "Hold still briefly while Dexter confirms the letter.",
        holdProgress
    );
}

function resetStaticUI(timestamp = performance.now()) {
    const remainingProgress = decayStaticHold(timestamp);
    staticSignStreak = 0;
    confidenceProgressBar.style.width = "0%";
    guessLabelDisplay.innerText = "Pose: None";
    confidencePctDisplay.innerText = "0%";
    if (!staticAwaitingRelease) {
        rawExtractionDisplay.innerText = "...";
    }
    if (remainingProgress > 0 && lastStaticGuess) {
        setCaptureGuide(
            "holding",
            `Steady ${lastStaticGuess}`,
            "The timer is paused briefly while Dexter finds the handshape again.",
            remainingProgress
        );
    } else {
        showWaitingForInputGuide();
    }
}

// rebuild the accumulated word string from whatever is actually in the DOM
function syncSentenceState() {
    let text = "";
    Array.from(sentenceBuilderDisplay.childNodes).forEach(node => {
        if (node.tagName === 'SELECT') {
            text += node.value + " ";
        } else if (node.tagName === 'SPAN') {
            text += node.innerText;
        } else if (node.nodeType === Node.TEXT_NODE) {
            text += node.textContent;
        }
    });
    accumulatedWord = text.replace(/\s+/g, ' ').trim();
}

function acceptStaticLetter(letter) {
    setCaptureActive(false);
    rawExtractionDisplay.innerText = letter;

    if (letter.trim().toLowerCase() !== "idle") {
        const charSpan = document.createElement('span');
        charSpan.className = "static-letter";
        charSpan.innerText = letter;
        sentenceBuilderDisplay.appendChild(charSpan);
        syncSentenceState();
    }

    staticAwaitingRelease = true;
    acceptedStaticLetter = letter;
    acceptedStaticSignature = normalizeHandShapeSignature(latestHandSignature);
    staticReleaseStartedAt = null;
    resetStaticHoldTimer();
    setCaptureGuide(
        "accepted",
        `${letter} accepted`,
        "Release or change your hand to continue.",
        1
    );
}

function acceptDynamicWord(
    options,
    releaseSignature = latestHandSignature,
    { lockRecognition = true } = {}
) {
    const primaryWord = options[0].word;
    rawExtractionDisplay.innerText = primaryWord;

    const dropdown = document.createElement('select');
    dropdown.className = "ai-correction-dropdown";

    // make it look like regular text until you click on it
    Object.assign(dropdown.style, {
        border: "none",
        background: "transparent",
        fontFamily: "inherit",
        fontSize: "inherit",
        color: "var(--text-color, inherit)",
        cursor: "pointer",
        outline: "none",
        textDecoration: "underline dashed gray"
    });


    options.forEach(choice => {
        const optionEl = document.createElement('option');
        optionEl.value = choice.word;
        optionEl.dataset.conf = Math.round(choice.conf * 100);
        optionEl.innerText = choice.word;
        dropdown.appendChild(optionEl);
    });

    // show confidence % when the dropdown is open so you can pick the right one
    dropdown.addEventListener('focus', function () {
        Array.from(this.options).forEach(opt => opt.innerText = `${opt.value} (${opt.dataset.conf}%)`);
    });

    dropdown.addEventListener('blur', function () {
        Array.from(this.options).forEach(opt => opt.innerText = opt.value);
    });

    dropdown.addEventListener('change', function () {
        Array.from(this.options).forEach(opt => opt.innerText = opt.value);
        this.blur();
        syncSentenceState();
    });

    sentenceBuilderDisplay.appendChild(dropdown);
    sentenceBuilderDisplay.appendChild(document.createTextNode(" "));

    syncSentenceState();
    if (lockRecognition) {
        if (currentTrackingMode === "asl_letter") {
            lockDynamicLetterRecognition(primaryWord, releaseSignature);
        }
        else lockWordRecognition(primaryWord, releaseSignature);
    }
}

function resetInactivity() {
    const timestamp = performance.now();
    if (dynamicLetterAwaitingRelease) {
        updateDynamicLetterRelease({ landmarks: [] }, timestamp);
        if (dynamicLetterAwaitingRelease) return;
    }
    if (wordAwaitingRelease) {
        updateWordRelease({ landmarks: [] }, timestamp);
        if (wordAwaitingRelease) return;
    }
    if (staticAwaitingRelease) {
        updateStaticRelease({ landmarks: [] }, timestamp);
        if (staticAwaitingRelease) return;
    }

    motionBuffer = [];
    legacyKslHandBuffer = [];
    handCropBuffer = [];
    motionTimestampBuffer = [];
    wlaslFrameBuffer = [];
    lastWlaslInferenceAt = 0;
    runtimeHandSlotState = createKslHandSlotState();
    predictionVotes = [];
    routeHistoryBuffer = [];
    wristVelocityBuffer = [];
    activeSubModel = null;
    activeSubVocabulary = null;
    activeSubMetadata = null;
    dynamicSignStreak = 0;
    resetStaticHoldTimer();
    rawExtractionDisplay.innerText = (currentTrackingMode === "asl_word" || currentTrackingMode === "csl_word")
        ? "Waiting for hands…"
        : "...";
    confidenceProgressBar.style.width = "0%";

    if (currentTrackingMode === "asl_word" || currentTrackingMode === "csl_word") {
        guessLabelDisplay.innerText = "Waiting for input... [Ready]";
    } else {
        guessLabelDisplay.innerText = "None";
    }
    confidencePctDisplay.innerText = "0%";
    setCaptureActive(false);
    showWaitingForInputGuide();
}
