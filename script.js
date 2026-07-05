import { HandLandmarker, PoseLandmarker, FilesetResolver }
    from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

// skip drawing skeletons on phones — they can't handle it
const isMobileDevice = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
let shouldDrawSkeletons = !isMobileDevice;

// camera stuff
let cameraStream;
let webcamVideo = document.getElementById('webcamFeed');
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

// mediapipe hand + pose trackers
let handTracker;
let poseTracker;

// we load the base ASL model plus an optional CSL fallback and the CSL sub-model registry
let aslModel, cslModel;
let aslVocab = [], cslVocab = [];

// whichever model is selected right now
let activeAIModel;
let activeVocabulary = [];

// routed word sub-models, keyed by handshape + body zone (for example "fist_face")
let subModelRegistry = {};
let activeSubModel = null;
let activeSubVocabulary = null;
let wristVelocityBuffer = [];
let routeHistoryBuffer = [];
let routingDebug = { handshape: "unknown", zone: "neutral", routeKey: null, matched: false, velocity: 0, dynThreshold: 0.90 };

// we collect 30 frames of hand data before making a prediction
let motionBuffer = [];
let accumulatedWord = "";
let actionCooldown = 0;
let previousVideoTime = -1;
const REQUIRED_FRAMES = 30;
const COOLDOWN_DURATION_MS = 1500;
const CONFIDENCE_THRESHOLD = 0.90;

// how many frames in a row the same sign has been detected
let dynamicSignStreak = 0;
let lastDynamicGuess = "";
let staticSignStreak = 0;
let lastStaticGuess = "";
const VOTE_WINDOW = 5;
let predictionVotes = [];

// the transparent canvas sitting on top of the video for drawing skeletons
const overlayCanvasElement = document.getElementsByClassName('overlayCanvas')[0];
const canvasContext = overlayCanvasElement.getContext('2d');

// mediapipe sometimes loses a hand for a frame or two, so last-seen hands fade out
// instead of dropping to zero or freezing forever.
const GHOST_DECAY_FRAMES = 8;
let cachedLeftHand = { data: new Array(21 * 3).fill(0), age: GHOST_DECAY_FRAMES };
let cachedRightHand = { data: new Array(21 * 3).fill(0), age: GHOST_DECAY_FRAMES };

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

    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");

    handTracker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`, delegate: "GPU" },
        runningMode: "VIDEO",
        numHands: 1
    });

    poseTracker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`, delegate: "GPU" },
        runningMode: "VIDEO"
    });

    try {
        const aslResponse = await fetch('./asl-labels.json');
        aslVocab = await aslResponse.json();
        aslModel = await tf.loadLayersModel('./asl-model.json');

        activeAIModel = aslModel;
        activeVocabulary = aslVocab;
        console.log("ASL model loaded successfully.");
    } catch (error) {
        console.error("Failed to load the ASL model. Check the filenames and paths.", error);
    }

    try {
        const cslResponse = await fetch('./csl_labels.json');
        if (cslResponse.ok) {
            cslVocab = await cslResponse.json();
            cslModel = await tf.loadLayersModel('./csl_model.json');
            console.log("Flat CSL model loaded successfully.");
        } else {
            console.log("No flat CSL model labels found; CSL Words will use routed sub-models when available.");
        }
    } catch (error) {
        console.log("Flat CSL model not available; continuing with routed sub-models only.");
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
                    subModelRegistry[routeKey] = { model: subModel, labels: subLabels };
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
    // fingerpose needs pixel coordinates, not the 0-1 normalized ones mediapipe gives us
    const keypoints3D = handLandmarks.map(pt => [pt.x * overlayCanvasElement.width, pt.y * overlayCanvasElement.height, pt.z * overlayCanvasElement.width]);
    const estimation = gestureEstimator.estimate(keypoints3D, 7.0);
    if (estimation.gestures.length > 0) {
        return estimation.gestures.reduce((prev, current) => (prev.score > current.score) ? prev : current);
    }
    return null;
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
    window.clearWord();
    staticSignStreak = 0;
    dynamicSignStreak = 0;
    clearHandCache();

    const guessLabelUI = document.getElementById('guessLabel');
    const rawOutputUI = document.getElementById('liveOutput');
    const outputLangUI = document.getElementById('outputLangLabel');

    if (currentTrackingMode === "csl_word") {
        guessLabelUI.innerText = "Mode: CSL Words";
        rawOutputUI.innerText = `[CSL]`;
        if (outputLangUI) outputLangUI.innerText = "CSL";
        activeAIModel = cslModel;
        activeVocabulary = cslVocab;
        if (handTracker) await handTracker.setOptions({ numHands: 2 });

    } else if (currentTrackingMode === "asl_word") {
        guessLabelUI.innerText = `Mode: ${modeLabel}`;
        rawOutputUI.innerText = `[ASL]`;
        if (outputLangUI) outputLangUI.innerText = "English";
        activeAIModel = aslModel;
        activeVocabulary = aslVocab;
        if (handTracker) await handTracker.setOptions({ numHands: 2 });

    } else {
        guessLabelUI.innerText = "Mode: ASL Spelling";
        rawOutputUI.innerText = "...";
        if (outputLangUI) outputLangUI.innerText = "English";
        activeAIModel = aslModel;
        activeVocabulary = aslVocab;
        if (handTracker) await handTracker.setOptions({ numHands: 1 });
    }
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
    window.clearWord();
    staticSignStreak = 0; dynamicSignStreak = 0;
    clearHandCache();


    const outputLangUI = document.getElementById('outputLangLabel');

    if (currentTrackingMode === "csl_word") {
        guessLabelDisplay.innerText = "Mode: CSL Words";
        rawExtractionDisplay.innerText = "[CSL]";
        if (outputLangUI) outputLangUI.innerText = "CSL";
        activeAIModel = cslModel;
        activeVocabulary = cslVocab;
        if (handTracker) await handTracker.setOptions({ numHands: 2 });

    } else if (currentTrackingMode === "asl_word") {
        guessLabelDisplay.innerText = "Mode: ASL Words";
        rawExtractionDisplay.innerText = "[ASL]";
        if (outputLangUI) outputLangUI.innerText = "English";
        activeAIModel = aslModel;
        activeVocabulary = aslVocab;
        if (handTracker) await handTracker.setOptions({ numHands: 2 });

    } else {
        guessLabelDisplay.innerText = "Mode: ASL Spelling";
        rawExtractionDisplay.innerText = "...";
        if (outputLangUI) outputLangUI.innerText = "English";
        activeAIModel = aslModel;
        activeVocabulary = aslVocab;
        if (handTracker) await handTracker.setOptions({ numHands: 1 });
    }
}


window.startCam = async function () {
    try {
        if (!isModelsReady) {
            const originalText = cameraToggleBtn.textContent;
            cameraToggleBtn.textContent = "Loading AI Models...";
            cameraToggleBtn.disabled = true;
            await modelInitPromise;
            cameraToggleBtn.disabled = false;
            cameraToggleBtn.textContent = originalText;
        }

        if (cameraStream) window.stopCam();

        // phones get a lighter 640x480 feed, desktops get full HD
        const cameraConfig = isMobileDevice
            ? { facingMode: activeCameraFacing, width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 15 } }
            : { facingMode: activeCameraFacing, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } };

        cameraStream = await navigator.mediaDevices.getUserMedia({ video: cameraConfig });

        webcamVideo.srcObject = cameraStream;
        await webcamVideo.play();

        // only mirror the video if we're using the front-facing "selfie" camera
        const videoContainer = webcamVideo.closest('.video-container');
        if (videoContainer) {
            if (activeCameraFacing === 'user') {
                videoContainer.classList.add('mirror-mode');
            } else {
                videoContainer.classList.remove('mirror-mode');
            }
        }

        isDetecting = true;
        processVideoFrames();

        cameraToggleBtn.textContent = "Stop Camera";
        cameraToggleBtn.onclick = window.stopCam;

        if (pauseToggleBtn) {
            pauseToggleBtn.style.display = "inline-block";
            pauseToggleBtn.textContent = "Pause Camera";
            pauseToggleBtn.onclick = window.pauseCam;
        }
    } catch (err) {
        alert("Unable to access camera. Please check your permissions.");
        console.error(err);
    }
}

window.pauseCam = function () {
    webcamVideo.pause();
    isDetecting = false;

    if (pauseToggleBtn) {
        pauseToggleBtn.textContent = "Resume Camera";
        pauseToggleBtn.onclick = window.resumeCam;
    }
}

window.resumeCam = async function () {
    await webcamVideo.play();
    isDetecting = true;
    processVideoFrames();

    if (pauseToggleBtn) {
        pauseToggleBtn.textContent = "Pause Camera";
        pauseToggleBtn.onclick = window.pauseCam;
    }
}

window.stopCam = function () {
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
}

window.flipCam = async function () {
    activeCameraFacing = activeCameraFacing === "user" ? "environment" : "user";
    window.stopCam();
    await window.startCam();
}



window.clearWord = function () {
    accumulatedWord = "";
    sentenceBuilderDisplay.innerHTML = "";

    const finalResultUI = document.getElementById('translatedResult');
    if (finalResultUI) finalResultUI.innerText = "";
    rawExtractionDisplay.innerText = "...";
}

window.translateSentence = async function () {
    if (!accumulatedWord) return;

    const finalResultUI = document.getElementById('translatedResult');
    finalResultUI.innerText = "Dexter is thinking...";

    try {
        const response = await fetch('/api/translate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ aslString: accumulatedWord })
        });

        if (!response.ok) {
            console.error("Backend error:", await response.text());
            finalResultUI.innerText = "Error: Translation service failed.";
            return;
        }

        const data = await response.json();
        finalResultUI.innerText = data.translation || "Error: Empty response.";
    } catch (err) {
        console.error("Network issue:", err);
        finalResultUI.innerText = "Network Error. Please try again.";
    }
}



const calculateDistance = (p1, p2) => Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));

// stitch together pose + both hands into one flat array the model expects
function buildHolisticFrame(poseData, handData) {
    let rawPose = new Array(33 * 3).fill(0);
    let leftSeen = false;
    let rightSeen = false;


    if (poseData?.landmarks?.length > 0) {
        const poseLandmarks = poseData.landmarks[0];
        for (let i = 0; i < poseLandmarks.length; i++) {
            rawPose[i * 3] = poseLandmarks[i].x;
            rawPose[i * 3 + 1] = poseLandmarks[i].y;
            rawPose[i * 3 + 2] = poseLandmarks[i].z;
        }
    }

    // grab hands and cache them in case they vanish next frame
    if (handData?.landmarks?.length > 0) {
        for (let h = 0; h < handData.landmarks.length; h++) {
            const handPts = handData.landmarks[h];
            const side = handData.handednesses[h][0].categoryName;

            let flatHand = new Array(63);
            for (let i = 0; i < handPts.length; i++) {
                flatHand[i * 3] = handPts[i].x;
                flatHand[i * 3 + 1] = handPts[i].y;
                flatHand[i * 3 + 2] = handPts[i].z;
            }

            if (side === "Left") {
                cachedLeftHand = { data: flatHand, age: 0 };
                leftSeen = true;
            } else {
                cachedRightHand = { data: flatHand, age: 0 };
                rightSeen = true;
            }
        }
    }

    if (!leftSeen) cachedLeftHand.age = Math.min(cachedLeftHand.age + 1, GHOST_DECAY_FRAMES);
    if (!rightSeen) cachedRightHand.age = Math.min(cachedRightHand.age + 1, GHOST_DECAY_FRAMES);

    const leftDecay = 1 - cachedLeftHand.age / GHOST_DECAY_FRAMES;
    const rightDecay = 1 - cachedRightHand.age / GHOST_DECAY_FRAMES;
    const leftHand = cachedLeftHand.data.map(value => value * leftDecay);
    const rightHand = cachedRightHand.data.map(value => value * rightDecay);

    return [...rawPose, ...leftHand, ...rightHand];
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

function clearHandCache() {
    cachedLeftHand = { data: new Array(21 * 3).fill(0), age: GHOST_DECAY_FRAMES };
    cachedRightHand = { data: new Array(21 * 3).fill(0), age: GHOST_DECAY_FRAMES };
    predictionVotes = [];
    wristVelocityBuffer = [];
    routeHistoryBuffer = [];
    activeSubModel = null;
    activeSubVocabulary = null;
    routingDebug = { handshape: "unknown", zone: "neutral", routeKey: null, matched: false, velocity: 0, dynThreshold: CONFIDENCE_THRESHOLD };
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

function routeToSubModel(handshape, zone) {
    const routeKey = `${handshape}_${zone}`;
    const entry = subModelRegistry[routeKey];

    if (entry) {
        activeSubModel = entry.model;
        activeSubVocabulary = entry.labels;
        routingDebug = { ...routingDebug, handshape, zone, routeKey, matched: true };
        return routeKey;
    }

    activeSubModel = null;
    activeSubVocabulary = null;
    routingDebug = { ...routingDebug, handshape, zone, routeKey, matched: false };
    return null;
}

function getDynamicThreshold() {
    if (wristVelocityBuffer.length < 2) return CONFIDENCE_THRESHOLD;

    const recent = wristVelocityBuffer.slice(-10);
    let totalDelta = 0;
    for (let i = 1; i < recent.length; i++) {
        const dx = recent[i].x - recent[i - 1].x;
        const dy = recent[i].y - recent[i - 1].y;
        totalDelta += Math.sqrt(dx * dx + dy * dy);
    }

    const avgVelocity = totalDelta / (recent.length - 1);
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

function modelAcceptsCurrentBuffer(model) {
    const expectedFrameSize = model?.inputs?.[0]?.shape?.[2];
    const actualFrameSize = motionBuffer[0]?.length;
    return !expectedFrameSize || expectedFrameSize === actualFrameSize;
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
        let detectedPose = null;

        if ((currentTrackingMode === "asl_word" || currentTrackingMode === "csl_word") && poseTracker) {
            detectedPose = poseTracker.detectForVideo(webcamVideo, currentTimeMs);
        }

        renderLandmarks(detectedPose, detectedHands);

        if ((detectedHands.landmarks && detectedHands.landmarks.length > 0) || detectedPose) {
            analyzeMovement(detectedHands, detectedPose);
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
function analyzeMovement(handData, poseData) {
    let currentFrame = [];
    const primaryHand = handData.landmarks?.length > 0 ? handData.landmarks[0] : null;

    if (currentTrackingMode === "asl_letter") {
        if (primaryHand) {
            for (let i = 0; i < primaryHand.length; i++) {
                currentFrame.push(primaryHand[i].x, primaryHand[i].y, primaryHand[i].z);
            }
        }
    } else {
        let rawHolistic = buildHolisticFrame(poseData, handData);
        currentFrame = centerFrameData(rawHolistic);
    }

    if (currentFrame.length > 0) {
        motionBuffer.push(currentFrame);
        if (motionBuffer.length > REQUIRED_FRAMES) motionBuffer.shift();
    }

    if (currentTrackingMode === "asl_word" || currentTrackingMode === "csl_word") {
        if (primaryHand) {
            const handshape = classifyHandshape(primaryHand);
            const zone = classifyZone(poseData, primaryHand);
            routeHistoryBuffer.push(`${handshape}_${zone}`);

            wristVelocityBuffer.push({ x: primaryHand[0].x, y: primaryHand[0].y });
            if (wristVelocityBuffer.length > 10) wristVelocityBuffer.shift();
        } else {
            routeHistoryBuffer.push("unknown_neutral");
        }

        if (routeHistoryBuffer.length > REQUIRED_FRAMES) routeHistoryBuffer.shift();
        updateRoutingDebugUI();
    }

    // need a full 30-frame window before we can predict anything
    if (motionBuffer.length === REQUIRED_FRAMES) {
        if (currentTrackingMode === "asl_letter") {
            const hand = primaryHand;
            const oldWristPos = motionBuffer[0];
            const newWristPos = motionBuffer[29]; // the wrist is always the first point (x, y)

            // if the hand moved a lot, it's probably a motion sign like J or Z
            const distanceMoved = Math.sqrt(
                Math.pow(newWristPos[0] - oldWristPos[0], 2) + Math.pow(newWristPos[1] - oldWristPos[1], 2)
            );

            if (distanceMoved > 0.08) {
                inferDynamicGesture();
            } else if (hand) {
                inferStaticGesture(hand);
            }
        } else {
            inferDynamicGesture();
        }
    }
}

// run the tensorflow model on the buffered frames to recognize a motion sign
function inferDynamicGesture() {
    staticSignStreak = 0;

    if (currentTrackingMode === "asl_word" || currentTrackingMode === "csl_word") {
        resolveDominantRoute();
    }

    const modelToUse = (currentTrackingMode === "asl_word" || currentTrackingMode === "csl_word")
        ? (activeSubModel || (modelAcceptsCurrentBuffer(activeAIModel) ? activeAIModel : null))
        : activeAIModel;
    const vocabularyToUse = (currentTrackingMode === "asl_word" || currentTrackingMode === "csl_word")
        ? (activeSubModel ? activeSubVocabulary : activeVocabulary)
        : activeVocabulary;

    if (!modelToUse || !vocabularyToUse?.length) {
        showDynamicIdle(0, true);
        return;
    }

    const tensorInput = tf.tensor3d([motionBuffer]);
    let aiPrediction;

    try {
        aiPrediction = modelToUse.predict(tensorInput);
    } catch (err) {
        tensorInput.dispose();
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

    if (currentTrackingMode === "asl_letter") {
        // check how far the wrist actually moved for motion letters like J/Z
        const wristStartX = motionBuffer[0][0];
        const wristStartY = motionBuffer[0][1];
        const wristEndX = motionBuffer[29][0];
        const wristEndY = motionBuffer[29][1];
        const distTraveled = Math.sqrt(Math.pow(wristEndX - wristStartX, 2) + Math.pow(wristEndY - wristStartY, 2));

        if (distTraveled < 0.03) {
            isUserIdle = true;
        }
    } else {
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

    predictionVotes.push({ word: predictedSign, conf: confidenceLevel });
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
    const isVotedIdle = isUserIdle || idleWords.includes(votedSign.trim().toLowerCase());
    const threshold = (currentTrackingMode === "asl_word" || currentTrackingMode === "csl_word")
        ? getDynamicThreshold()
        : CONFIDENCE_THRESHOLD;

    if (!isVotedIdle && confidenceLevel > threshold) {
        showDynamicMatch(votedSign, confidenceLevel);

        if (votedSign === lastDynamicGuess) {
            dynamicSignStreak++;
        } else {
            lastDynamicGuess = votedSign;
            dynamicSignStreak = 1;
        }

        const requiredHits = (currentTrackingMode === "asl_letter")
            ? (votedSign.toUpperCase() === "J" ? 8 : 12)
            : 4;

        if (dynamicSignStreak >= requiredHits && Date.now() > actionCooldown) {
            const topChoices = mappedResults
                .filter(res => !idleWords.includes(res.word.trim().toLowerCase()))
                .slice(0, 3);

            if (topChoices.length > 0) {
                acceptDynamicWord(topChoices);
            }

            dynamicSignStreak = 0;
            motionBuffer = [];
            clearHandCache();
            showDynamicIdle(0, true);
        }
    } else {
        showDynamicIdle(confidenceLevel, isVotedIdle);
    }

    tensorInput.dispose();
    aiPrediction.dispose();
}

// for still hand shapes (like most of the alphabet), use fingerpose instead of the AI model
function inferStaticGesture(hand) {
    dynamicSignStreak = 0;
    const fpGuess = evaluateFingerPose(hand);

    if (fpGuess) {
        let adjustedLetter = refineGeometricEdges(fpGuess.name, hand);

        if (adjustedLetter === lastStaticGuess) {
            staticSignStreak++;
        } else {
            lastStaticGuess = adjustedLetter;
            staticSignStreak = 1;
        }

        showStaticMatch(adjustedLetter, fpGuess.score);

        // 3 frames in a row = confident enough to lock it in
        if (staticSignStreak >= 3 && Date.now() > actionCooldown) {
            acceptStaticLetter(adjustedLetter);
        }
    } else {
        resetStaticUI();
    }
}

// fingerpose gets confused between similar hand shapes, so we do some
// extra geometry checks to tell apart letters like U/V/R or A/S/E
function refineGeometricEdges(currentGuess, hand) {
    let corrected = currentGuess;


    if (['Y', 'I'].includes(corrected)) {
        const leftHanded = hand[5].x > hand[17].x;
        // Check how far the thumb is extended horizontally "outward"
        const thumbSideOffset = leftHanded ? (hand[4].x - hand[5].x) : (hand[5].x - hand[4].x);
        
        // Y requires the thumb to be fully extended to the side (outside the hand)
        corrected = thumbSideOffset > 0.045 ? 'Y' : 'I';
    }


    if (['U', 'V', 'R', 'P', 'H', 'K'].includes(corrected)) {
        const dx = Math.abs(hand[8].x - hand[5].x);
        const dy = Math.abs(hand[8].y - hand[5].y);

        if (dx > dy * 1.2) {
            corrected = 'H';
        } else if (hand[8].y > hand[5].y) {
            corrected = 'P';
        } else {

            const knuckleCross = hand[5].x - hand[9].x;
            const tipCross = hand[8].x - hand[12].x;
            if (knuckleCross * tipCross < 0) {
                corrected = 'R';
            } else {
                const fingerGap = calculateDistance(hand[8], hand[12]);
                const baseGap = calculateDistance(hand[5], hand[9]);

                if (fingerGap < baseGap * 1.4) {
                    corrected = 'U';
                } else {
                    const leftHanded = hand[5].x > hand[17].x;
                    let tX = hand[4].x, iX = hand[5].x, mX = hand[9].x;
                    if (leftHanded) { tX = -tX; iX = -iX; mX = -mX; }

                    // K: Thumb tip must be elevated AND horizontally between index and middle fingers
                    if (hand[4].y < hand[9].y - 0.01 && tX > iX && tX < mX) {
                        corrected = 'K';
                    } else {
                        corrected = 'V';
                    }
                }
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

    // the hardest group — these all look like a closed fist with slight thumb differences
    if (['A', 'E', 'S', 'T', 'M', 'N'].includes(corrected)) {
        const leftHanded = hand[5].x > hand[17].x;

        // Use knuckle boundaries for horizontal columns and depth
        let tX = hand[4].x, tY = hand[4].y, tZ = hand[4].z;
        let iX = hand[5].x, mX = hand[9].x, rX = hand[13].x;
        let iZ = hand[5].z, mZ = hand[9].z, rZ = hand[13].z;
        let mPIP_Y = hand[10].y;

        if (leftHanded) {
            tX = -tX; iX = -iX; mX = -mX; rX = -rX;
        }

        const imBoundary = (iX + mX) / 2;
        const mrBoundary = (mX + rX) / 2;
        const thumbSideOffset = leftHanded ? (hand[4].x - hand[5].x) : (hand[5].x - hand[4].x);

        // Average depth of the knuckles to determine what's "in front"
        const knuckleZ = (iZ + mZ + rZ) / 3;

        // 1. Identify 'E' (Thumb tip tucked low, below all curled fingers)
        const maxFingerY = Math.max(hand[8].y, hand[12].y, hand[16].y, hand[20].y);
        if (tY > maxFingerY - 0.01) {
            corrected = 'E';
        }
        // 2. Identify 'A' (Thumb on the side)
        else if (thumbSideOffset > 0.02) {
            corrected = 'A';
        }
        // 3. Identify 'S' (On Top / In Front of knuckles)
        // S is identified when the thumb tip is physically closer to the camera (smaller Z)
        // than the finger knuckles, and it's not tucked into a side column.
        else if (tZ < knuckleZ - 0.015 && tY < mPIP_Y + 0.05 && tX > iX) {
            corrected = 'S';
        }
        // 4. Distinguish T, N, M (Tucked behind or between fingers)
        else if (tX < imBoundary) {
            corrected = 'T'; // Thumb between Index and Middle
        } else if (tX < mrBoundary) {
            corrected = 'N'; // Side of Middle (Between Middle and Ring)
        } else {
            corrected = 'M'; // Between Ring and Pinky
        }
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

function showDynamicIdle(confidence, isUserIdle) {
    dynamicSignStreak = 0;
    lastDynamicGuess = "";

    const percent = Math.round(confidence * 100);
    confidenceProgressBar.style.width = isUserIdle ? percent + "%" : "0%";
    confidenceProgressBar.style.background = "gray";
    guessLabelDisplay.innerText = isUserIdle ? "Motion: Idle" : "Swaying...";
    confidencePctDisplay.innerText = isUserIdle ? percent + "%" : "0%";
}

function showStaticMatch(letter, score) {
    // Score out of 10 from fingerpose
    const percent = Math.round((score / 10) * 100);
    confidenceProgressBar.style.width = percent + "%";
    confidenceProgressBar.style.background = "cyan";
    guessLabelDisplay.innerText = `Pose: ${letter} (Streak: ${staticSignStreak}/3)`;
    confidencePctDisplay.innerText = percent + "%";
}

function resetStaticUI() {
    staticSignStreak = 0;
    confidenceProgressBar.style.width = "0%";
    guessLabelDisplay.innerText = "Pose: None";
    confidencePctDisplay.innerText = "0%";
    if (Date.now() > actionCooldown) {
        rawExtractionDisplay.innerText = "...";
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
    rawExtractionDisplay.innerText = letter;

    if (letter.trim().toLowerCase() !== "idle") {
        const charSpan = document.createElement('span');
        charSpan.className = "static-letter";
        charSpan.innerText = letter;
        sentenceBuilderDisplay.appendChild(charSpan);
        syncSentenceState();
    }

    actionCooldown = Date.now() + COOLDOWN_DURATION_MS;
    staticSignStreak = 0;
}

function acceptDynamicWord(options) {
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
    actionCooldown = Date.now() + COOLDOWN_DURATION_MS;
}

function resetInactivity() {
    motionBuffer = [];
    predictionVotes = [];
    routeHistoryBuffer = [];
    wristVelocityBuffer = [];
    activeSubModel = null;
    activeSubVocabulary = null;
    dynamicSignStreak = 0;
    staticSignStreak = 0;
    lastStaticGuess = "";

    if (Date.now() > actionCooldown) {
        rawExtractionDisplay.innerText = "...";
        confidenceProgressBar.style.width = "0%";

        if (currentTrackingMode === "asl_word" || currentTrackingMode === "csl_word") {
            guessLabelDisplay.innerText = "Waiting for input... [Ready]";
        } else {
            guessLabelDisplay.innerText = "None";
        }
        confidencePctDisplay.innerText = "0%";
    }
}
