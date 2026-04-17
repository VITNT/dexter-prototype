import { HandLandmarker, PoseLandmarker, FilesetResolver }
    from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";


// --- NEW: DEVICE DETECTOR ---
const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
let showSkeletons = !isMobile;


//camera and mode
let stream;
let videoEle = document.getElementById('videoEl');
let btnCam = document.getElementById('camBtn');
let btnPauseCam = document.getElementById('pauseCamBtn');
let curCamera = "user";
let isPredicting = false;
let curMode = "asl_letter";


//confidence bar
let extractTxt = document.getElementById('translationOutput');
let conCol = document.getElementById('confidenceFill');
let conTxt = document.getElementById('confidenceText');
let guessTxt = document.getElementById('topGuessText');
let displayWrd = document.getElementById('wordDisplay');


//marker
let hLM; //hand
let pLM; //pose


// --- NEW: DUAL AI BRAINS ---
let aslBrain, cslBrain;
let aslLabels = [], cslLabels = [];


// The "Active" brain currently driving the app
let activeBrain;
let activeLabels = [];


//time and extract
let buffer = [];
let extWord = "";
let cooldown = 0;
let lastTime = -1;
const F_LENGTH = 30;  //frames
const DURATION = 1500;
const THRESHOLD = 0.90; //confidence lvl


//streak
let dynStr = 0;
let ldg = "";
let staStr = 0;
let lsg = "";


//canvas
const canvasElement = document.getElementsByClassName('canvasEl')[0];
const canvasCtx = canvasElement.getContext('2d');


// Global Memory Cache (The Ghost Hand fix)
let lastLeftHand = new Array(21 * 3).fill(0);
let lastRightHand = new Array(21 * 3).fill(0);


//finger
const { Thumb, Index, Middle, Ring, Pinky } = fp.Finger;
const { NoCurl, HalfCurl, FullCurl } = fp.FingerCurl;


//alphabets
const staticGestures = [
    createSign('A', [Thumb], [], [Index, Middle, Ring, Pinky]),
    createSign('B', [Index, Middle, Ring, Pinky], [Thumb], []),
    createSign('C', [], [Thumb, Index, Middle, Ring, Pinky], []),
    createSign('D', [Index], [Thumb], [Middle, Ring, Pinky]),
    createSign('E', [], [], [Thumb, Index, Middle, Ring, Pinky]),
    createSign('F', [Middle, Ring, Pinky], [Thumb, Index], []),
    createSign('G', [Index], [Thumb], [Middle, Ring, Pinky]),
    createSign('H', [Index, Middle], [Thumb], [Ring, Pinky]),
    createSign('I', [Pinky], [], [Thumb, Index, Middle, Ring]),
    createSign('K', [Index, Middle], [Thumb], [Ring, Pinky]),
    createSign('L', [Thumb, Index], [], [Middle, Ring, Pinky]),
    createSign('M', [], [Thumb], [Index, Middle, Ring, Pinky]),
    createSign('N', [], [Thumb], [Index, Middle, Ring, Pinky]),
    createSign('O', [], [Thumb, Index, Middle, Ring, Pinky], []),
    createSign('P', [Index, Middle], [Thumb], [Ring, Pinky]),
    createSign('Q', [Index], [Thumb], [Middle, Ring, Pinky]),
    createSign('R', [Index, Middle], [], [Thumb, Ring, Pinky]),
    createSign('S', [], [], [Thumb, Index, Middle, Ring, Pinky]),
    createSign('T', [], [], [Thumb, Index, Middle, Ring, Pinky]),
    createSign('U', [Index, Middle], [], [Thumb, Ring, Pinky]),
    createSign('V', [Index, Middle], [], [Thumb, Ring, Pinky]),
    createSign('W', [Index, Middle, Ring], [], [Thumb, Pinky]),
    createSign('X', [], [Index], [Thumb, Middle, Ring, Pinky]),
    createSign('Y', [Thumb, Pinky], [], [Index, Middle, Ring])
];
const GE = new fp.GestureEstimator(staticGestures);


//lines
const HAND_LINE = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]
];
const POSE_LINE = [
    [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8],
    [9, 10], [11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
    [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
    [11, 23], [12, 24], [23, 24], [23, 25], [24, 26], [25, 27], [26, 28],
    [27, 29], [28, 30], [29, 31], [30, 32], [27, 31], [28, 32]
];


//this function basically set the marker
async function setUpMediaPipe() {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");


    hLM = await HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`, delegate: "GPU" },
        runningMode: "VIDEO",
        numHands: 1
    });


    pLM = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`, delegate: "GPU" },
        runningMode: "VIDEO"
    });


    // --- NEW: LOAD BOTH BRAINS ---
    try {
        // Load ASL
        const aslResponse = await fetch('./asl-labels.json');
        aslLabels = await aslResponse.json();
        aslBrain = await tf.loadLayersModel('./asl-model.json');


        // Load CSL
        const cslResponse = await fetch('./csl_labels.json');
        cslLabels = await cslResponse.json();
        cslBrain = await tf.loadLayersModel('./csl_model.json');


        // Default to ASL on startup
        activeBrain = aslBrain;
        activeLabels = aslLabels;


        console.log("Both ASL and CSL Brains Online!");
    } catch (error) {
        console.error("Error loading models. Check your filenames!", error);
    }
}


setUpMediaPipe();


function createSign(name, noCurls, halfCurls, fullCurls) {
    const sign = new fp.GestureDescription(name);
    noCurls.forEach(f => sign.addCurl(f, NoCurl, 1.0));
    halfCurls.forEach(f => sign.addCurl(f, HalfCurl, 1.0));
    fullCurls.forEach(f => sign.addCurl(f, FullCurl, 1.0));
    return sign;
}


function detectStaticAlphabet(hand) {
    const keypoints3D = hand.map(point => [point.x * canvasElement.width, point.y * canvasElement.height, point.z * canvasElement.width]);
    const estimation = GE.estimate(keypoints3D, 7.0);
    if (estimation.gestures.length > 0) return estimation.gestures.reduce((p, c) => (p.score > c.score) ? p : c);
    return null;
}


document.getElementById('themeToggleBtn').addEventListener('click', () => {
    document.body.classList.toggle('dark-mode');
});


// ==========================================
// CUSTOM DROPDOWN UI LOGIC
// ==========================================


window.toggleDropdown = function () {
    // Matches id="modeDropdownContent" in your HTML
    document.getElementById('modeDropdownContent').classList.toggle('show');
}


window.selectMode = async function (modeKey, modeTitle, el) {
    // 1. Update text (Matches id="selectedModeText")
    document.getElementById('selectedModeText').innerText = modeTitle;

    // 2. Change active highlight (Matches class="dropdown-item")
    document.querySelectorAll('.dropdown-item').forEach(d => d.classList.remove('active'));
    el.classList.add('active');

    // 3. Close the menu
    document.getElementById('modeDropdownContent').classList.remove('show');

    // 4. Reset variables
    curMode = modeKey;
    window.clearWord();
    staStr = 0;
    dynStr = 0;


    // Grab UI elements
    const guessTextEl = document.getElementById('topGuessText');
    const outTextUI = document.getElementById('translationOutput');


    // 5. Swap modes and AI Brains
    if (curMode === "csl_word") {
        guessTextEl.innerText = `Mode: ${modeTitle} (Holistic)`;
        outTextUI.innerText = `[CSL]`;
        activeBrain = cslBrain;
        activeLabels = cslLabels;
        if (hLM) await hLM.setOptions({ numHands: 2 });

    } else if (curMode === "asl_word") {
        guessTextEl.innerText = `Mode: ${modeTitle} (Holistic)`;
        outTextUI.innerText = `[ASL]`;
        activeBrain = aslBrain;
        activeLabels = aslLabels;
        if (hLM) await hLM.setOptions({ numHands: 2 });

    } else {
        guessTextEl.innerText = "Mode: ASL Spelling";
        outTextUI.innerText = "...";
        activeBrain = aslBrain;
        activeLabels = aslLabels;
        if (hLM) await hLM.setOptions({ numHands: 1 });
    }
}


// Safely close dropdown if clicked outside (Matches class="custom-dropdown")
window.addEventListener('click', function (event) {
    if (!event.target.closest('.custom-dropdown')) {
        let dc = document.getElementById("modeDropdownContent");
        if (dc && dc.classList.contains('show')) dc.classList.remove('show');
    }
});


// --- UPDATED: SMART MODE SWITCHER ---
window.changeMode = async function () {
    const selector = document.getElementById('modeSelector');
    curMode = selector.value;
    window.clearWord();
    staStr = 0; dynStr = 0;


    if (curMode === "csl_word") {
        guessTxt.innerText = "Mode: CSL Words (Holistic)";
        extractTxt.innerText = "[CSL]";
        activeBrain = cslBrain;     // Switch to CSL
        activeLabels = cslLabels;   // Switch to CSL
        if (hLM) await hLM.setOptions({ numHands: 2 });

    } else if (curMode === "asl_word") {
        guessTxt.innerText = "Mode: ASL Words (Holistic)";
        extractTxt.innerText = "[ASL]";
        activeBrain = aslBrain;     // Switch to ASL
        activeLabels = aslLabels;   // Switch to ASL
        if (hLM) await hLM.setOptions({ numHands: 2 });

    } else {
        guessTxt.innerText = "Mode: ASL Spelling";
        extractTxt.innerText = "...";
        activeBrain = aslBrain;     // Default fallback
        activeLabels = aslLabels;
        if (hLM) await hLM.setOptions({ numHands: 1 });
    }


    console.log("Mode Switched:", curMode);
}


window.startCam = async function () {
    try {
        if (stream) window.stopCam();

        const cameraSettings = isMobile
            ? { facingMode: curCamera, width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 15 } }
            : { facingMode: curCamera, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } };


        stream = await navigator.mediaDevices.getUserMedia({ video: cameraSettings });

        videoEle.srcObject = stream;
        await videoEle.play();
        isPredicting = true;
        predictSign();


        // Update Start/Stop button
        btnCam.textContent = "Stop Camera";
        btnCam.onclick = window.stopCam;


        // Show and Reset Pause button
        if (btnPauseCam) {
            btnPauseCam.style.display = "inline-block";
            btnPauseCam.textContent = "Pause Camera";
            btnPauseCam.onclick = window.pauseCam;
        }
    } catch (err) { alert("Unable to access camera!"); }
}


window.pauseCam = function () {
    videoEle.pause();
    isPredicting = false;

    if (btnPauseCam) {
        btnPauseCam.textContent = "Resume Camera";
        btnPauseCam.onclick = window.resumeCam;
    }
}


window.resumeCam = async function () {
    await videoEle.play();
    isPredicting = true;
    predictSign();


    if (btnPauseCam) {
        btnPauseCam.textContent = "Pause Camera";
        btnPauseCam.onclick = window.pauseCam;
    }
}


window.stopCam = function () {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    if (videoEle) {
        videoEle.pause();
        videoEle.removeAttribute('src');
        videoEle.removeAttribute('srcObject');
        videoEle.srcObject = null;
        videoEle.load();
    }
    isPredicting = false;


    // Reset Start Button
    btnCam.textContent = "Start Camera";
    btnCam.onclick = window.startCam;


    // Hide Pause Button
    if (btnPauseCam) {
        btnPauseCam.style.display = "none";
        btnPauseCam.textContent = "Pause Camera";
        btnPauseCam.onclick = window.pauseCam;
    }

    resetHandMemory();
}


window.flipCam = async function () {
    curCamera = curCamera === "user" ? "environment" : "user";
    window.stopCam();
    await window.startCam();
}


window.clearWord = function () {
    extWord = "";
    displayWrd.innerText = "";
}


window.translateSentence = async function () {
    if (!extWord) return;
    const translationUI = document.getElementById('finalTranslation');
    translationUI.innerText = "Gemma is thinking...";
    try {
        const response = await fetch('/api/translate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ aslString: extWord })
        });


        if (!response.ok) {
            const errorText = await response.text();
            console.error("Backend failed:", errorText);
            translationUI.innerText = "Server Error: Model timed out or failed.";
            return;
        }


        const data = await response.json();
        translationUI.innerText = data.translation ? data.translation : "Error: Could not translate.";
    } catch (err) {
        console.error("Fetch Error:", err);
        translationUI.innerText = "Network Error.";
    }
}


const distant = (p1, p2) => Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));


// ==========================================
// THE HOLISTIC DATA MERGER (With Ghost Hands)
// ==========================================
function extractHolisticData(poseResults, handResults) {
    let poseData = new Array(33 * 3).fill(0);
    let lhData = [...lastLeftHand];
    let rhData = [...lastRightHand];


    if (poseResults && poseResults.landmarks && poseResults.landmarks.length > 0) {
        const pose = poseResults.landmarks[0];
        for (let i = 0; i < pose.length; i++) {
            poseData[i * 3] = pose[i].x;
            poseData[i * 3 + 1] = pose[i].y;
            poseData[i * 3 + 2] = pose[i].z;
        }
    }


    if (handResults && handResults.landmarks && handResults.landmarks.length > 0) {
        for (let h = 0; h < handResults.landmarks.length; h++) {
            const hand = handResults.landmarks[h];
            const handedness = handResults.handednesses[h][0].categoryName;


            let tempHandArray = new Array(63);
            for (let i = 0; i < hand.length; i++) {
                tempHandArray[i * 3] = hand[i].x;
                tempHandArray[i * 3 + 1] = hand[i].y;
                tempHandArray[i * 3 + 2] = hand[i].z;
            }


            if (handedness === "Left") {
                lhData = tempHandArray;
                lastLeftHand = tempHandArray;
            } else {
                rhData = tempHandArray;
                lastRightHand = tempHandArray;
            }
        }
    }


    return [...poseData, ...lhData, ...rhData];
}


function resetHandMemory() {
    lastLeftHand = new Array(21 * 3).fill(0);
    lastRightHand = new Array(21 * 3).fill(0);
}


//main action
async function predictSign() {
    if (!isPredicting) return;
    if (videoEle.videoWidth === 0) {
        window.requestAnimationFrame(predictSign);
        return;
    }


    let startTimeMs = performance.now();
    if (videoEle.currentTime !== lastTime) {
        lastTime = videoEle.currentTime;


        prepareCanvas();


        const handResults = hLM.detectForVideo(videoEle, startTimeMs);
        let poseResults = null;
        if (curMode === "asl_word" || curMode === "csl_word") {
            poseResults = pLM.detectForVideo(videoEle, startTimeMs);
        }


        drawSkeletons(poseResults, handResults);


        if (handResults.landmarks && handResults.landmarks.length > 0 || poseResults) {
            processHandLogic(handResults, poseResults);
        } else {
            resetTrackingState();
        }
    }
    window.requestAnimationFrame(predictSign);
}


function prepareCanvas() {
    if (canvasElement.width !== videoEle.videoWidth) {
        canvasElement.width = videoEle.videoWidth;
        canvasElement.height = videoEle.videoHeight;
    }
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
}


function drawSkeletons(poseResults, handResults) {
    if (!showSkeletons) return;


    if (poseResults && poseResults.landmarks && poseResults.landmarks.length > 0) {
        const pose = poseResults.landmarks[0];
        for (const connection of POSE_LINE) {
            const startX = pose[connection[0]].x * canvasElement.width;
            const startY = pose[connection[0]].y * canvasElement.height;
            const endX = pose[connection[1]].x * canvasElement.width;
            const endY = pose[connection[1]].y * canvasElement.height;
            canvasCtx.beginPath(); canvasCtx.moveTo(startX, startY); canvasCtx.lineTo(endX, endY);
            canvasCtx.strokeStyle = "lime";
            canvasCtx.lineWidth = 2;
            canvasCtx.stroke();
        }
    }


    if (handResults && handResults.landmarks) {
        for (const detectedHand of handResults.landmarks) {
            for (const connection of HAND_LINE) {
                const startX = detectedHand[connection[0]].x * canvasElement.width;
                const startY = detectedHand[connection[0]].y * canvasElement.height;
                const endX = detectedHand[connection[1]].x * canvasElement.width;
                const endY = detectedHand[connection[1]].y * canvasElement.height;
                canvasCtx.beginPath(); canvasCtx.moveTo(startX, startY); canvasCtx.lineTo(endX, endY);
                canvasCtx.strokeStyle = "cyan"; canvasCtx.lineWidth = 2; canvasCtx.stroke();
            }
        }
    }
}


//brain
function processHandLogic(handResults, poseResults) {
    let frameData = [];


    if (curMode === "asl_letter") {
        if (handResults.landmarks && handResults.landmarks.length > 0) {
            const hand = handResults.landmarks[0];
            for (let i = 0; i < hand.length; i++) {
                frameData.push(hand[i].x, hand[i].y, hand[i].z);
            }
        }
    } else {
        frameData = extractHolisticData(poseResults, handResults);
    }


    if (frameData.length > 0) {
        buffer.push(frameData);
        if (buffer.length > F_LENGTH) buffer.shift();
    }


    if (buffer.length === F_LENGTH) {
        if (curMode === "asl_letter") {
            const hand = handResults.landmarks[0];
            const oldWrist = buffer[0];
            const newWrist = buffer[29];
            const movementDistance = Math.sqrt(
                Math.pow(newWrist[0] - oldWrist[0], 2) + Math.pow(newWrist[1] - oldWrist[1], 2)
            );


            if (movementDistance > 0.08) {
                evaluateDynamicSign();
            } else if (hand) {
                evaluateStaticSign(hand);
            }
        } else {
            evaluateDynamicSign();
        }
    }
}


//dynamic
function evaluateDynamicSign() {
    staStr = 0;


    // Make sure we have a loaded brain before trying to predict!
    if (!activeBrain) return;


    const inputTensor = tf.tensor3d([buffer]);
    let prediction;
    try {
        prediction = activeBrain.predict(inputTensor); // <-- Uses the active brain!
    } catch (err) {
        inputTensor.dispose();
        return;
    }


    const predictionArray = prediction.dataSync();
    const highestIndex = prediction.argMax(1).dataSync()[0];
    const confidence = predictionArray[highestIndex];
    const rawPredictedWord = activeLabels[highestIndex] || ""; // <-- Uses the active labels!
    const sanitizedWord = rawPredictedWord.trim().toLowerCase();
    const isIdle = (sanitizedWord === "idle" || sanitizedWord === "none" || sanitizedWord === "" || sanitizedWord === "still");


    if (!isIdle && confidence > THRESHOLD) {
        updateDynamicUI(rawPredictedWord, confidence);


        if (rawPredictedWord === ldg) {
            dynStr++;
        } else {
            ldg = rawPredictedWord;
            dynStr = 1;
        }


        let requiredStreak = 12;


        if (rawPredictedWord.toUpperCase() === "J") {
            requiredStreak = 8;
        }


        if (dynStr >= requiredStreak && Date.now() > cooldown) {
            commitWord(rawPredictedWord);
            dynStr = 0;
            buffer = [];
            resetHandMemory();
        }
    } else {
        resetDynamicUI(confidence, isIdle);
    }


    inputTensor.dispose();
    prediction.dispose();
}


//static
function evaluateStaticSign(hand) {
    dynStr = 0;


    const staticGuess = detectStaticAlphabet(hand);


    if (staticGuess) {
        let finalLetter = applyGeometricFixes(staticGuess.name, hand);


        if (finalLetter === lsg) {
            staStr++;
        } else {
            lsg = finalLetter;
            staStr = 1;
        }


        updateStaticUI(finalLetter, staticGuess.score);


        if (staStr >= 3 && Date.now() > cooldown) {
            commitStaticLetter(finalLetter);
        }
    } else {
        resetStaticUI();
    }
}


function applyGeometricFixes(letter, hand) {
    let finalLetter = letter;


    if (['Y', 'I'].includes(finalLetter)) {
        const handWidth = distant(hand[5], hand[17]);
        const thumbStretch = distant(hand[4], hand[5]);
        finalLetter = thumbStretch > handWidth * 0.8 ? 'Y' : 'I';
    }


    if (['U', 'V', 'R', 'P', 'H', 'K'].includes(finalLetter)) {
        const deltaX = Math.abs(hand[8].x - hand[5].x);
        const deltaY = Math.abs(hand[8].y - hand[5].y);


        if (deltaX > deltaY * 1.2) { finalLetter = 'H'; }
        else if (hand[8].y > hand[5].y) { finalLetter = 'P'; }
        else {
            const knucklesDiff = hand[5].x - hand[9].x;
            const tipsDiff = hand[8].x - hand[12].x;
            if (knucklesDiff * tipsDiff < 0) { finalLetter = 'R'; }
            else {
                const tipGap = distant(hand[8], hand[12]);
                const knuckleGap = distant(hand[5], hand[9]);
                if (tipGap < knuckleGap * 1.4) { finalLetter = 'U'; }
                else if (hand[4].y < hand[9].y + 0.02) { finalLetter = 'K'; }
                else { finalLetter = 'V'; }
            }
        }
    }


    if (['D', 'Q', 'L', 'G'].includes(finalLetter)) {
        const deltaX = Math.abs(hand[8].x - hand[5].x);
        const deltaY = Math.abs(hand[8].y - hand[5].y);
        if (deltaX > deltaY * 1.2) { finalLetter = 'G'; }
        else if (hand[8].y > hand[5].y) { finalLetter = 'Q'; }
        else {
            const handWidth = distant(hand[5], hand[17]);
            const thumbStretch = distant(hand[4], hand[5]);
            finalLetter = thumbStretch > handWidth * 0.7 ? 'L' : 'D';
        }
    }


    if (['C', 'O', 'F'].includes(finalLetter)) {
        const isMiddleStraightUp = hand[12].y < hand[10].y;
        if (isMiddleStraightUp) { finalLetter = 'F'; }
        else {
            const pinchDistance = distant(hand[8], hand[4]);
            const knuckleWidth = distant(hand[5], hand[6]);
            finalLetter = pinchDistance < knuckleWidth ? 'O' : 'C';
        }
    }


    if (['A', 'E', 'S', 'T', 'M', 'N'].includes(finalLetter)) {
        const isFlipped = hand[5].x > hand[17].x;
        let thumbX = hand[4].x; let indX = hand[5].x; let ringX = hand[13].x;
        if (isFlipped) { thumbX = -thumbX; indX = -indX; ringX = -ringX; }


        const handWidth = ringX - indX;
        const thumbProgress = thumbX - indX;


        if (hand[4].y > hand[5].y + 0.05) { finalLetter = 'E'; }
        else if (hand[4].y < hand[6].y + 0.01 && thumbProgress > 0) { finalLetter = 'S'; }
        else if (thumbProgress < 0 && hand[4].y < hand[5].y + 0.03) { finalLetter = 'A'; }
        else if (thumbProgress > handWidth * 0.65) { finalLetter = 'M'; }
        else if (thumbProgress > handWidth * 0.25) { finalLetter = 'N'; }
        else { finalLetter = 'T'; }
    }


    return finalLetter;
}


//ui
function updateDynamicUI(word, confidence) {
    const confidencePercent = Math.round(confidence * 100);
    conCol.style.width = confidencePercent + "%";
    conCol.style.background = "lime";
    conTxt.innerText = confidencePercent + "%";
    guessTxt.innerText = "Motion: " + word;
}


function resetDynamicUI(confidence, isIdle) {
    dynStr = 0;
    ldg = "";
    const confPct = Math.round(confidence * 100);
    conCol.style.width = isIdle ? confPct + "%" : "0%";
    conCol.style.background = "gray";
    guessTxt.innerText = isIdle ? "Motion: Idle" : "Swaying...";
    conTxt.innerText = isIdle ? confPct + "%" : "0%";
}


function updateStaticUI(letter, score) {
    const confPercent = Math.round((score / 10) * 100);
    conCol.style.width = confPercent + "%";
    conCol.style.background = "cyan";
    guessTxt.innerText = "Pose: " + letter + " (Streak: " + staStr + "/3)";
    conTxt.innerText = confPercent + "%";
}


function resetStaticUI() {
    staStr = 0;
    conCol.style.width = "0%";
    guessTxt.innerText = "Pose: None";
    conTxt.innerText = "0%";
    if (Date.now() > cooldown) extractTxt.innerText = "...";
}


function commitStaticLetter(letter) {
    extractTxt.innerText = letter;
    if (letter.trim().toLowerCase() !== "idle") {
        extWord += letter;
        displayWrd.innerText = extWord;
    }
    cooldown = Date.now() + DURATION;
    staStr = 0;
}


function commitWord(word) {
    extractTxt.innerText = word;
    extWord += word + " ";
    displayWrd.innerText = extWord;
    cooldown = Date.now() + DURATION;
}


function resetTrackingState() {
    buffer = [];
    dynStr = 0;
    staStr = 0;
    lsg = "";


    if (Date.now() > cooldown) {
        extractTxt.innerText = "...";
        conCol.style.width = "0%";
        if (curMode === "asl_word" || curMode === "csl_word") {
            guessTxt.innerText = "Waiting for input... [Ready]";
        } else {
            guessTxt.innerText = "None";
        }
        conTxt.innerText = "0%";
    }
}