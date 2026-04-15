import { HandLandmarker, PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";


let stream;
let videoEle = document.getElementById('videoEl');
let btn = document.getElementById('camBtn');
let flip = document.getElementById('flipBtn');
let modeBtn = document.getElementById('modeBtn');
let currentCamera = "user";


const canvasElement = document.getElementsByClassName('canvasEl')[0];
const canvasCtx = canvasElement.getContext('2d');


const SEQUENCE_LENGTH = 30;
let frameBuffer = [];


// --- STATE VARIABLES ---
let isRecording = false;
let recordedFrames = [];
let trackingMode = "hands"; // Starts in Alphabet mode


// --- MEDIAPIPE VARIABLES ---
let handLandmarker;
let poseLandmarker;
let lastVideoTime = -1;
let isPredicting = false;


const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]
];


const POSE_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8],
    [9, 10], [11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
    [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
    [11, 23], [12, 24], [23, 24], [23, 25], [24, 26], [25, 27], [26, 28],
    [27, 29], [28, 30], [29, 31], [30, 32], [27, 31], [28, 32]
];


// --- INITIALIZE AI ---
async function initMediaPipe() {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");

    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`, delegate: "GPU" },
        runningMode: "VIDEO", numHands: 1 // Defaults to 1 hand for clean Alphabet tracking
    });

    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`, delegate: "GPU" },
        runningMode: "VIDEO"
    });


    console.log("Studio Ready!");
}
initMediaPipe();


// --- UI CONTROLS ---
window.toggleMode = async function () {
    trackingMode = trackingMode === "hands" ? "holistic" : "hands";

    if (trackingMode === "hands") {
        modeBtn.textContent = "Mode: Alphabet (1 Hand)";
        modeBtn.style.backgroundColor = "#ff9800";
        await handLandmarker.setOptions({ numHands: 1 });
    } else {
        modeBtn.textContent = "Mode: Words (Body + 2 Hands)";
        modeBtn.style.backgroundColor = "#e91e63";
        await handLandmarker.setOptions({ numHands: 2 });
    }

    // Wipe buffers so 63-point arrays never mix with 225-point arrays!
    frameBuffer = [];
    recordedFrames = [];
    isRecording = false;
    document.getElementById('recordStatus').innerText = "Mode switched. Buffers cleared.";
    document.getElementById('recordStatus').style.color = "yellow";
}


window.startCamera = async function () {
    try {
        if (stream) window.stopCamera();
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: currentCamera } });
        videoEle.srcObject = stream;
        await videoEle.play();
        isPredicting = true;
        predictWebcam();
        btn.textContent = "Pause Video";
        btn.onclick = window.pauseCamera;
    } catch (err) {
        console.error(err);
        alert("Unable to access camera!");
    }
}


window.pauseCamera = function () {
    videoEle.pause();
    isPredicting = false;
    btn.textContent = "Resume Video";
    btn.onclick = window.startCamera;
}


window.stopCamera = function () {
    if (stream) { stream.getTracks().forEach(track => track.stop()); stream = null; }
    isPredicting = false;
}


window.flipCamera = async function () {
    currentCamera = currentCamera === "user" ? "environment" : "user";
    window.stopCamera();
    await window.startCamera();
}


window.recordGesture = function () {
    const gestureName = document.getElementById('gestureName').value;
    if (!gestureName) { alert("Please type a gesture name first!"); return; }


    isRecording = true;
    recordedFrames = [];
    document.getElementById('recordStatus').innerText = `Recording ${trackingMode} data...`;
    document.getElementById('recordStatus').style.color = "red";
    frameBuffer = []; // Clear to ensure we catch 30 completely fresh frames
}


function saveDataToFile(label, data) {
    const dataToSave = { label: label, frames: data };
    const blob = new Blob([JSON.stringify(dataToSave)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${label}_${trackingMode}_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}


// --- MEDIAPIPE DRAWING LOOP ---
async function predictWebcam() {
    if (!isPredicting) return;
    if (videoEle.videoWidth === 0) { window.requestAnimationFrame(predictWebcam); return; }


    let startTimeMs = performance.now();

    if (videoEle.currentTime !== lastVideoTime) {
        lastVideoTime = videoEle.currentTime;

        if (canvasElement.width !== videoEle.videoWidth) {
            canvasElement.width = videoEle.videoWidth;
            canvasElement.height = videoEle.videoHeight;
        }


        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);


        const handResults = handLandmarker.detectForVideo(videoEle, startTimeMs);
        let frameData = [];


        // ==========================================
        // MODE: HOLISTIC (Body + 2 Hands = 225 numbers)
        // ==========================================
        if (trackingMode === "holistic") {
            const poseResults = poseLandmarker.detectForVideo(videoEle, startTimeMs);

            // 1. EXTRACT BODY (99)
            if (poseResults.landmarks && poseResults.landmarks.length > 0) {
                const pose = poseResults.landmarks[0];
                for (let i = 0; i < pose.length; i++) frameData.push(pose[i].x, pose[i].y, pose[i].z);

                for (const connection of POSE_CONNECTIONS) {
                    const startX = pose[connection[0]].x * canvasElement.width;
                    const startY = pose[connection[0]].y * canvasElement.height;
                    const endX = pose[connection[1]].x * canvasElement.width;
                    const endY = pose[connection[1]].y * canvasElement.height;
                    canvasCtx.beginPath(); canvasCtx.moveTo(startX, startY); canvasCtx.lineTo(endX, endY);
                    canvasCtx.strokeStyle = "lime"; canvasCtx.lineWidth = 2; canvasCtx.stroke();
                }
            } else {
                for (let i = 0; i < 99; i++) frameData.push(0);
            }


            // 2. SET UP BUCKETS
            let leftHandData = new Array(63).fill(0);
            let rightHandData = new Array(63).fill(0);


            // 3. SORT HANDS
            if (handResults.landmarks) {
                for (let i = 0; i < handResults.landmarks.length; i++) {
                    const hand = handResults.landmarks[i];
                    const isRightHand = handResults.handednesses[i][0].categoryName === "Right";

                    let tempArray = [];
                    for (let point of hand) tempArray.push(point.x, point.y, point.z);


                    if (isRightHand) rightHandData = tempArray;
                    else leftHandData = tempArray;
                }
            }


            // 4. COMBINE
            frameData.push(...leftHandData);
            frameData.push(...rightHandData);
            frameBuffer.push(frameData);


            // ==========================================
            // MODE: HANDS ONLY (1 Hand = 63 numbers)
            // ==========================================
        } else if (trackingMode === "hands") {
            if (handResults.landmarks && handResults.landmarks.length > 0) {
                const hand = handResults.landmarks[0];
                for (let i = 0; i < hand.length; i++) frameData.push(hand[i].x, hand[i].y, hand[i].z);
                frameBuffer.push(frameData);
            } else {
                frameBuffer = [];
            }
        }


        // --- RECORDING BUFFER LOGIC ---
        if (frameBuffer.length > SEQUENCE_LENGTH) {
            frameBuffer.shift();
        }


        if (isRecording && frameBuffer.length > 0) {
            recordedFrames.push(frameBuffer[frameBuffer.length - 1]);
            if (recordedFrames.length === SEQUENCE_LENGTH) {
                isRecording = false;
                document.getElementById('recordStatus').innerText = "Done! Downloading...";
                document.getElementById('recordStatus').style.color = "lime";
                saveDataToFile(document.getElementById('gestureName').value, recordedFrames);
            }
        }


        // --- DRAW HANDS ---
        if (handResults.landmarks) {
            for (const hand of handResults.landmarks) {
                for (const connection of HAND_CONNECTIONS) {
                    const startX = hand[connection[0]].x * canvasElement.width;
                    const startY = hand[connection[0]].y * canvasElement.height;
                    const endX = hand[connection[1]].x * canvasElement.width;
                    const endY = hand[connection[1]].y * canvasElement.height;
                    canvasCtx.beginPath(); canvasCtx.moveTo(startX, startY); canvasCtx.lineTo(endX, endY);
                    canvasCtx.strokeStyle = "cyan"; canvasCtx.lineWidth = 2; canvasCtx.stroke();
                }
            }
        }
    }
    window.requestAnimationFrame(predictWebcam);
}

