import {
  HandLandmarker,
  PoseLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

let cameraStream;
let videoEl = document.getElementById("webcamFeed");
let cameraToggle = document.getElementById("cameraToggleBtn");
let flipBtn = document.getElementById("flipCameraBtn");
let modeToggle = document.getElementById("trackingModeBtn");
let activeCamera = "user";

const overlayCanvas = document.getElementsByClassName("overlayCanvas")[0];
const ctx = overlayCanvas.getContext("2d");

const SEQUENCE_LENGTH = 30;
let frameBuffer = [];

let isCapturing = false;
let capturedFrames = [];
let trackingMode = "hands"; // starts in alphabet mode

// mediapipe setup
let handTracker;
let poseTracker;
let lastFrameTime = -1;
let isRunning = false;

const HAND_BONES = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [13, 17],
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20],
];

const POSE_BONES = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 7],
  [0, 4],
  [4, 5],
  [5, 6],
  [6, 8],
  [9, 10],
  [11, 12],
  [11, 13],
  [13, 15],
  [15, 17],
  [15, 19],
  [15, 21],
  [17, 19],
  [12, 14],
  [14, 16],
  [16, 18],
  [16, 20],
  [16, 22],
  [18, 20],
  [11, 23],
  [12, 24],
  [23, 24],
  [23, 25],
  [24, 26],
  [25, 27],
  [26, 28],
  [27, 29],
  [28, 30],
  [29, 31],
  [30, 32],
  [27, 31],
  [28, 32],
];

// load the heavy AI models
async function initTrackers() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm",
  );

  handTracker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 1, // defaults to 1 hand for alphabet tracking
  });

  poseTracker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
      delegate: "GPU",
    },
    runningMode: "VIDEO",
  });

  console.log("Studio Ready!");
}
initTrackers();

// camera toggles and handlers
window.toggleMode = async function () {
  trackingMode = trackingMode === "hands" ? "holistic" : "hands";

  if (trackingMode === "hands") {
    modeToggle.textContent = "Mode: Alphabet (1 Hand)";
    modeToggle.style.backgroundColor = "#ff9800";
    await handTracker.setOptions({ numHands: 1 });
  } else {
    modeToggle.textContent = "Mode: Words (Body + 2 Hands)";
    modeToggle.style.backgroundColor = "#e91e63";
    await handTracker.setOptions({ numHands: 2 });
  }

  // wipe buffers so 63-point arrays never mix with 225-point arrays
  frameBuffer = [];
  capturedFrames = [];
  isCapturing = false;
  document.getElementById("captureStatus").innerText =
    "Mode switched. Buffers cleared.";
  document.getElementById("captureStatus").style.color = "yellow";
};

window.startCamera = async function () {
  try {
    if (cameraStream) window.stopCamera();
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: activeCamera },
    });
    videoEl.srcObject = cameraStream;
    await videoEl.play();
    isRunning = true;
    runDetectionLoop();
    cameraToggle.textContent = "Pause Video";
    cameraToggle.onclick = window.pauseCamera;
  } catch (err) {
    console.error(err);
    alert("Unable to access camera!");
  }
};

window.pauseCamera = function () {
  videoEl.pause();
  isRunning = false;
  cameraToggle.textContent = "Resume Video";
  cameraToggle.onclick = window.startCamera;
};

window.stopCamera = function () {
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }
  isRunning = false;
};

window.flipCamera = async function () {
  activeCamera = activeCamera === "user" ? "environment" : "user";
  window.stopCamera();
  await window.startCamera();
};

window.recordGesture = function () {
  const labelName = document.getElementById("signLabel").value;
  if (!labelName) {
    alert("Please type a gesture name first!");
    return;
  }

  isCapturing = true;
  capturedFrames = [];
  document.getElementById("captureStatus").innerText =
    `Recording ${trackingMode} data...`;
  document.getElementById("captureStatus").style.color = "red";
  frameBuffer = []; // clear so we get 30 completely fresh frames
};

function downloadAsJSON(label, frames) {
  const payload = { label: label, frames: frames };
  const blob = new Blob([JSON.stringify(payload)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${label}_${trackingMode}_${Date.now()}.json`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// the heart of the recorder — runs constantly while camera is active
async function runDetectionLoop() {
  if (!isRunning) return;
  if (videoEl.videoWidth === 0) {
    window.requestAnimationFrame(runDetectionLoop);
    return;
  }

  let now = performance.now();

  if (videoEl.currentTime !== lastFrameTime) {
    lastFrameTime = videoEl.currentTime;

    if (overlayCanvas.width !== videoEl.videoWidth) {
      overlayCanvas.width = videoEl.videoWidth;
      overlayCanvas.height = videoEl.videoHeight;
    }

    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    const handResults = handTracker.detectForVideo(videoEl, now);
    let frameData = [];

    // if we're tracking full words, we need body posture plus both hands (225 data points total)
    if (trackingMode === "holistic") {
      const poseResults = poseTracker.detectForVideo(videoEl, now);

      // grab the body skeleton
      if (poseResults.landmarks && poseResults.landmarks.length > 0) {
        const pose = poseResults.landmarks[0];
        for (let i = 0; i < pose.length; i++)
          frameData.push(pose[i].x, pose[i].y, pose[i].z);

        for (const bone of POSE_BONES) {
          const startX = pose[bone[0]].x * overlayCanvas.width;
          const startY = pose[bone[0]].y * overlayCanvas.height;
          const endX = pose[bone[1]].x * overlayCanvas.width;
          const endY = pose[bone[1]].y * overlayCanvas.height;
          ctx.beginPath();
          ctx.moveTo(startX, startY);
          ctx.lineTo(endX, endY);
          ctx.strokeStyle = "lime";
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      } else {
        for (let i = 0; i < 99; i++) frameData.push(0);
      }

      // empty arrays in case a hand is missing
      let leftHandData = new Array(63).fill(0);
      let rightHandData = new Array(63).fill(0);

      // figure out which hand is which before saving
      if (handResults.landmarks) {
        for (let i = 0; i < handResults.landmarks.length; i++) {
          const hand = handResults.landmarks[i];
          const isRightHand =
            handResults.handednesses[i][0].categoryName === "Right";

          let tempArray = [];
          for (let point of hand) tempArray.push(point.x, point.y, point.z);

          if (isRightHand) rightHandData = tempArray;
          else leftHandData = tempArray;
        }
      }

      // smash them all together
      frameData.push(...leftHandData);
      frameData.push(...rightHandData);
      frameBuffer.push(frameData);

      // for simple spelling, we only care about the primary hand (63 data points)
    } else if (trackingMode === "hands") {
      if (handResults.landmarks && handResults.landmarks.length > 0) {
        const hand = handResults.landmarks[0];
        for (let i = 0; i < hand.length; i++)
          frameData.push(hand[i].x, hand[i].y, hand[i].z);
        frameBuffer.push(frameData);
      } else {
        frameBuffer = [];
      }
    }

    // manage the sliding window of frames
    if (frameBuffer.length > SEQUENCE_LENGTH) {
      frameBuffer.shift();
    }

    if (isCapturing && frameBuffer.length > 0) {
      capturedFrames.push(frameBuffer[frameBuffer.length - 1]);
      if (capturedFrames.length === SEQUENCE_LENGTH) {
        isCapturing = false;
        document.getElementById("captureStatus").innerText =
          "Done! Downloading...";
        document.getElementById("captureStatus").style.color = "lime";
        downloadAsJSON(
          document.getElementById("signLabel").value,
          capturedFrames,
        );
      }
    }

    // draw the lines so the user knows they're being tracked
    if (handResults.landmarks) {
      for (const hand of handResults.landmarks) {
        for (const bone of HAND_BONES) {
          const startX = hand[bone[0]].x * overlayCanvas.width;
          const startY = hand[bone[0]].y * overlayCanvas.height;
          const endX = hand[bone[1]].x * overlayCanvas.width;
          const endY = hand[bone[1]].y * overlayCanvas.height;
          ctx.beginPath();
          ctx.moveTo(startX, startY);
          ctx.lineTo(endX, endY);
          ctx.strokeStyle = "cyan";
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      }
    }
  }
  window.requestAnimationFrame(runDetectionLoop);
}
