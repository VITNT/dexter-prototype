let stream;
let videoElement = document.getElementById('videoEl');
let btn = document.getElementById('camBtn');
let flip = document.getElementById('flipBtn');
let currentCamera = "user";

async function startCamera() {
    try {
        if (stream) {
            stopCamera();
        }
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: currentCamera } });
        videoElement.srcObject = stream;
        await videoElement.play();
        flip.onclick = flipCamera;
        btn.textContent = "Pause Video";
        btn.onclick = pauseCamera;
    } catch (err) {
        console.error(err);
        alert("Unable to access camera!");
    }
}
function pauseCamera() {
    videoElement.pause();
    btn.textContent = "Resume Video";
    btn.onclick = startCamera;
}
function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
}
async function flipCamera() {
    currentCamera = currentCamera === "user" ? "environment" : "user";
    stopCamera();
    await startCamera();
}