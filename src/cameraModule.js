// src/cameraModule.js - Camera Capture & Reference Selection Module

document.addEventListener('DOMContentLoaded', () => {
  let mediaStream = null;

  // DOM Elements
  const cameraModal = document.getElementById('cameraModal');
  const cameraButton = document.getElementById('cameraButton');
  const closeCameraModal = document.getElementById('closeCameraModal');
  const cancelCameraButton = document.getElementById('cancelCameraButton');
  const captureButton = document.getElementById('captureButton');
  const webcamVideo = document.getElementById('webcamVideo');
  const cameraCanvas = document.getElementById('cameraCanvas');
  const referenceFileName = document.getElementById('referenceFileName');
  const referencePreview = document.getElementById('referencePreview');

  if (!cameraButton || !cameraModal) {
    console.warn("Camera modal or button elements not found in DOM.");
    return;
  }

  // 1. Open Webcam Stream
  async function openCamera() {
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } }
      });
      if (webcamVideo) {
        webcamVideo.srcObject = mediaStream;
      }
      cameraModal.classList.remove('hidden');
    } catch (err) {
      alert('Unable to access camera: ' + err.message);
    }
  }

  // 2. Stop Webcam Stream
  function stopCamera() {
    if (mediaStream) {
      mediaStream.getTracks().forEach(track => track.stop());
      mediaStream = null;
    }
    if (webcamVideo) webcamVideo.srcObject = null;
    if (cameraModal) cameraModal.classList.add('hidden');
  }

  // 3. Capture Photo & Pass to Backend
  async function capturePhoto() {
    if (!mediaStream) return;

    if (!cameraCanvas || !webcamVideo) return;
    const context = cameraCanvas.getContext('2d');
    cameraCanvas.width = webcamVideo.videoWidth || 640;
    cameraCanvas.height = webcamVideo.videoHeight || 480;

    context.drawImage(webcamVideo, 0, 0, cameraCanvas.width, cameraCanvas.height);
    const dataUrl = cameraCanvas.toDataURL('image/jpeg', 0.95);

    stopCamera();

    // Call IPC exposed via preload.js
    if (window.electronAPI && window.electronAPI.saveCameraPhoto) {
      const savedPath = await window.electronAPI.saveCameraPhoto(dataUrl);

      if (savedPath) {
        // Update UI Reference Preview
        if (referenceFileName) {
          const fileName = savedPath.split(/[\\/]/).pop();
          referenceFileName.textContent = `Captured: ${fileName}`;
        }
        if (referencePreview) {
          referencePreview.innerHTML = `<img src="${dataUrl}" style="max-height: 180px; width: auto; border-radius: 8px; object-fit: contain;" />`;
        }

        // Notify main renderer about the newly set reference photo path
        window.dispatchEvent(new CustomEvent('reference-photo-selected', { detail: { path: savedPath } }));
      }
    } else {
      console.warn("electronAPI.saveCameraPhoto is not defined in preload.js");
    }
  }

  // Attach Event Handlers Safely
  cameraButton.addEventListener('click', openCamera);
  if (closeCameraModal) closeCameraModal.addEventListener('click', stopCamera);
  if (cancelCameraButton) cancelCameraButton.addEventListener('click', stopCamera);
  if (captureButton) captureButton.addEventListener('click', capturePhoto);
});