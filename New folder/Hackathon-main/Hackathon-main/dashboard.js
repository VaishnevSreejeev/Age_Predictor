// ==== BACKEND CONNECTION ====
async function checkBackendConnection() {
  try {
    const response = await fetch('http://localhost:5000/', { method: 'GET' });
    document.getElementById('status').innerHTML = '<span style="color: green;">✓ Backend connected</span>';
    return true;
  } catch (error) {
    document.getElementById('status').innerHTML = '<span style="color: red;">✗ Backend offline - Start backend server first</span>';
    return false;
  }
}

// Check connection on page load
window.addEventListener('load', checkBackendConnection);

// ==== API FUNCTIONS ====
async function predictImage(file) {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('http://localhost:5000/predict', {
    method: 'POST',
    body: formData
  });
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  
  return await response.json();
}

async function predictAudio(file) {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('http://localhost:5000/predict_audio', {
    method: 'POST',
    body: formData
  });
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  
  return await response.json();
}

function displayResults(results, type) {
  const modal = document.createElement('div');
  modal.className = 'modal';
  modal.style.display = 'flex';
  
  const content = document.createElement('div');
  content.className = 'modal-content';
  content.style.maxWidth = '600px';
  
  const closeBtn = document.createElement('span');
  closeBtn.className = 'close';
  closeBtn.innerHTML = '&times;';
  closeBtn.onclick = () => modal.remove();
  
  const title = document.createElement('h2');
  title.textContent = `${type === 'image' ? 'Image' : 'Audio'} Prediction Results`;
  
  const resultsDiv = document.createElement('div');
  resultsDiv.style.maxHeight = '400px';
  resultsDiv.style.overflowY = 'auto';
  
  results.forEach(result => {
    const item = document.createElement('div');
    item.style.cssText = 'border: 1px solid #ddd; margin: 10px 0; padding: 10px; border-radius: 5px;';
    
    if (result.error) {
      item.innerHTML = `<strong>${result.filename}</strong><br><span style="color: red;">Error: ${result.error}</span>`;
    } else if (type === 'image' && result.predictions) {
      const predictions = result.predictions.map(p => 
        `Age Group: ${p.age_group} (Confidence: ${(p.confidence * 100).toFixed(1)}%)`
      ).join('<br>');
      item.innerHTML = `<strong>${result.filename}</strong><br>${predictions}`;
    } else if (type === 'audio' && result.age_group) {
      item.innerHTML = `<strong>${result.filename}</strong><br>Predicted Age Group: ${result.age_group}`;
    }
    
    resultsDiv.appendChild(item);
  });
  
  // Add CSV download button for multiple results
  if (results.length > 1) {
    const downloadBtn = document.createElement('button');
    downloadBtn.textContent = '📥 Download CSV';
    downloadBtn.style.cssText = 'margin: 10px 0; padding: 8px 16px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer;';
    downloadBtn.onclick = () => downloadResultsCSV(results, type);
    content.appendChild(downloadBtn);
  }
  
  content.appendChild(closeBtn);
  content.appendChild(title);
  content.appendChild(resultsDiv);
  modal.appendChild(content);
  document.body.appendChild(modal);
  
  showNotification(`Processed ${results.length} ${type === 'image' ? 'images' : 'audio files'}`, "success");
}

// ==== UTILITY FUNCTIONS ====
function showNotification(message, type = "info") {
  // Create notification element if it doesn't exist
  let notification = document.getElementById("notification");
  if (!notification) {
    notification = document.createElement("div");
    notification.id = "notification";
    notification.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      padding: 12px 20px;
      border-radius: 4px;
      color: white;
      font-weight: bold;
      z-index: 10000;
      transition: opacity 0.3s ease;
    `;
    document.body.appendChild(notification);
  }
  
  // Set styles based on type
  const colors = {
    success: "#4CAF50",
    error: "#f44336",
    info: "#2196F3"
  };
  
  notification.style.backgroundColor = colors[type] || colors.info;
  notification.textContent = message;
  notification.style.opacity = "1";
  
  // Auto-hide after 3 seconds
  setTimeout(() => {
    notification.style.opacity = "0";
  }, 3000);
}

function downloadCSV(files, filename) {
  let csvContent = "filename,class\n";
  Array.from(files).forEach(file => {
    // Escape filename to prevent CSV injection
    const safeName = file.name.replace(/"/g, '""');
    csvContent += `"${safeName}",\n`;
  });

  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);

  // Download
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  
  // Clean up
  setTimeout(() => URL.revokeObjectURL(url), 100);
}

function downloadResultsCSV(results, type) {
  let csvContent = type === 'image' ? "filename,age_group,confidence\n" : "filename,age_group\n";
  
  results.forEach(result => {
    const safeName = result.filename.replace(/"/g, '""');
    
    if (result.error) {
      csvContent += `"${safeName}","Error: ${result.error.replace(/"/g, '""')}",\n`;
    } else if (type === 'image' && result.predictions) {
      result.predictions.forEach(pred => {
        csvContent += `"${safeName}","${pred.age_group}",${pred.confidence.toFixed(3)}\n`;
      });
    } else if (type === 'audio' && result.age_group) {
      csvContent += `"${safeName}","${result.age_group}"\n`;
    }
  });

  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${type}_predictions.csv`;
  link.click();
  
  setTimeout(() => URL.revokeObjectURL(url), 100);
  showNotification("CSV downloaded successfully!", "success");
}

// ==== IMAGE UPLOAD ====
function openImageModal() {
  document.getElementById("imageModal").style.display = "flex";
  document.getElementById("uploadedImg").style.display = "none";
  document.getElementById("imagePicker").value = "";
  document.getElementById("multiImagePicker").value = "";
  document.getElementById("multiUploadedImgs").innerHTML = "";
  const submitBtn = document.getElementById("singleImageSubmit");
  if (submitBtn) submitBtn.style.display = "none";
}

function closeImageModal() {
  document.getElementById("imageModal").style.display = "none";
}

function previewImage(event) {
  const file = event.target.files[0];
  if (!file || !file.type.startsWith('image/')) return;
  
  const reader = new FileReader();
  reader.onload = function(e) {
    const img = document.getElementById("uploadedImg");
    const submitBtn = document.getElementById("singleImageSubmit");
    img.src = e.target.result;
    img.style.display = "block";
    if (submitBtn) submitBtn.style.display = "inline-block";
  };
  reader.readAsDataURL(file);
}

async function submitSingleImage() {
  const fileInput = document.getElementById("imagePicker");
  const file = fileInput.files[0];
  
  if (!file) {
    showNotification("No image selected!", "error");
    return;
  }
  
  try {
    showNotification("Processing image...", "info");
    const prediction = await predictImage(file);
    displayResults([{ filename: file.name, ...prediction }], "image");
  } catch (error) {
    showNotification("Error processing image: " + error.message, "error");
  }
}

function previewMultipleImages(event) {
  const files = event.target.files;
  const container = document.getElementById("multiUploadedImgs");
  container.innerHTML = "";

  Array.from(files).forEach(file => {
    if (!file.type.startsWith('image/')) return;
    
    const reader = new FileReader();
    reader.onload = function(e) {
      const img = document.createElement("img");
      img.src = e.target.result;
      img.style.width = "100px";
      img.style.height = "100px";
      img.style.borderRadius = "10px";
      img.style.objectFit = "cover";
      container.appendChild(img);
    };
    reader.readAsDataURL(file);
  });
}

async function submitImages() {
  const files = document.getElementById("multiImagePicker").files;

  if (files.length === 0) {
    showNotification("No images selected!", "error");
    return;
  }

  showNotification("Processing images...", "info");
  const results = [];
  
  for (const file of files) {
    try {
      const prediction = await predictImage(file);
      results.push({ filename: file.name, ...prediction });
    } catch (error) {
      results.push({ filename: file.name, error: error.message });
    }
  }
  
  displayResults(results, "image");
}

// ==== REAL-TIME FACE DETECTION ====
let cameraStream;

function openCameraModal() {
  document.getElementById("cameraModal").style.display = "flex";
  document.getElementById("cameraContainer").style.display = "block";
  document.getElementById("capturedContainer").style.display = "none";

  navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
      cameraStream = stream;
      const video = document.getElementById("camera");
      video.srcObject = stream;
    })
    .catch(err => showNotification("Camera access denied: " + err.message, "error"));
}

function closeCameraModal() {
  document.getElementById("cameraModal").style.display = "none";
  if (cameraStream) {
    cameraStream.getTracks().forEach(track => track.stop());
  }
}

function capturePhoto() {
  const video = document.getElementById("camera");
  
  if (video.videoWidth === 0 || video.videoHeight === 0) {
    showNotification("Camera not ready. Please wait.", "error");
    return;
  }
  
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const capturedImage = document.getElementById("capturedImage");
  capturedImage.src = canvas.toDataURL("image/png");

  document.getElementById("cameraContainer").style.display = "none";
  document.getElementById("capturedContainer").style.display = "block";

  if (cameraStream) {
    cameraStream.getTracks().forEach(track => track.stop());
  }
}

function retryCamera() {
  closeCameraModal();
  openCameraModal();
}

async function submitCapturedPhoto() {
  const canvas = document.createElement("canvas");
  const img = document.getElementById("capturedImage");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);
  
  canvas.toBlob(async (blob) => {
    try {
      showNotification("Processing captured photo...", "info");
      const file = new File([blob], "captured_photo.jpg", { type: "image/jpeg" });
      const prediction = await predictImage(file);
      displayResults([{ filename: "Captured Photo", ...prediction }], "image");
    } catch (error) {
      showNotification("Error processing photo: " + error.message, "error");
    }
  }, "image/jpeg");
}

// ==== AUDIO UPLOAD ====
function openAudioModal() {
  document.getElementById("audioModal").style.display = "flex";
  document.getElementById("uploadedAudio").style.display = "none";
  document.getElementById("audioPicker").value = "";
  document.getElementById("multiAudioPicker").value = "";
  document.getElementById("multiUploadedAudios").innerHTML = "";
}

function closeAudioModal() {
  document.getElementById("audioModal").style.display = "none";
}

function previewAudio(event) {
  const file = event.target.files[0];
  if (!file || !file.type.startsWith('audio/')) return;
  
  const audio = document.getElementById("uploadedAudio");
  const audioURL = URL.createObjectURL(file);
  audio.src = audioURL;
  audio.style.display = "block";
  
  // Clean up previous URL
  audio.addEventListener('loadstart', () => {
    if (audio.dataset.prevUrl) {
      URL.revokeObjectURL(audio.dataset.prevUrl);
    }
    audio.dataset.prevUrl = audioURL;
  });
}

function previewMultipleAudios(event) {
  const files = event.target.files;
  const container = document.getElementById("multiUploadedAudios");
  
  // Clean up previous URLs
  container.querySelectorAll('audio').forEach(audio => {
    if (audio.src.startsWith('blob:')) {
      URL.revokeObjectURL(audio.src);
    }
  });
  container.innerHTML = "";

  Array.from(files).forEach(file => {
    if (!file.type.startsWith('audio/')) return;
    
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.style.display = "block";
    audio.style.marginTop = "10px";
    audio.src = URL.createObjectURL(file);
    container.appendChild(audio);
  });
}

async function submitAudios() {
  const files = document.getElementById("multiAudioPicker").files;

  if (files.length === 0) {
    showNotification("No audios selected!", "error");
    return;
  }

  showNotification("Processing audio files...", "info");
  const results = [];
  
  for (const file of files) {
    try {
      const prediction = await predictAudio(file);
      results.push({ filename: file.name, ...prediction });
    } catch (error) {
      results.push({ filename: file.name, error: error.message });
    }
  }
  
  displayResults(results, "audio");
}

// ==== REAL-TIME VOICE RECORDING ====
let mediaRecorder;
let audioChunks = [];

function openVoiceModal() {
  document.getElementById("voiceModal").style.display = "flex";
  document.getElementById("recordedContainer").style.display = "none";
}

function closeVoiceModal() {
  document.getElementById("voiceModal").style.display = "none";
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
}

function startRecording() {
  navigator.mediaDevices.getUserMedia({ audio: true })
    .then(stream => {
      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];
      mediaRecorder.start();

      const startBtn = document.getElementById("startRecordBtn") || document.querySelector("#voiceControls button:first-child");
      const stopBtn = document.getElementById("stopRecordBtn") || document.querySelector("#voiceControls button:last-child");
      if (startBtn) startBtn.style.display = "none";
      if (stopBtn) stopBtn.style.display = "inline-block";

      mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
        const audioURL = URL.createObjectURL(audioBlob);
        const player = document.getElementById("audioPlayer");

        player.src = audioURL;
        document.getElementById("recordedContainer").style.display = "block";

        stream.getTracks().forEach(track => track.stop());
      };
    })
    .catch(err => showNotification("Microphone access denied: " + err.message, "error"));
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
  }
  const startBtn = document.getElementById("startRecordBtn") || document.querySelector("#voiceControls button:first-child");
  const stopBtn = document.getElementById("stopRecordBtn") || document.querySelector("#voiceControls button:last-child");
  if (stopBtn) stopBtn.style.display = "none";
  if (startBtn) startBtn.style.display = "inline-block";
}

async function submitVoice() {
  const audioPlayer = document.getElementById("audioPlayer");
  if (!audioPlayer.src) {
    showNotification("No recording found!", "error");
    return;
  }
  
  try {
    showNotification("Processing voice recording...", "info");
    const response = await fetch(audioPlayer.src);
    const blob = await response.blob();
    const file = new File([blob], "voice_recording.wav", { type: "audio/wav" });
    const prediction = await predictAudio(file);
    displayResults([{ filename: "Voice Recording", ...prediction }], "audio");
  } catch (error) {
    showNotification("Error processing recording: " + error.message, "error");
  }
}