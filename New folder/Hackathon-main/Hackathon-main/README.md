# Age Estimator Dashboard

## Setup Instructions

### 1. Install Python Dependencies
```bash
cd backend
pip install flask flask-cors torch torchvision opencv-python pillow numpy librosa joblib scikit-learn
```

### 2. Start Backend Server
- Double-click `start_backend.bat` OR
- Run manually:
```bash
cd backend
python app.py
```

### 3. Open Frontend
- Open `dashboard.html` in your web browser
- Check that "Backend connected" appears in green

## Features
- **Image Upload**: Upload single/multiple images for age prediction
- **Real-time Camera**: Capture photos for instant age estimation  
- **Audio Upload**: Upload audio files for voice-based age prediction
- **Voice Recording**: Record voice in real-time for age estimation

## Models Used
- **Image**: ResNet-18 trained on age classification
- **Audio**: Machine learning model trained on voice features (MFCC, Chroma, Spectral Centroid)