# app.py

import base64
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import io
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- NEW AUDIO MODEL IMPORTS ---
import librosa
import joblib

# =================================================================
# --- 1. IMAGE MODEL SETUP (Existing Code) ---
# =================================================================
AGE_CLASSES_IMG = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-120"]

app = Flask(__name__)
CORS(app)

def load_image_model(model_path, device):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 9)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    return model

def get_image_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_MODEL_PATH = "resnet_age_best.pth"
IMAGE_MODEL = load_image_model(IMAGE_MODEL_PATH, DEVICE)
IMAGE_TRANSFORM = get_image_transform()
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# =================================================================
# --- 2. NEW AUDIO MODEL SETUP (From your audio age.py) ---
# =================================================================
print("Loading audio models...")
AUDIO_MODEL = joblib.load('final_model.sav')
SCALER = joblib.load('scaler.sav')
FEATURE_SELECTOR = joblib.load('feature_selector.sav')
LABEL_ENCODER = joblib.load('label_encoder.sav')
print("Audio models loaded successfully!")

def extract_audio_features(audio_path):
    """Extract audio features (logic from your script)"""
    y, sr = librosa.load(audio_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    
    features = []
    features.extend(np.mean(mfccs, axis=1))
    features.extend(np.std(mfccs, axis=1))
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))
    features.extend([np.mean(spectral_centroid), np.std(spectral_centroid)])
    
    return np.array(features).reshape(1, -1)

# =================================================================
# --- 3. API ENDPOINTS ---
# =================================================================

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'status': 'Backend is running', 'models_loaded': True}), 200

# --- EXISTING IMAGE PREDICTION ENDPOINT ---
@app.route('/predict', methods=['POST'])
def predict():
    # ... (Your existing image prediction code remains here, unchanged)
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            return jsonify({'message': 'No faces detected.'})
        predictions = []
        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_rgb)
            input_tensor = IMAGE_TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                outputs = IMAGE_MODEL(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                pred_idx = torch.argmax(outputs, dim=1).item()
                confidence = probs[0][pred_idx].item()
            age_group = AGE_CLASSES_IMG[pred_idx]
            predictions.append({'age_group': age_group, 'confidence': round(confidence, 3)})
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            text = f"{age_group} ({confidence:.2f})"
            cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        _, buffer = cv2.imencode('.jpg', img)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'predictions': predictions, 'image': image_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- NEW AUDIO PREDICTION ENDPOINT ---
@app.route('/predict_audio', methods=['POST'])
def predict_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No audio file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No audio file selected'}), 400
    
    try:
        # librosa can load the file directly from the request stream
        features = extract_audio_features(file)
        
        # Apply scaling and feature selection
        features_scaled = SCALER.transform(features)
        features_selected = FEATURE_SELECTOR.transform(features_scaled)
        
        # Predict
        prediction = AUDIO_MODEL.predict(features_selected)
        
        # Decode the prediction to the age group label
        age_group = LABEL_ENCODER.inverse_transform(prediction)[0]
        
        return jsonify({'age_group': age_group})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =================================================================
# --- 4. RUN THE APP ---
# =================================================================
if __name__ == '__main__':
    app.run(debug=True, port=5000)