# app.py

import base64
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- AUDIO IMPORTS ---
import os
import tempfile
import librosa
import joblib

# =================================================================
# --- 1. IMAGE MODEL SETUP ---
# =================================================================
AGE_CLASSES_IMG = ["0-2", "3-9", "10-19", "20-29", "30-39",
                   "40-49", "50-59", "60-69", "70-120"]

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
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_MODEL_PATH = "resnet_age_best.pth"
IMAGE_MODEL = load_image_model(IMAGE_MODEL_PATH, DEVICE)
IMAGE_TRANSFORM = get_image_transform()
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# =================================================================
# --- 2. AUDIO MODEL SETUP ---
# =================================================================
class AudioPredictor:
    def __init__(self, model_dir='./models/'):
        try:
            self.model = joblib.load(os.path.join(model_dir, 'final_model.sav'))
            self.scaler = joblib.load(os.path.join(model_dir, 'scaler.sav'))
            self.feature_selector = joblib.load(os.path.join(model_dir, 'feature_selector.sav'))
            self.label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.sav'))
            self.is_loaded = True
        except Exception as e:
            print(f"❌ Error loading audio models: {e}")
            print(f"📁 Looking in directory: {model_dir}")
            self.is_loaded = False

    def extract_features(self, audio_path):
        try:
            features = []
            audio, _ = librosa.load(audio_path, sr=48000, duration=5)
            gender_placeholder = 0.0
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=48000))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=48000))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=48000))

            features.extend([gender_placeholder, spectral_centroid, spectral_bandwidth, spectral_rolloff])

            mfcc = librosa.feature.mfcc(y=audio, sr=48000)
            for el in mfcc:
                features.append(np.mean(el))

            return np.array(features).reshape(1, -1)
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None

    def predict_age(self, audio_path):
        print(f"🎤 Predicting age for: {audio_path}")
        if not self.is_loaded:
            print("❌ Models not loaded")
            return "Error: Models not loaded."

        try:
            features = self.extract_features(audio_path)
            if features is None:
                print("❌ Feature extraction failed")
                return "Feature extraction failed."

            print(f"✅ Features extracted: {features.shape}")
            features_scaled = self.scaler.transform(features)
            features_selected = self.feature_selector.transform(features_scaled)
            prediction = self.model.predict(features_selected)
            age_group = self.label_encoder.inverse_transform(prediction)[0]
            print(f"🎯 Predicted: {age_group}")
            return age_group
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return f"Error: {str(e)}"

# Initialize audio predictor
AUDIO_PREDICTOR = AudioPredictor(model_dir='./models/')

# =================================================================
# --- 3. API ENDPOINTS ---
# =================================================================
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'Backend is running',
        'image_model_loaded': True,
        'audio_model_loaded': AUDIO_PREDICTOR.is_loaded
    }), 200

# --- IMAGE PREDICTION ENDPOINT ---
@app.route('/predict', methods=['POST'])
def predict():
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
            predictions.append({'age_group': age_group,
                                'confidence': round(confidence, 3)})
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            text = f"{age_group} ({confidence:.2f})"
            cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
        _, buffer = cv2.imencode('.jpg', img)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'predictions': predictions, 'image': image_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- AUDIO PREDICTION ENDPOINT ---
@app.route('/predict_audio', methods=['POST'])
def predict_audio():
    print("\n🎵 Audio prediction request received")
    
    if not AUDIO_PREDICTOR.is_loaded:
        print("❌ Audio models not loaded")
        return jsonify({'error': 'Audio models not loaded'}), 500
        
    if 'file' not in request.files:
        print("❌ No file in request")
        return jsonify({'error': 'No audio file'}), 400
        
    file = request.files['file']
    if file.filename == '':
        print("❌ Empty filename")
        return jsonify({'error': 'No audio file selected'}), 400

    print(f"📁 Processing file: {file.filename}")
    temp_path = None
    try:
        temp_path = tempfile.mktemp(suffix='.wav')
        file.save(temp_path)
        print(f"💾 Saved to: {temp_path}")
        
        age_group = AUDIO_PREDICTOR.predict_age(temp_path)
        
        if age_group.startswith('Error'):
            print(f"❌ Prediction failed: {age_group}")
            return jsonify({'error': age_group}), 500
            
        print(f"✅ Success: {age_group}")
        return jsonify({'age_group': age_group})
        
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"🗑️ Cleaned up: {temp_path}")

# =================================================================
# --- 4. RUN THE APP ---
# =================================================================
if __name__ == '__main__':
    app.run(debug=True, port=5000)
