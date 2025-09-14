# app.py

import os
import numpy as np
import librosa
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

# =========================================================
# --- 1. LOAD MODELS ---
# =========================================================
print("Loading audio models...")
MODEL_DIR = './models/'

AUDIO_MODEL = joblib.load(os.path.join(MODEL_DIR, 'final_model.sav'))
SCALER = joblib.load(os.path.join(MODEL_DIR, 'scaler.sav'))
FEATURE_SELECTOR = joblib.load(os.path.join(MODEL_DIR, 'feature_selector.sav'))
LABEL_ENCODER = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.sav'))
print("Audio models loaded successfully!")

# =========================================================
# --- 2. FEATURE EXTRACTION ---
# =========================================================
def extract_audio_features(audio_path):
    """
    Extract the same features the model was trained on.
    """
    features = []
    audio, _ = librosa.load(audio_path, sr=48000, duration=5)

    # Placeholder for gender if it was part of training
    gender_placeholder = 0.0

    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=48000))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=48000))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=48000))

    features.extend([gender_placeholder, spectral_centroid, spectral_bandwidth, spectral_rolloff])

    mfcc = librosa.feature.mfcc(y=audio, sr=48000)
    for el in mfcc:
        features.append(np.mean(el))

    return np.array(features).reshape(1, -1)

# =========================================================
# --- 3. FLASK APP ---
# =========================================================
app = Flask(_name_)
CORS(app)

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'status': 'Backend running', 'model': 'audio_age_predictor'}), 200

@app.route('/predict_audio', methods=['POST'])
def predict_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No audio file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No audio file selected'}), 400

    try:
        # Save uploaded file temporarily
        filepath = os.path.join("temp_audio.wav")
        file.save(filepath)

        # Extract features
        features = extract_audio_features(filepath)
        features_scaled = SCALER.transform(features)
        features_selected = FEATURE_SELECTOR.transform(features_scaled)

        # Predict
        prediction = AUDIO_MODEL.predict(features_selected)
        age_group = LABEL_ENCODER.inverse_transform(prediction)[0]

        os.remove(filepath)  # cleanup temp file
        return jsonify({'age_group': age_group})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =========================================================
# --- 4. RUN ---
# =========================================================
if _name_ == '_main_':
    app.run(debug=True, port=5000)