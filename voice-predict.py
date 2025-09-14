import numpy as np
import pandas as pd
import librosa
import joblib
import os
import soundfile as sf
from pathlib import Path

print("Libraries imported successfully.")


# --- Step 2: DEFINE THE CORRECTED PREDICTOR CLASS ---

class AudioPredictor:
    def __init__(self, model_dir='./models/'):
        """
        Initializes the predictor by loading all pipeline components.
        """
        print("--- Loading pipeline components ---")
        try:
            self.model = joblib.load(os.path.join(model_dir, 'final_model.sav'))
            self.scaler = joblib.load(os.path.join(model_dir, 'scaler.sav'))
            self.feature_selector = joblib.load(os.path.join(model_dir, 'feature_selector.sav'))
            self.label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.sav'))
            print("All components loaded successfully.")
            self.is_loaded = True
        except FileNotFoundError as e:
            print(f"ERROR: Could not find a model file. {e}")
            print("Please make sure your .sav files are in the specified directory.")
            self.is_loaded = False

    # --- FIXED: Use the CORRECT feature extraction function from training ---
    def extract_features(self, audio_path):
        """
        Extracts the exact same features the model was trained on.
        """
        try:
            features = list()
            audio, _ = librosa.load(audio_path, sr=48000, duration=5)

            # We use a placeholder for gender since it's unknown for a new file.
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
            print(f"Error during feature extraction: {e}")
            return None

    def predict_age(self, audio_path):
        """
        Runs the full prediction pipeline on a single audio file.
        """
        if not self.is_loaded:
            return "Error: Models are not loaded."

        try:
            # 1. Extract features
            features = self.extract_features(audio_path)
            if features is None: return "Feature extraction failed."

            # 2. Scale features
            features_scaled = self.scaler.transform(features)

            # 3. Select best features
            features_selected = self.feature_selector.transform(features_scaled)

            # 4. Predict
            prediction = self.model.predict(features_selected)

            # 5. Decode the label
            age_group = self.label_encoder.inverse_transform(prediction)[0]
            return age_group
        except Exception as e:
            return f"Error during prediction: {str(e)}"

    def predict_from_file(self, audio_path):
        """
        Predicts age from a local audio file.
        """
        if not os.path.exists(audio_path):
            print(f"File not found: {audio_path}")
            return None
            
        print(f"\n--- Processing: {audio_path} ---")
        age_group = self.predict_age(audio_path)
        print(f"Predicted Age Group: '{age_group}'")
        return age_group

    def list_audio_files(self, directory='./audio_samples/'):
        """
        Lists available audio files in the specified directory.
        """
        audio_extensions = ['.wav', '.mp3', '.opus', '.flac', '.m4a']
        audio_files = []
        
        if os.path.exists(directory):
            for file in os.listdir(directory):
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    audio_files.append(os.path.join(directory, file))
        
        return audio_files

# --- MAIN EXECUTION ---
def main():
    # Set up directories
    model_directory = './models/'  # Directory containing your .sav files
    audio_directory = './audio_samples/'  # Directory containing audio files to test
    
    # Create directories if they don't exist
    os.makedirs(model_directory, exist_ok=True)
    os.makedirs(audio_directory, exist_ok=True)
    
    print(f"Model directory: {model_directory}")
    print(f"Audio directory: {audio_directory}")
    
    # Initialize predictor
    predictor = AudioPredictor(model_dir=model_directory)
    
    if not predictor.is_loaded:
        print("\nModels not loaded. Please ensure the following files are in the models/ directory:")
        print("  - final_model.sav")
        print("  - scaler.sav")
        print("  - feature_selector.sav")
        print("  - label_encoder.sav")
        return
    
    # List available audio files
    audio_files = predictor.list_audio_files(audio_directory)
    
    if not audio_files:
        print(f"\nNo audio files found in {audio_directory}")
        print("Please add some audio files (.wav, .mp3, .opus, .flac, .m4a) to test.")
        return
    
    print("\nAvailable audio files:")
    for i, file in enumerate(audio_files, 1):
        print(f"  {i}. {os.path.basename(file)}")
    
    print("\nChoose an option:")
    print("1. Test a specific file")
    print("2. Test all files")
    print("3. Enter custom file path")
    
    choice = input("Enter choice (1, 2, or 3): ")
    
    if choice == '1':
        try:
            file_num = int(input(f"Enter file number (1-{len(audio_files)}): ")) - 1
            if 0 <= file_num < len(audio_files):
                predictor.predict_from_file(audio_files[file_num])
            else:
                print("Invalid file number.")
        except ValueError:
            print("Please enter a valid number.")
    
    elif choice == '2':
        for audio_file in audio_files:
            predictor.predict_from_file(audio_file)
    
    elif choice == '3':
        custom_path = input("Enter the full path to your audio file: ")
        predictor.predict_from_file(custom_path)
    
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()