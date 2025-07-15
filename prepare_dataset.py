import os
import numpy as np
from collections import Counter
from feature_extraction import extract_features

# Emotion code mapping based on RAVDESS
EMOTION_LABELS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def get_emotion_label(filename):
    # RAVDESS file format: '03-01-05-01-01-01-01.wav'
    parts = filename.split('-')
    emotion_code = parts[2]
    return EMOTION_LABELS.get(emotion_code, "unknown")

def build_dataset(data_dir):
    X, y = [], []

    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav"):
                path = os.path.join(root, file)
                label = get_emotion_label(file)

                features = extract_features(path)
                if features is not None and label != "unknown":
                    X.append(features)
                    y.append(label)

    return np.array(X), np.array(y)

if __name__ == "__main__":
    X, y = build_dataset("data/Audio_Speech_Actors_01-24")

    print(f"Dataset shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    from collections import Counter
    print(f"Sample label distribution: {Counter(y)}")

    # Save for training
    np.save("features.npy", X)
    np.save("labels.npy", y)
    print("Saved features and labels to .npy files")

