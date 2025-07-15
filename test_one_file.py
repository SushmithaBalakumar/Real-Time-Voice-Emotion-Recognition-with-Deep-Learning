import os
from feature_extraction import extract_features

path = "data/Audio_Speech_Actors_01-24/Actor_01"

for file in os.listdir(path):
    if file.endswith(".wav"):
        print("Found:", file)
        full_path = os.path.join(path, file)
        features = extract_features(full_path)
        print("Extracted features shape:", features.shape if features is not None else "None")
        break
else:
    print("No .wav files found in Actor_01.")
