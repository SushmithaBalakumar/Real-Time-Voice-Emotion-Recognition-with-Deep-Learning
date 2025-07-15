from feature_extraction import extract_features

file_path = "data/sample_audio.wav"  # Place a sample audio file in `data/` folder
features = extract_features(file_path)

if features is not None:
    print(f"Extracted {len(features)} features:")
    print(features)
else:
    print("Failed to extract features.")
