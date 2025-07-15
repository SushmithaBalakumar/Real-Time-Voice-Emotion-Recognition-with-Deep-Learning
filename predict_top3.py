import torch
import numpy as np
from model import EmotionCNN
from feature_extraction import extract_features
from sklearn.preprocessing import LabelEncoder

# Define the emotion labels (same as training)
EMOTION_LABELS = [
    "angry", "calm", "disgust", "fearful",
    "happy", "neutral", "sad", "surprised"
]

# Load label encoder
le = LabelEncoder()
le.fit(EMOTION_LABELS)

# Load the trained model
model = EmotionCNN()
model.load_state_dict(torch.load("emotion_cnn_model.pt"))
model.eval()

# Load features from the recorded audio
features = extract_features("data/sample_audio.wav")  # or any .wav file

if features is not None:
    x_input = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(x_input)
        probabilities = torch.softmax(output, dim=1).squeeze().numpy()

        # Get top 3 predictions
        top3_idx = np.argsort(probabilities)[-3:][::-1]
        top3_labels = le.inverse_transform(top3_idx)
        top3_conf = probabilities[top3_idx] * 100

        print("\n🧠 Top 3 Predicted Emotions:")
        for i in range(3):
            print(f"{i+1}. {top3_labels[i].capitalize():<10} – {top3_conf[i]:.2f}%")

        print(f"\n🎯 Final Prediction: {top3_labels[0].upper()} ({top3_conf[0]:.2f}%)")

else:
    print("❌ Could not extract features from the audio.")
