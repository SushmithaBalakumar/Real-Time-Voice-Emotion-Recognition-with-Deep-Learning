import torch
from sklearn.preprocessing import LabelEncoder
from model import EmotionCNN
from feature_extraction import extract_features

# Label encoder
EMOTION_LABELS = [
    "angry", "calm", "disgust", "fearful",
    "happy", "neutral", "sad", "surprised"
]
le = LabelEncoder()
le.fit(EMOTION_LABELS)

# Load trained model
model = EmotionCNN()
model.load_state_dict(torch.load("emotion_cnn_model.pt"))
model.eval()

# Load features from your recorded audio file
features = extract_features("data/sample_audio.wav")

if features is not None:
    x_input = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(x_input)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)
        predicted_emotion = le.inverse_transform([predicted_idx.item()])[0]

        # Set a confidence threshold (try 0.70 or 0.75)
        threshold = 0.7
        if confidence.item() > threshold:
            print(f"🧠 Predicted Emotion: {predicted_emotion.upper()} with {confidence.item() * 100:.2f}% confidence")
        else:
            print("😐 Low confidence. Please speak again or more clearly.")
else:
    print("❌ Could not extract features.")
