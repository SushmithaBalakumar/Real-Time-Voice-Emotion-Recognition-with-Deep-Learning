import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from model import EmotionCNN
import matplotlib.pyplot as plt

# Load features and labels
X = np.load("features.npy")
y = np.load("labels.npy")

# Encode labels to integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_

# Split data (same as training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Convert to PyTorch tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Load the model and weights
model = EmotionCNN()
model.load_state_dict(torch.load("emotion_cnn_model.pt"))
model.eval()

# Predict on test set
all_preds = []
with torch.no_grad():
    for x in X_test_tensor:
        x = x.unsqueeze(0)  # Add batch dim
        output = model(x)
        predicted = torch.argmax(output, dim=1).item()
        all_preds.append(predicted)

# Confusion matrix
cm = confusion_matrix(y_test, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix - Emotion Recognition")
plt.tight_layout()
plt.show()
