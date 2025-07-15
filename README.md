Sushmitha
AI & ML Intern – IIITDM Kancheepuram
LinkedIn | GitHub

# 🎙️ Real-Time Voice Emotion Recognition

This project implements an end-to-end deep learning pipeline to recognize emotions from live voice input using Python and PyTorch. Built as part of my AI/ML internship at IIITDM Kancheepuram.

---

## 📌 Features

- 🔊 Records audio using microphone in real-time
- 🧠 Extracts Log-Mel Spectrogram features using `librosa`
- 🧪 Trains a custom 2D CNN model from scratch (no pre-trained models)
- 🎯 Predicts 8 emotions: `angry`, `calm`, `disgust`, `fearful`, `happy`, `neutral`, `sad`, `surprised`
- 📊 Visualizes training loss and accuracy
- 🛡️ Confidence threshold & Top-3 predictions shown for better reliability

---

## 🧠 Model Overview

- Input: Log-Mel Spectrograms (128×128)
- Architecture: 3-layer CNN with BatchNorm, Dropout
- Framework: PyTorch

---

## 📁 Dataset

- 🎭 [RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)](https://zenodo.org/record/1188976)
- 1440 `.wav` speech samples across 8 emotions
- Optional: Custom user voice samples for personalization

---

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/voice-emotion-recognition.git
cd voice-emotion-recognition
conda create -n voiceenv python=3.10
conda activate voiceenv
pip install -r requirements.txt
