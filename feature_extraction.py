import numpy as np
import librosa

def extract_features(audio_path_or_array, sr=22050, n_mels=128):
    try:
        if isinstance(audio_path_or_array, str):
            y, sr = librosa.load(audio_path_or_array, sr=sr)
        else:
            y = audio_path_or_array  # raw audio passed directly

        if len(y) < 1:
            return None

        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize to fixed shape (e.g. pad or crop)
        if log_mel_spec.shape[1] < 128:
            pad_width = 128 - log_mel_spec.shape[1]
            log_mel_spec = np.pad(log_mel_spec, ((0,0), (0, pad_width)), mode='constant')
        else:
            log_mel_spec = log_mel_spec[:, :128]

        return log_mel_spec  # shape: (128, 128)

    except Exception as e:
        print("Feature extraction error:", str(e))
        return None