import sounddevice as sd
from scipy.io.wavfile import write

def record_voice(filename="data/sample_audio.wav", duration=5, sample_rate=44100):
    print("🎤 Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    write(filename, sample_rate, audio)
    print(f"✅ Recording saved to {filename}")

if __name__ == "__main__":
    record_voice()
