import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np

def record_audio(filename="student_attempt.wav", duration=3, fs=16000, device_index=9):
    """
    Records audio from a specific device. 
    Index 9 is the Elgato Wave:3 on this system.
    """
    print(f"🎤 Recording for {duration} seconds on device {device_index}... GO!")
    
    # We explicitly set the 'device' parameter to index 9
    recording = sd.rec(
        int(duration * fs), 
        samplerate=fs, 
        channels=1, 
        device=device_index
    )
    sd.wait()
    
    print("✅ Recording complete.")
    
    # Save to WAV file (convert float to 16-bit PCM)
    wav.write(filename, fs, (recording * 32767).astype(np.int16))
