import librosa
import numpy as np

def extract_pitch_contour(audio_path):
    """
    Extracts pitch and timestamps while preserving absolute time 
    so syllable interval masking works perfectly.
    """
    # 1. Load the audio (No trimming allowed here!)
    y, sr = librosa.load(audio_path, sr=None)

    # 2. Extract Pitch with Human Limits (fixes the 1100 Hz glitch)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=65, 
        fmax=500,
        sr=sr,
        frame_length=2048
    )

    # Replace NaNs with 0
    f0 = np.nan_to_num(f0)

    # 3. Generate the Time Array
    # THIS IS THE MISSING LINE! It creates the 'times' variable.
    times = librosa.times_like(f0, sr=sr)

    # 4. Z-Score Normalization (Equalizes deep vs high voices)
    voiced_f0 = f0[f0 > 0]
    if len(voiced_f0) > 0:
        mean_val = np.mean(voiced_f0)
        std_val = np.std(voiced_f0)
        
        if std_val > 0:
            f0[f0 > 0] = (f0[f0 > 0] - mean_val) / std_val
        else:
            f0[f0 > 0] = f0[f0 > 0] - mean_val

    # 5. Return exactly the two values app.py is looking for
    return times, f0
