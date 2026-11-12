import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

# --- 1. YOUR PROVIDED HELPER FUNCTIONS ---

def parabolic(f, x):
    """
    Quadratic interpolation for estimating the true position of an inter-sample maximum.
    (Missing from your snippet, but essential for accurate Hz)
    """
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)

def indexes(y, thres=0.3, min_dist=1, thres_abs=False):
    """Peak detection routine (Your provided code)"""
    if isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.unsignedinteger):
        raise ValueError("y must be signed")
    if not thres_abs:
        thres = thres * (np.max(y) - np.min(y)) + np.min(y)
    min_dist = int(min_dist)
    dy = np.diff(y)
    zeros, = np.where(dy == 0)
    if len(zeros) == len(y) - 1: return np.array([])
    
    if len(zeros):
        zeros_diff = np.diff(zeros)
        zeros_diff_not_one, = np.add(np.where(zeros_diff != 1), 1)
        zero_plateaus = np.split(zeros, zeros_diff_not_one)
        if zero_plateaus[0][0] == 0:
            dy[zero_plateaus[0]] = dy[zero_plateaus[0][-1] + 1]
            zero_plateaus.pop(0)
        if len(zero_plateaus) and zero_plateaus[-1][-1] == len(dy) - 1:
            dy[zero_plateaus[-1]] = dy[zero_plateaus[-1][0] - 1]
            zero_plateaus.pop(-1)
        for plateau in zero_plateaus:
            median = np.median(plateau)
            dy[plateau[plateau < median]] = dy[plateau[0] - 1]
            dy[plateau[plateau >= median]] = dy[plateau[-1] + 1]

    peaks = np.where(
        (np.hstack([dy, 0.0]) < 0.0) & 
        (np.hstack([0.0, dy]) > 0.0) & 
        (np.greater(y, thres))
    )[0]

    if peaks.size > 1 and min_dist > 1:
        highest = peaks[np.argsort(y[peaks])][::-1]
        rem = np.ones(y.size, dtype=bool)
        rem[peaks] = False
        for peak in highest:
            if not rem[peak]:
                sl = slice(max(0, peak - min_dist), peak + min_dist + 1)
                rem[sl] = True
                rem[peak] = False
        peaks = np.arange(y.size)[~rem]
    return peaks

def get_pitch_from_frame(signal, fs):
    """
    Calculates pitch for ONE small slice of audio.
    """
    signal = signal.astype(float)
    signal -= np.mean(signal)  # Remove DC offset
    
    # 1. Autocorrelation using FFT (Fast!)
    corr = fftconvolve(signal, signal[::-1], mode='full')
    corr = corr[len(corr)//2:]

    # 2. Find the first peak (The Fundamental Frequency)
    # We ignore the very first peak at 0 lag (correlation with self)
    try:
        # min_dist=5 ignores high freq noise
        found_peaks = indexes(corr, thres=0.4, min_dist=5) 
        
        # Filter out the peak at lag 0
        found_peaks = found_peaks[found_peaks > 10] 
        
        if len(found_peaks) == 0:
            return np.nan # Unvoiced / Silence

        i_peak = found_peaks[0] # The first major peak is the Pitch
        
        # 3. Refine using parabolic interpolation
        i_interp = parabolic(corr, i_peak)[0]
        pitch_hz = fs / i_interp
        
        # Sanity check for human speech (50Hz to 500Hz)
        if 50 < pitch_hz < 500:
            return pitch_hz
        else:
            return np.nan
            
    except Exception:
        return np.nan

# --- 2. THE MAIN LOOP (What you asked for) ---

def analyze_wav_file(wav_path):
    print(f"Loading {wav_path}...")
    # Load audio
    y, sr = librosa.load(wav_path, sr=16000)
    
    # Parameters for analysis
    frame_length = 2048 # ~128ms window
    hop_length = 512    # ~32ms step (Overlap)
    
    pitches = []
    times = []
    
    # "Frame Blocking" - Sliding a window over the audio
    for i in range(0, len(y) - frame_length, hop_length):
        frame = y[i : i + frame_length]
        
        # Run YOUR algorithm on this tiny slice
        pitch = get_pitch_from_frame(frame, sr)
        
        pitches.append(pitch)
        times.append(i / sr)
        
    return times, pitches

# --- 3. RUN IT ---
# Use the file we generated earlier!
wav_file = "target_audio.wav" 
times, pitches = analyze_wav_file(wav_file)

# Plotting
plt.figure(figsize=(10, 4))
plt.plot(times, pitches, 'bo', markersize=2, label="Detected Pitch")
plt.title(f"Pitch Contour of {wav_file} (Custom Autocorrelation)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.ylim(0, 400) # Speech range
plt.grid(True)
plt.legend()
plt.savefig("pitch_contour.png")

