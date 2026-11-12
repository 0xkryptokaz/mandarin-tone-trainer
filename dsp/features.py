import librosa
import numpy as np

def extract_pitch_contour(audio_path: str | None = None, *, y: np.ndarray | None = None, sr: int | None = None):
    if y is None or sr is None:
        if audio_path is None:
            raise ValueError("Provide either audio_path or (y, sr).")
        y, sr = librosa.load(audio_path, sr=None)

    f0_hz, _, _ = librosa.pyin(
        y,
        fmin=65,
        fmax=500,
        sr=sr,
        frame_length=2048,
    )
    times = librosa.times_like(f0_hz, sr=sr)
    return times, f0_hz

def to_utterance_semitones(f0_hz: np.ndarray) -> np.ndarray:
    f0 = np.asarray(f0_hz, dtype=float)
    semitones = np.full_like(f0, np.nan)
    voiced = np.isfinite(f0) & (f0 > 0)
    if not np.any(voiced):
        return semitones

    raw_st = 12.0 * np.log2(f0[voiced])
    semitones[voiced] = raw_st - np.mean(raw_st)
    return semitones
