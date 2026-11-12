import numpy as np
import librosa
from scipy.signal import find_peaks


def split_into_n(path, n, top_db=30, hop=256):
    """Split a recording into exactly n syllable intervals (start, end) in seconds."""
    y, sr = librosa.load(path, sr=None)
    spans = librosa.effects.split(y, top_db=top_db)
    if len(spans) == 0 or n < 1:
        return []
    start, end = spans[0][0], spans[-1][1]
    if n == 1:
        return [(start / sr, end / sr)]

    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms = np.convolve(rms, np.ones(5) / 5, mode="same")  # smooth
    f0, f1 = start // hop, end // hop
    seg = rms[f0:f1]

    # deepest valleys inside the voiced span, at least ~80 ms apart
    min_gap = max(1, int(0.08 * sr / hop))
    valleys, props = find_peaks(-seg, distance=min_gap, prominence=0)
    if len(valleys) >= n - 1:
        best = valleys[np.argsort(props["prominences"])[::-1][: n - 1]]
        cuts = sorted(int(f0 + v) for v in best)
        bounds = [start] + [c * hop for c in cuts] + [end]
    else:
        bounds = list(np.linspace(start, end, n + 1))  # even split fallback

    return [(bounds[i] / sr, bounds[i + 1] / sr) for i in range(n)]
