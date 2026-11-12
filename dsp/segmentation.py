import numpy as np
import librosa
from scipy.signal import find_peaks

def warmup_segmentation() -> None:
    dummy = np.zeros(2048, dtype=np.float32)
    librosa.effects.split(dummy, top_db=30)
    librosa.feature.rms(y=dummy, hop_length=256)

def split_into_n(
    path: str | None = None,
    n: int = 1,
    top_db: int = 30,
    hop: int = 256,
    *,
    y: np.ndarray | None = None,
    sr: int | None = None,
) -> list[tuple[float, float]]:
    if y is None or sr is None:
        if path is None:
            raise ValueError("Provide either path or (y, sr).")
        y, sr = librosa.load(path, sr=None)

    spans = librosa.effects.split(y, top_db=top_db)
    
    min_span_samples = int(0.05 * sr)
    valid_spans = [s for s in spans if (s[1] - s[0]) > min_span_samples]
    
    if len(valid_spans) == 0 or n < 1:
        return []
        
    start, end = valid_spans[0][0], valid_spans[-1][1]
    if n == 1:
        return [(start / sr, end / sr)]

    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms = np.convolve(rms, np.ones(5) / 5, mode="same") 
    f0, f1 = start // hop, end // hop
    seg = rms[f0:f1]

    min_gap = max(1, int(0.08 * sr / hop))
    valleys, props = find_peaks(-seg, distance=min_gap, prominence=0)
    
    if len(valleys) >= n - 1:
        best = valleys[np.argsort(props["prominences"])[::-1][: n - 1]]
        cuts = sorted(int(f0 + v) for v in best)
        bounds = [start] + [c * hop for c in cuts] + [end]
    else:
        bounds = list(np.linspace(start, end, n + 1))

    return [(bounds[i] / sr, bounds[i + 1] / sr) for i in range(n)]
