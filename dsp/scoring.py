from dataclasses import dataclass
import numpy as np
from fastdtw import fastdtw
from scipy.ndimage import median_filter

FLAT_STD_ST = 0.35      
CONTOUR_STD_ST = 0.60   
MAX_DIST_ST = 5.0       

@dataclass
class LessonAssessment:
    scored: bool
    overall_score: float | None
    syllable_scores: list[float | None]
    syllable_tips: list[str]
    viz_u: np.ndarray
    viz_t: np.ndarray
    reason: str | None = None

def _prepare_segment(seg: np.ndarray, min_voiced: int = 3) -> np.ndarray | None:
    arr = np.asarray(seg, dtype=float)
    voiced_idx = np.flatnonzero(np.isfinite(arr))
    
    if len(voiced_idx) < min_voiced:
        return None

    trimmed = arr[voiced_idx[0] : voiced_idx[-1] + 1].copy()
    unvoiced = ~np.isfinite(trimmed)

    if np.any(unvoiced):
        x = np.arange(len(trimmed))
        trimmed[unvoiced] = np.interp(x[unvoiced], x[~unvoiced], trimmed[~unvoiced])

    if len(trimmed) >= 5:
        trimmed = median_filter(trimmed, size=3, mode="nearest")

    return trimmed

def _shape_factor(u_aligned: np.ndarray, t_aligned: np.ndarray) -> float:
    u_std = float(np.std(u_aligned))
    t_std = float(np.std(t_aligned))

    if u_std < FLAT_STD_ST and t_std < FLAT_STD_ST:
        return 1.0
    if t_std >= CONTOUR_STD_ST and u_std < FLAT_STD_ST:
        return 0.25
    if u_std >= CONTOUR_STD_ST and t_std < FLAT_STD_ST:
        return 0.35
    if u_std < 1e-6 or t_std < 1e-6:
        return 0.5

    corr = float(np.corrcoef(u_aligned, t_aligned)[0, 1])
    return (corr + 1.0) / 2.0 if corr > -0.6 else 0.1

def diagnose_syllable(u_aligned: np.ndarray, t_aligned: np.ndarray, score: float | None) -> str:
    if score is None or len(u_aligned) == 0 or len(t_aligned) == 0:
        return "Unvoiced or too short to diagnose."
        
    if score >= 85:
        return "Accurate contour and pitch range."

    onset_diff = u_aligned[0] - t_aligned[0]
    u_span = u_aligned[-1] - u_aligned[0]
    t_span = t_aligned[-1] - t_aligned[0]

    if abs(t_span) > 2.5 and abs(u_span) < 1.2:
        return "Contour too flat — exaggerate the pitch movement."
    if onset_diff < -3.0:
        return "Started too low before sliding into the tone."
    if onset_diff > 3.0:
        return "Started too high relative to your vocal midpoint."
    if (t_span > 2.0 and u_span < -1.0) or (t_span < -2.0 and u_span > 1.0):
        return "Pitch moved in the opposite direction of the target tone."
    
    return "Good overall direction; smooth out mid-syllable pitch wobble."

def calculate_segment_score(
    u_segment: np.ndarray,
    t_segment: np.ndarray,
) -> tuple[float | None, np.ndarray, np.ndarray]:
    u_clean = _prepare_segment(u_segment)
    t_clean = _prepare_segment(t_segment)
    
    if u_clean is None or t_clean is None:
        return None, np.array([]), np.array([])

    u_clean = u_clean - np.mean(u_clean)
    t_clean = t_clean - np.mean(t_clean)

    dist, path = fastdtw(u_clean, t_clean, dist=lambda x, y: abs(x - y))
    avg_dist = dist / len(path)

    u_aligned = np.array([u_clean[i] for i, _ in path])
    t_aligned = np.array([t_clean[j] for _, j in path])

    dist_score = max(0.0, 100.0 * (1.0 - avg_dist / MAX_DIST_ST))
    shape_factor = _shape_factor(u_aligned, t_aligned)

    if shape_factor < 0.45:
        final = dist_score * shape_factor
    else:
        final = 0.50 * dist_score + 0.50 * (shape_factor * 100.0)

    return float(final), u_aligned, t_aligned

def grade_lesson(
    user_pitch: np.ndarray,
    target_pitch: np.ndarray,
    u_intervals: list[tuple[float, float]],
    t_intervals: list[tuple[float, float]],
    user_times: np.ndarray,
    target_times: np.ndarray,
) -> LessonAssessment:
    if not t_intervals or len(u_intervals) != len(t_intervals):
        return LessonAssessment(
            scored=False,
            overall_score=None,
            syllable_scores=[],
            syllable_tips=[],
            viz_u=np.array([]),
            viz_t=np.array([]),
            reason=f"Interval mismatch: expected {len(t_intervals)} syllables, got {len(u_intervals)}."
        )

    syllable_scores: list[float | None] = []
    syllable_tips: list[str] = []
    full_u_aligned: list[np.ndarray] = []
    full_t_aligned: list[np.ndarray] = []

    for (u_start, u_end), (t_start, t_end) in zip(u_intervals, t_intervals):
        t_mask = (target_times >= t_start) & (target_times <= t_end)
        u_mask = (user_times >= u_start) & (user_times <= u_end)

        score, u_aligned, t_aligned = calculate_segment_score(
            user_pitch[u_mask],
            target_pitch[t_mask],
        )
        syllable_scores.append(score)
        syllable_tips.append(diagnose_syllable(u_aligned, t_aligned, score))

        if score is not None:
            full_u_aligned.append(u_aligned)
            full_t_aligned.append(t_aligned)

    processed_scores = [s if s is not None else 0.0 for s in syllable_scores]

    viz_u_separated: list[float] = []
    viz_t_separated: list[float] = []
    
    for u, t in zip(full_u_aligned, full_t_aligned):
        viz_u_separated.extend([*u, np.nan])
        viz_t_separated.extend([*t, np.nan])

    return LessonAssessment(
        scored=True,
        overall_score=float(np.mean(processed_scores)) if processed_scores else 0.0,
        syllable_scores=syllable_scores,
        syllable_tips=syllable_tips,
        viz_u=np.array(viz_u_separated),
        viz_t=np.array(viz_t_separated),
        reason=None,
    )
