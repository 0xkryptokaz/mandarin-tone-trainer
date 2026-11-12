import numpy as np
from fastdtw import fastdtw
from scipy.stats import pearsonr

# Keep your existing normalize_pitch function
def normalize_pitch(pitch_series):
    valid_pitch = pitch_series[pitch_series > 0]
    if len(valid_pitch) == 0:
        return np.zeros_like(pitch_series)
    
    mean_f0 = np.mean(valid_pitch)
    std_f0 = np.std(valid_pitch)
    if std_f0 == 0: std_f0 = 1.0

    normalized = (pitch_series - mean_f0) / std_f0
    normalized[pitch_series == 0] = 0
    return normalized

def calculate_segment_score(u_segment, t_segment):
    """
    Helper function: Scores a single syllable pair.
    Auto-aligns by stripping dead air before comparing shapes.
    """
    # 1. Strip leading and trailing silence (zeros) to auto-align the start times
    u_valid = np.trim_zeros(u_segment)
    t_valid = np.trim_zeros(t_segment)

    # 2. Edge Case: Empty segment (user stopped speaking or segment too short)
    if len(u_valid) < 5 or len(t_valid) < 5:
        return 0.0, np.zeros_like(u_segment), np.zeros_like(t_segment)

    # 3. Normalize JUST this aligned syllable (Local Normalization)
    u_norm = normalize_pitch(u_valid)
    t_norm = normalize_pitch(t_valid)

    # 3b. RE-TRIM normalized arrays (User's specific request to avoid noise)
    u_norm = np.trim_zeros(u_norm)
    t_norm = np.trim_zeros(t_norm)

    # Re-check length after re-trim
    if len(u_norm) < 5 or len(t_norm) < 5:
        return 0.0, np.zeros_like(u_norm), np.zeros_like(t_norm)

    # 4. DTW Distance (Now mathematically starting at the exact same index)
    dist, path = fastdtw(u_norm, t_norm, dist=lambda x, y: np.abs(x - y))
    avg_dist = dist / len(path)
    
    # 5. Correlation (Shape check)
    u_aligned = np.array([u_norm[i] for i, j in path])
    t_aligned = np.array([t_norm[j] for i, j in path])
    
    try:
        corr, _ = pearsonr(u_aligned, t_aligned)
        if np.isnan(corr):
            corr = 0.5 # Default to 0.5 for flat/monotone speech
    except:
        corr = 0.5 # Catch constant-array warnings
    
    # 6. Fairer Scoring Math
    # Relaxed the distance threshold slightly (1.0 instead of 0.8)
    dist_score = max(0, 100 * (1 - (avg_dist / 2.0)))
    
    # Softened shape penalty: Maps correlation from [-1, 1] to [0.1, 1.0]
    # This prevents an instant 0% just because you dipped slightly instead of rising
    shape_penalty = (corr + 1) / 2 if corr > -0.6 else 0.1
    
    return dist_score * shape_penalty, u_aligned, t_aligned

def grade_lesson(user_pitch, target_pitch, u_intervals, t_intervals, user_times, target_times):
    """
    The Production-Level Grader. 
    It doesn't just return a number; it returns a diagnostic report.
    """
    
    # 1. Rhythm Check
    # If syllable counts don't match, we can't align them 1:1.
    if len(u_intervals) != len(t_intervals):
        print(f"⚠️ Rhythm Mismatch: Expected {len(t_intervals)}, got {len(u_intervals)}")
        # Fallback: Just score the whole file globally (Legacy mode)
        # In a real app, you might return a specific error code here.
        # Global normalization for fallback
        u_norm = normalize_pitch(user_pitch)
        t_norm = normalize_pitch(target_pitch)
        dist, path = fastdtw(u_norm, t_norm, dist=lambda x, y: np.abs(x - y))
        u_aligned = np.array([u_norm[i] for i, j in path])
        t_aligned = np.array([t_norm[j] for i, j in path])
        
        # Simple global correlation
        try:
            corr, _ = pearsonr(u_aligned, t_aligned)
            if np.isnan(corr):
                corr = 0.5
        except:
            corr = 0.5
        avg_dist = dist / len(path)
        dist_score = max(0, 100 * (1 - (avg_dist / 2.0)))
        shape_penalty = (corr + 1) / 2 if corr > -0.6 else 0.1
        global_score = dist_score * shape_penalty

        return global_score, [], u_aligned, t_aligned 
    
    syllable_scores = []
    full_u_aligned = []
    full_t_aligned = []
    
    # 2. Loop through each syllable (The "Surgeon" Logic)
    for i in range(len(t_intervals)):
        # Extract the specific slice of audio for this syllable
        t_start, t_end = t_intervals[i]
        u_start, u_end = u_intervals[i]
        
        # Convert Time -> Array Indices
        # We find indices where 'times' array is within the start/end window
        t_mask = (target_times >= t_start) & (target_times <= t_end)
        u_mask = (user_times >= u_start) & (user_times <= u_end)
        
        t_seg = target_pitch[t_mask]
        u_seg = user_pitch[u_mask]
        
        # Score this specific chunk
        score, u_aligned, t_aligned = calculate_segment_score(u_seg, t_seg)
        syllable_scores.append(score)
        
        # Collect aligned data for visualization
        full_u_aligned.append(u_aligned)
        full_t_aligned.append(t_aligned)

    # 3. Final Weighted Average
    final_score = np.mean(syllable_scores)
    
    # Concatenate aligned segments for visualization
    viz_u = np.concatenate(full_u_aligned) if full_u_aligned else np.array([])
    viz_t = np.concatenate(full_t_aligned) if full_t_aligned else np.array([])
    
    return final_score, syllable_scores, viz_u, viz_t
