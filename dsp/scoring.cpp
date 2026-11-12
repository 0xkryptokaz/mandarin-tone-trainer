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
    """
    # 1. Edge Case: Empty segment (user stopped speaking)
    if len(u_segment) < 5 or len(t_segment) < 5:
        return 0.0

    # 2. Normalize JUST this syllable (Local Normalization)
    # This fixes the issue where a high pitch on word 1 skews word 2.
    u_norm = normalize_pitch(u_segment)
    t_norm = normalize_pitch(t_segment)

    # 3. DTW Distance
    dist, path = fastdtw(u_norm, t_norm, dist=lambda x, y: np.abs(x - y))
    avg_dist = dist / len(path)
    
    # 4. Correlation (Shape)
    # Extract aligned paths
    u_aligned = np.array([u_norm[i] for i, j in path])
    t_aligned = np.array([t_norm[j] for i, j in path])
    
    try:
        corr, _ = pearsonr(u_aligned, t_aligned)
    except:
        corr = 0
    
    # 5. Strict Scoring Math
    dist_score = max(0, 100 * (1 - (avg_dist / 0.8)))
    shape_penalty = max(0, corr) ** 2
    
    return dist_score * shape_penalty

def grade_lesson(user_pitch, target_pitch, u_intervals, t_intervals, user_times, target_times):
    """
    The Production-Level Grader. 
    It doesn't just return a number; it returns a diagnostic report.
    """
    
    # 1. Rhythm Check
    # If syllable counts don't match, we can't align them 1:1.
    if len(u_intervals) != len(t_intervals):/
        print(f"⚠️ Rhythm Mismatch: Expected {len(t_intervals)}, got {len(u_intervals)}")
        # Fallback: Just score the whole file globally (Legacy mode)
        # In a real app, you might return a specific error code here.
        global_score, _, _, _ = calculate_segment_score(user_pitch, target_pitch), [], [], [] 
        return global_score * 0.5, [] # Massive penalty

    syllable_scores = []
    
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
        score = calculate_segment_score(u_seg, t_seg)
        syllable_scores.append(score)

    # 3. Final Weighted Average
    final_score = np.mean(syllable_scores)
    
    return final_score, syllable_scores




std::vector<double> u_aligned; //Results list
std::vector<double> t_aligned;

for (auto step: path) {
    int i = step.first;
    int j = step.second;

    u_aligned.push_back(u_norm[i]);
    t_aligned.push_back(t_norm[j]);
}

double sum = 0.0

for (double val: u_aligned) {
    sum += val;

}

double mean_u = sum / u_aligned.size();












