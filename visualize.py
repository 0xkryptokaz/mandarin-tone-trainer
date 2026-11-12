# app_visualizer.py
import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from dsp.features import extract_pitch_contour
from dsp.scoring import normalize_pitch

def generate_interview_plot(teacher_wav, student_wav, save_path="portfolio_plot.png"):
    print("Extracting features for visualization...")
    
    # 1. Extract actual pitch data using your fixed features.py
    t_times, t_pitch = extract_pitch_contour(teacher_wav)
    s_times, s_pitch = extract_pitch_contour(student_wav)
    
    # 2. Normalize to compare shapes (using your production logic)
    t_norm = normalize_pitch(t_pitch)
    s_norm = normalize_pitch(s_pitch)
    
    # 3. Calculate DTW Path
    dist, path = fastdtw(s_norm, t_norm, dist=lambda x, y: np.abs(x - y))
    
    # Unpack path for plotting
    student_indices = [p[0] for p in path]
    teacher_indices = [p[1] for p in path]
    
    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Mandarin Tone Analysis Engine", fontsize=16, fontweight='bold')

    # Panel 1: Pitch Contours (Time vs Pitch)
    ax1.plot(t_times, t_pitch, label='Teacher Reference', color='green', linewidth=2.5)
    ax1.plot(s_times, s_pitch, label='Student Attempt', color='blue', linestyle='--', linewidth=2)
    ax1.set_title("Raw Fundamental Frequency (F0) Extraction")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Pitch (Hz)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: The DTW Warp Path
    ax2.plot(student_indices, teacher_indices, 'b-', linewidth=2.5, label="Calculated Warp Path")
    ax2.plot([0, max(student_indices)], [0, max(teacher_indices)], 'r--', alpha=0.5, label="Perfect Speed Sync")
    ax2.set_title(f"Dynamic Time Warping (DTW) Alignment\nDistance: {dist:.2f}")
    ax2.set_xlabel("Student Time Steps")
    ax2.set_ylabel("Teacher Time Steps")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ Professional plot saved to {save_path}")

if __name__ == "__main__":
    # Ensure you have run main.py at least once to generate these files
    generate_interview_plot("teacher_ref.wav", "student_attempt.wav")
