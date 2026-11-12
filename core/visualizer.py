import matplotlib.pyplot as plt
import numpy as np

def plot_tone_comparison(viz_t: np.ndarray, viz_u: np.ndarray, final_score: float):
    """
    Plots the aligned semitone contours cleanly, without connecting distinct syllables.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if len(viz_t) > 0 and len(viz_u) > 0:
        # NaNs in the array will naturally break the line between syllables
        ax.plot(viz_t, label="Teacher (Target)", color="#2e7d32", linewidth=2.5, alpha=0.9)
        ax.plot(viz_u, label="Student (You)", color="#1976d2", linewidth=2.5, linestyle="--", alpha=0.9)
    
    ax.set_title(f"Tone Contour Comparison (Score: {final_score:.1f}%)", fontsize=14, pad=15)
    ax.set_ylabel("Pitch (Relative Semitones)", fontsize=11)
    ax.set_xlabel("Time (Aligned Frames)", fontsize=11)
    
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    return fig
