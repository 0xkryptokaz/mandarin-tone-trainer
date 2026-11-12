import matplotlib.pyplot as plt
import numpy as np


import matplotlib.pyplot as plt
import numpy as np

def plot_tone_comparison(viz_t, viz_u, score):
    """
    Plots the Teacher's normalized pitch vs. the Student's aligned normalized pitch.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Teacher (The Goal)
    ax.plot(viz_t, label='Teacher (Reference)', color='green', linewidth=2)
    
    # Plot Student (The Attempt)
    ax.plot(viz_u, label='Student (Aligned)', color='blue', linestyle='--', linewidth=2)
    
    ax.set_title(f"Tone Comparison (Score: {score:.2f}%)")
    ax.set_xlabel("Time (DTW Aligned Frames)")
    ax.set_ylabel("Normalized Pitch (Standard Deviations)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Streamlit compatibility: return the figure object
    return fig
