# 🈯 Mandarin Tone Trainer (Prosody-AI)

A high-performance **Pronunciation Assessment (PPA)** engine designed to provide syllable-level feedback for Mandarin Chinese learners. This project analyzes the **Acoustic DNA** of tones through Fundamental Frequency ($F_0$) extraction and Dynamic Time Warping.

---

## 🚀 Key Features

* **Adaptive Syllable Segmentation:** Uses energy envelope analysis with adaptive noise-floor detection to slice audio into distinct phonological units.
* **Multiplicative Scoring Engine:** Implements strict scoring math combining Z-score normalized Euclidean distance with Pearson Correlation ($r^2$) to punish incorrect tonal shapes.
* **Time-Elastic Alignment:** Leverages **Dynamic Time Warping (DTW)** to fairly compare a student's attempt against a native teacher, regardless of speaking speed.
* **Diagnostic Feedback:** Provides a granular "Report Card" breaking down accuracy for every individual syllable in a sentence.
* **Developer Test Suite:** Includes a "Fake Student" module to simulate English-accented Mandarin for algorithmic verification.

---

## 🛠️ Technical Architecture

The system is built as a modular pipeline to ensure **Microlatency** and ease of future **Edge Deployment**.



### Data Pipeline:
1.  **Extraction:** Pitch tracking using Autocorrelation and RMS energy extraction.
2.  **Transformation:** Pitch values converted to relative semitones (Z-scores) to ensure speaker-invariance (grading the tone, not the voice deepness).
3.  **Alignment:** $O(N \cdot M)$ DTW pathfinding to synchronize non-linear time series.
4.  **Analysis:** Statistical correlation used as a confidence multiplier for final scoring.

---

## 📂 Project Structure

```text
.
├── core/
│   ├── generator.py     # Teacher audio synthesis (gTTS/Pypinyin)
│   ├── recorder.py      # Local audio capture
│   ├── visualizer.py    # Matplotlib-based pitch & segment plotting
│   └── fake_student.py  # ACC/CI Test Mode for English accents
├── dsp/
│   ├── features.py      # Pitch ($F_0$) and Energy (RMS) extraction
│   ├── segmentation.py  # Adaptive energy envelope syllable splitting
│   └── scoring.py       # DTW and Pearson Correlation logic
└── main.py              # Orchestration layer and lesson CLI
