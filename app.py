from dsp.segmentation import split_into_n
from pypinyin import lazy_pinyin
from dsp.segmentation import split_into_n
from pypinyin import lazy_pinyin
import streamlit as st
import os
import librosa              # <-- ADD THIS LINE
import soundfile as sf      # <-- ADD THIS LINE
from core.generator import generate_reference_audio
from dsp.features import extract_pitch_contour
from dsp.scoring import grade_lesson
from core.visualizer import plot_tone_comparison
from wrappers import ASRWrapper
import shutil
from pathlib import Path

# --- CACHED ASR INITIALIZATION ---
@st.cache_resource
def get_asr_model():
    return ASRWrapper(model_size="base")

asr = get_asr_model()

# Create samples directory if it doesn't exist
Path("samples").mkdir(exist_ok=True)

# --- SIDEBAR: TESTING CONTROLS ---
with st.sidebar:
    st.header("🧪 Testing Suite")
    test_mode = st.toggle("Enable Sample Mode", help="Switch from microphone to fixed test files.")
    
    # Persistent Sample Storage
    uploaded_test_file = st.file_uploader("Add to Sample Library", type=['wav', 'webm'])
    if uploaded_test_file:
        sample_path = Path("samples") / uploaded_test_file.name
        with open(sample_path, "wb") as f:
            f.write(uploaded_test_file.getbuffer())
        st.success(f"Saved to library: {uploaded_test_file.name}")

    # Select from existing samples
    sample_files = [f.name for f in Path("samples").glob("*") if f.suffix in ['.wav', '.webm']]
    selected_sample = st.selectbox("Select Test Sample", options=sample_files) if sample_files else None



st.write("### 1. Teacher Reference")
practice_text = st.text_input("Enter Mandarin text to practice:", value="你好")

if st.button("Generate Teacher Audio"):
    with st.spinner("Generating native pronunciation..."):
        generate_reference_audio(practice_text, "teacher_ref.wav")
    st.success("Target audio ready!")
    st.audio("teacher_ref.wav")

st.markdown("---")
# --- UPDATED RECORDING LOGIC ---
st.write("### 2. Provide Your Attempt")
if test_mode and selected_sample:
    st.info(f"Using Sample: **{selected_sample}**")
    # Load the bytes from the saved file to simulate the audio_input behavior
    with open(Path("samples") / selected_sample, "rb") as f:
        audio_value = f.read() 
    st.audio(audio_value)
else:
    audio_value = st.audio_input("Speak into your microphone")# 3. Processing & Grading Section
if audio_value is not None:
    if st.button("3. Grade My Pronunciation"):
        if not os.path.exists("teacher_ref.wav"):
            st.error("Please generate the teacher reference first!")
        else:
            with st.spinner("Analyzing Acoustic DNA and aligning time series..."):
                
                # 1. Write the raw browser buffer to disk
                with open("student_raw.wav", "wb") as f:
                    f.write(audio_value.getbuffer())

                # 2. Load the audio (FFmpeg will now safely decode the browser's WebM format)
                y_raw, sr_raw = librosa.load("student_raw.wav", sr=None)
                
                # 3. Trim the leading/trailing silence (The Fundamental Fix)
                y_trimmed, _ = librosa.effects.trim(y_raw, top_db=25) 
                
                # 4. Save as a TRUE standard WAV file for Whisper and the Grader
                sf.write("student_attempt.wav", y_trimmed, sr_raw)

                # --- Run the DSP Pipeline ---
                # Teacher Data
                t_times, t_pitch = extract_pitch_contour("teacher_ref.wav")
                t_transcription = asr.transcribe("teacher_ref.wav")
                t_intervals = split_into_n("teacher_ref.wav", len(lazy_pinyin(practice_text)))
                
                # Student Data (Now using the clean, trimmed WAV!)
                s_times, s_pitch = extract_pitch_contour("student_attempt.wav")
                s_transcription = asr.transcribe("student_attempt.wav")
                s_intervals = split_into_n("student_attempt.wav", len(lazy_pinyin(practice_text)))
                
                # Scoring Engine
                final_score, breakdown, viz_u, viz_t = grade_lesson(
                    s_pitch, t_pitch, 
                    s_intervals, t_intervals, 
                    s_times, t_times
                )
                
                # --- UI Rendering ---
                st.markdown("---")
                st.header(f"🏆 Final Score: {final_score:.1f}%")
                
                # Syllable Breakdown Cards
                if breakdown:
                    st.write("#### Syllable Breakdown")
                    cols = st.columns(len(breakdown))
                    for i, score in enumerate(breakdown):
                        status = "✅" if score > 80 else "⚠️" if score > 50 else "❌"
                        with cols[i]:
                            st.metric(label=f"Syllable {i+1}", value=f"{score:.1f}%", delta=status, delta_color="off")

                # Visualization Plot
                fig = plot_tone_comparison(viz_t, viz_u, final_score)
                st.pyplot(fig)
