import streamlit as st
import os
import time
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

from core.linguistics import analyze_phrase
from dsp.features import extract_pitch_contour, to_utterance_semitones
from dsp.segmentation import split_into_n, warmup_segmentation
from dsp.scoring import grade_lesson
from core.generator import generate_reference_audio
from core.visualizer import plot_tone_comparison
from wrappers import ASRWrapper

# --- 1. SYSTEM INITIALIZATION ---
@st.cache_resource
def init_system():
    asr_model = ASRWrapper(model_size="base")
    warmup_segmentation()  
    return asr_model

asr = init_system()
Path("samples").mkdir(exist_ok=True)

# --- SIDEBAR: TESTING CONTROLS ---
with st.sidebar:
    st.header("🧪 Testing Suite")
    test_mode = st.toggle("Enable Sample Mode", help="Switch from microphone to fixed test files.")
    
    uploaded_test_file = st.file_uploader("Add to Sample Library", type=['wav', 'webm'])
    if uploaded_test_file:
        sample_path = Path("samples") / uploaded_test_file.name
        with open(sample_path, "wb") as f:
            f.write(uploaded_test_file.getbuffer())
        st.success(f"Saved to library: {uploaded_test_file.name}")

    sample_files = [f.name for f in Path("samples").glob("*") if f.suffix in ['.wav', '.webm']]
    selected_sample = st.selectbox("Select Test Sample", options=sample_files) if sample_files else None

# --- MAIN UI ---
st.write("### 1. Teacher Reference")
practice_text = st.text_input("Enter Mandarin text to practice:", value="你好")

# Auto-generate teacher reference on load to prevent recruiter crashes
if not os.path.exists("teacher_ref.wav"):
    with st.spinner("Initializing Native Teacher Reference..."):
        generate_reference_audio(practice_text, "teacher_ref.wav")

if st.button("Generate Teacher Audio"):
    with st.spinner("Generating native pronunciation..."):
        generate_reference_audio(practice_text, "teacher_ref.wav")
    st.success("Target audio ready!")
    st.audio("teacher_ref.wav")

st.markdown("---")
st.write("### 2. Provide Your Attempt")

if test_mode and selected_sample:
    st.info(f"Using Sample: **{selected_sample}**")
    with open(Path("samples") / selected_sample, "rb") as f:
        audio_value = f.read() 
    st.audio(audio_value)
else:
    audio_value = st.audio_input("Speak into your microphone")

# --- 3. PROCESSING & GRADING SECTION ---
if audio_value is not None:
    if st.button("3. Grade My Pronunciation"):
        if not os.path.exists("teacher_ref.wav"):
            st.error("Please generate the teacher reference first!")
        else:
            with st.spinner("Analyzing pitch contours..."):
                t_start_perf = time.perf_counter()
                
                # 1. Clean and normalize audio
                with open("student_raw.wav", "wb") as f:
                    f.write(audio_value if isinstance(audio_value, bytes) else audio_value.getbuffer())
                
                y_raw, sr_raw = librosa.load("student_raw.wav", sr=None)
                y_norm = librosa.util.normalize(y_raw)
                y_trimmed, _ = librosa.effects.trim(y_norm, top_db=35) 
                sf.write("student_attempt.wav", y_trimmed, sr_raw)

                # 2. Extract Features
                t_y, t_sr = librosa.load("teacher_ref.wav", sr=None)
                s_y, s_sr = librosa.load("student_attempt.wav", sr=None)
                
                linguistic_data = analyze_phrase(practice_text)
                n_syllables = len(linguistic_data)

                t_times, t_f0 = extract_pitch_contour(y=t_y, sr=t_sr)
                s_times, s_f0 = extract_pitch_contour(y=s_y, sr=s_sr)
                t_st = to_utterance_semitones(t_f0)
                s_st = to_utterance_semitones(s_f0)

                # 3. Segment & Grade
                t_intervals = split_into_n(n=n_syllables, y=t_y, sr=t_sr)
                s_intervals = split_into_n(n=n_syllables, y=s_y, sr=s_sr)
                
                assessment = grade_lesson(s_st, t_st, s_intervals, t_intervals, s_times, t_times)
                
                t_end_perf = time.perf_counter()
                dsp_latency_ms = (t_end_perf - t_start_perf) * 1000
                
            # --- UI RENDERING ---
            st.markdown("---")
            if assessment.scored:
                st.header(f"🏆 Final Score: {assessment.overall_score:.1f}%")
                st.caption(f"⚡ Tone DSP analyzed in {dsp_latency_ms:.1f} ms • ASR verified asynchronously")
                
                st.write("#### Syllable Breakdown")
                cols = st.columns(len(assessment.syllable_scores))
                
                for i, (score, tip, ling) in enumerate(zip(assessment.syllable_scores, assessment.syllable_tips, linguistic_data)):
                    with cols[i]:
                        char_display = f"{ling['char']} ({ling['pinyin']})"
                        if score is not None:
                            status = "✅" if score > 80 else "⚠️" if score > 50 else "❌"
                            st.metric(label=char_display, value=f"{score:.1f}%", delta=status, delta_color="off")
                            if ling["sandhi"]:
                                st.caption(f"**Sandhi:** {ling['sandhi']}")
                            st.caption(f"_{tip}_") 
                        else:
                            st.metric(label=char_display, value="Unscored", delta="Too short", delta_color="off")
                            st.caption(f"_{tip}_")

                fig = plot_tone_comparison(assessment.viz_t, assessment.viz_u, assessment.overall_score)
                st.pyplot(fig)
            else:
                st.error(f"Assessment Failed: {assessment.reason}")

            # Whisper Sanity Check
            with st.status("Running Whisper Transcript Check (Background)...", expanded=False):
                transcription = asr.transcribe("student_attempt.wav")
                st.write(f"**Heard:** {transcription.get('text', 'Could not transcribe')}")
