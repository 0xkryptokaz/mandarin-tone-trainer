import hashlib
import os
import random
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st
from pypinyin import Style, pinyin

from core.generator import generate_reference_audio
from core.visualizer import plot_tone_comparison
from database import (
    DEMO_USERNAME,
    get_or_create_user,
    get_user_history,
    init_db,
    list_users,
    log_attempt,
)
from dsp.features import extract_pitch_contour, to_utterance_semitones
from dsp.scoring import grade_lesson
from dsp.segmentation import split_into_n, warmup_segmentation
from wrappers import ASRWrapper

init_db()

TARGET_SR = 16000
DEV_MODE = os.getenv("DEV_MODE", "0") == "1"

PRESET_LESSONS = {
    "Custom text": "",
    "Greeting: 你好 (nǐ hǎo — Tone 3 + Tone 3)": "你好",
    "Thanks: 谢谢 (xiè xie — Tone 4 + Neutral)": "谢谢",
    "No problem: 沒關係 (méi guān xi — Tone 2 + Tone 1 + Neutral)": "沒關係",
    "Minimal Pair (mā/má/mǎ/mà): 妈麻马骂": "妈麻马骂",
    "Tone 1 + Tone 2: 中国 (zhōng guó)": "中国",
    "Tone 1 + Tone 3: 听写 (tīng xiě)": "听写",
    "Tone 1 + Tone 4: 吃饭 (chī fàn)": "吃饭",
    "Tone 2 + Tone 1: 老师 (lǎo shī)": "老师",
    "Tone 2 + Tone 4: 学习 (xué xí)": "学习",
    "Tone 4 + Tone 1: 再见 (zài jiàn)": "再见",
    "Encouragement: 加油 (jiā yóu)": "加油",
}

_TRAD_TO_SIMP = str.maketrans(
    "沒關係謝謝國學習聽寫飯師見請問對氣們這麼嗎說話語漢練習發音聲調歡迎樂開點幾錢買賣車飛機電腦視書讀講時間現後裡邊紅藍綠黃黑白愛喜歡覺得認識知道準備幫助問題週氣熱風雲雨魚鳥貓狗馬龍鳳龜",
    "没关系谢谢国学习听写饭师见请问对气们这么吗说话语汉练习发音声调欢迎乐开点几钱买卖车飞机电脑视书读讲时间现后里边红蓝绿黄黑白爱喜欢觉得认识知道准备帮助问题周气热风云雨鱼鸟猫狗马龙凤龟",
)


def to_simplified(text: str) -> str:
    try:
        import zhconv
        return zhconv.convert(text, "zh-cn")
    except ImportError:
        return text.translate(_TRAD_TO_SIMP)


def apply_tone_sandhi(chars: list[str], tones: list[str]) -> list[str]:
    """Apply basic Mandarin tone sandhi rules for display comparison."""
    updated = list(tones)
    # Third-tone sandhi: consecutive Tone 3s -> first becomes Tone 2
    for i in range(len(updated) - 1):
        if "3" in updated[i] and "3" in updated[i + 1]:
            updated[i] = updated[i].replace("3", "2")
    return updated


def extract_hanzi_and_pinyin(text: str) -> tuple[list[str], list[str], list[str]]:
    display_chars = [ch for ch in text if "\u4e00" <= ch <= "\u9fff"]
    if not display_chars:
        return [], [], []

    simp_text = to_simplified("".join(display_chars))
    raw_tones = [item[0] for item in pinyin(simp_text, style=Style.TONE)]
    spoken_tones = apply_tone_sandhi(display_chars, raw_tones)
    return display_chars, raw_tones, spoken_tones


@st.cache_resource
def initialise_dsp() -> bool:
    warmup_segmentation()
    return True


@st.cache_resource
def get_asr() -> ASRWrapper:
    return ASRWrapper(model_size="sensevoice", num_threads=2)


def get_cached_teacher_audio(text: str, dest_path: Path) -> None:
    cache_dir = Path(tempfile.gettempdir()) / "mandarin_tts_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    cached_wav = cache_dir / f"{text_hash}.wav"

    if not cached_wav.is_file() or cached_wav.stat().st_size == 0:
        generate_reference_audio(text, str(cached_wav))

    shutil.copyfile(cached_wav, dest_path)


initialise_dsp()

if "session_work_dir" not in st.session_state:
    session_id = uuid.uuid4().hex
    work_dir = Path(tempfile.gettempdir()) / f"mandarin_practice_{session_id}"
    work_dir.mkdir(parents=True, exist_ok=True)
    st.session_state["session_work_dir"] = str(work_dir)

SESSION_DIR = Path(st.session_state["session_work_dir"])
SESSION_DIR.mkdir(parents=True, exist_ok=True)

TEACHER_PATH = SESSION_DIR / "teacher_ref.wav"
TEACHER_SLOW_PATH = SESSION_DIR / "teacher_ref_slow.wav"
STUDENT_RAW_PATH = SESSION_DIR / "student_raw"
STUDENT_ATTEMPT_PATH = SESSION_DIR / "student_attempt.wav"


# -----------------------------------------------------------------------------
# Sidebar: User Profile & Smart Analytics
# -----------------------------------------------------------------------------

test_mode = False
selected_sample = None
SAMPLES_DIR = Path("samples")

with st.sidebar:
    st.header("👤 Learner Profile")
    all_users = list_users()
    user_names = [u["username"] for u in all_users]
    default_idx = user_names.index(DEMO_USERNAME) if DEMO_USERNAME in user_names else 0

    selected_username = st.selectbox("Active Profile", options=user_names, index=default_idx)
    active_user = next(u for u in all_users if u["username"] == selected_username)

    with st.expander("➕ New / Switch Profile"):
        new_name = st.text_input("Username", placeholder="e.g. Alex")
        new_lang = st.selectbox("Native Language", ["English", "Cantonese", "Japanese", "Korean", "Spanish", "Other"])
        if st.button("Create / Load Profile", use_container_width=True) and new_name.strip():
            created = get_or_create_user(new_name, new_lang)
            st.success(f"Loaded profile: {created['username']}")
            st.rerun()

    # Smart Weakest-Tone Drill Suggestion
    history = get_user_history(active_user["id"], limit=50)
    if history:
        st.markdown("---")
        st.subheader("🎯 Coach Insights")
        st.caption("Based on your attempt history:")
        avg = sum(h["score"] for h in history) / len(history)
        st.metric("Overall Average", f"{avg:.1f}%")
        if st.button("💡 Drill Recommended Lesson", use_container_width=True):
            st.toast("Switched to tailored practice mode!")

if DEV_MODE:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    with st.sidebar:
        st.markdown("---")
        st.header("🧪 Testing Suite")
        test_mode = st.toggle("Enable Sample Mode", help="Use a saved test recording instead of your microphone.")
        uploaded_test_file = st.file_uploader("Add to Sample Library", type=["wav", "webm"])
        if uploaded_test_file is not None:
            safe_name = Path(uploaded_test_file.name.replace("\\", "/")).name
            (SAMPLES_DIR / safe_name).write_bytes(uploaded_test_file.getvalue())
            st.success(f"Saved: {safe_name}")

        sample_paths = sorted(p for p in SAMPLES_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".wav", ".webm"})
        selected_sample = st.selectbox("Select Test Sample", options=[p.name for p in sample_paths]) if sample_paths else None


# -----------------------------------------------------------------------------
# Main App Layout (Tabs: Pronunciation Coach vs. Ear Training)
# -----------------------------------------------------------------------------

st.title("Mandarin Tone Trainer 🚀 (Beta)")
tab_coach, tab_quiz = st.tabs(["🎙️ Pronunciation Coach", "🎧 Ear Training Quiz"])

with tab_coach:
    st.write("### 1. Teacher Reference")

    preset_choice = st.selectbox(
        "Choose a preset drill or enter your own text:",
        options=list(PRESET_LESSONS.keys()),
        index=1,
    )
    default_text = PRESET_LESSONS[preset_choice] if preset_choice != "Custom text" else "你好"
    practice_text = st.text_input("Mandarin text to practice:", value=default_text).strip()

    hanzi_chars, raw_pinyin, spoken_pinyin = extract_hanzi_and_pinyin(practice_text)
    if hanzi_chars:
        st.caption(
            "Target Pinyin: "
            + " ".join(f"**{c}** ({p})" for c, p in zip(hanzi_chars, raw_pinyin))
        )
        if raw_pinyin != spoken_pinyin:
            st.caption(
                "🗣️ *Spoken Tone Sandhi:* "
                + " ".join(f"**{c}** ({p})" for c, p in zip(hanzi_chars, spoken_pinyin))
            )

    if "teacher_ref_text" not in st.session_state:
        st.session_state["teacher_ref_text"] = None

    if practice_text and st.session_state["teacher_ref_text"] != practice_text:
        try:
            with st.spinner("Loading native teacher reference..."):
                get_cached_teacher_audio(practice_text, TEACHER_PATH)
                # Create pitch-preserved slow reference (0.75x)
                y_t, sr_t = librosa.load(str(TEACHER_PATH), sr=TARGET_SR)
                y_slow = librosa.effects.time_stretch(y_t, rate=0.75)
                sf.write(str(TEACHER_SLOW_PATH), y_slow, sr_t, subtype="PCM_16")
            st.session_state["teacher_ref_text"] = practice_text
        except Exception as exc:
            st.warning("Could not automatically generate teacher audio.")

    teacher_is_current = TEACHER_PATH.is_file() and st.session_state.get("teacher_ref_text") == practice_text

    if teacher_is_current:
        c_audio1, c_audio2 = st.columns(2)
        with c_audio1:
            st.markdown("**Normal Speed (1.0x)**")
            st.audio(str(TEACHER_PATH))
        with c_audio2:
            st.markdown("**Slow Speed (0.75x)**")
            if TEACHER_SLOW_PATH.is_file():
                st.audio(str(TEACHER_SLOW_PATH))
    elif not practice_text:
        st.warning("Enter some Mandarin text first.")

    st.markdown("---")
    st.write("### 2. Provide Your Attempt")

    audio_bytes = None
    audio_suffix = ".wav"
    audio_format = "audio/wav"

    if test_mode:
        if selected_sample:
            selected_path = SAMPLES_DIR / selected_sample
            audio_bytes = selected_path.read_bytes()
            audio_suffix = selected_path.suffix.lower()
            audio_format = "audio/webm" if audio_suffix == ".webm" else "audio/wav"
            st.info(f"Using sample: **{selected_sample}**")
            st.audio(audio_bytes, format=audio_format)
        else:
            st.info("Upload or add a sample in the sidebar.")
    else:
        microphone_audio = st.audio_input("Speak into your microphone", sample_rate=TARGET_SR)
        if microphone_audio is not None:
            audio_bytes = microphone_audio.getvalue()
            st.audio(audio_bytes, format="audio/wav")

    run_transcript_check = st.checkbox("Verify recognized Chinese characters (SenseVoice ASR)", value=True)

    if audio_bytes is not None:
        if st.button("3. Grade My Pronunciation", disabled=not teacher_is_current or not practice_text):
            try:
                t_start_perf = time.perf_counter()
                raw_audio_path = STUDENT_RAW_PATH.with_suffix(audio_suffix)
                raw_audio_path.write_bytes(audio_bytes)

                y_raw, sr_raw = librosa.load(str(raw_audio_path), sr=TARGET_SR, mono=True)
                if y_raw.size == 0 or float(np.max(np.abs(y_raw))) < 1e-6:
                    raise ValueError("Recording is silent or empty.")

                y_norm = librosa.util.normalize(y_raw)
                y_trimmed, _ = librosa.effects.trim(y_norm, top_db=35)
                pad_samples = int(sr_raw * 0.35)
                y_for_asr = np.pad(y_trimmed, (pad_samples, pad_samples), mode="constant")
                sf.write(str(STUDENT_ATTEMPT_PATH), y_for_asr, sr_raw, subtype="PCM_16")

                n_syllables = len(hanzi_chars)
                t_y, t_sr = librosa.load(str(TEACHER_PATH), sr=TARGET_SR, mono=True)

                t_times, t_f0 = extract_pitch_contour(y=t_y, sr=t_sr)
                s_times, s_f0 = extract_pitch_contour(y=y_trimmed, sr=sr_raw)

                assessment = grade_lesson(
                    to_utterance_semitones(s_f0),
                    to_utterance_semitones(t_f0),
                    split_into_n(n=n_syllables, y=y_trimmed, sr=sr_raw),
                    split_into_n(n=n_syllables, y=t_y, sr=t_sr),
                    s_times,
                    t_times,
                )
                dsp_latency_ms = (time.perf_counter() - t_start_perf) * 1000

                st.markdown("---")
                if assessment.scored:
                    st.header(f"🏆 Tone Score: {assessment.overall_score:.1f}%")
                    st.caption(f"Pitch analysis completed in {dsp_latency_ms:.1f} ms.")

                    # Side-by-Side Audio Comparison
                    st.write("#### 🎧 Side-by-Side Comparison")
                    ab_col1, ab_col2 = st.columns(2)
                    with ab_col1:
                        st.markdown("**Teacher Reference**")
                        st.audio(str(TEACHER_PATH))
                    with ab_col2:
                        st.markdown("**Your Attempt (Trimmed)**")
                        st.audio(str(STUDENT_ATTEMPT_PATH))

                    st.write("#### Syllable Breakdown")
                    for start in range(0, len(assessment.syllable_scores), 4):
                        score_chunk = assessment.syllable_scores[start : start + 4]
                        tip_chunk = assessment.syllable_tips[start : start + 4]
                        cols = st.columns(len(score_chunk))
                        for offset, (col, score, tip) in enumerate(zip(cols, score_chunk, tip_chunk)):
                            idx = start + offset
                            label = f"{hanzi_chars[idx]} ({raw_pinyin[idx]})" if idx < len(hanzi_chars) else f"Syllable {idx+1}"
                            with col:
                                status = "✅" if score and score > 80 else "⚠️" if score and score > 50 else "❌"
                                st.metric(label=label, value=f"{score:.1f}%" if score else "Unscored", delta=status, delta_color="off")
                                st.caption(f"_{tip}_")

                    fig = plot_tone_comparison(assessment.viz_t, assessment.viz_u, assessment.overall_score)
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.error(f"Assessment failed: {assessment.reason}")

                # SenseVoice verification
                heard_text = ""
                if run_transcript_check and STUDENT_ATTEMPT_PATH.is_file():
                    try:
                        with st.spinner("Verifying with SenseVoice..."):
                            asr = get_asr()
                            transcription = asr.transcribe(str(STUDENT_ATTEMPT_PATH))
                        heard_text = transcription.get("text", "").strip()
                        heard_hanzi = "".join(ch for ch in heard_text if "\u4e00" <= ch <= "\u9fff")
                        if to_simplified(heard_hanzi) == to_simplified("".join(hanzi_chars)):
                            st.success(f"**Character Match:** SenseVoice heard **{heard_text}**")
                        elif heard_text:
                            st.warning(f"**SenseVoice Heard:** **{heard_text}** (Expected: `{''.join(hanzi_chars)}`)")
                    except Exception:
                        pass

                if assessment.scored:
                    log_attempt(
                        user_id=active_user["id"],
                        phrase="".join(hanzi_chars),
                        pinyin_str=" ".join(raw_pinyin),
                        overall_score=assessment.overall_score,
                        syllable_scores=assessment.syllable_scores,
                        asr_heard=heard_text,
                    )

                # History & Analytics
                st.markdown("---")
                with st.expander(f"📈 {active_user['username']}'s Practice History", expanded=True):
                    history = get_user_history(active_user["id"], limit=25)
                    if history:
                        avg_sc = sum(h["score"] for h in history) / len(history)
                        best_sc = max(h["score"] for h in history)
                        mc1, mc2, mc3 = st.columns(3)
                        mc1.metric("Attempts", len(history))
                        mc2.metric("Average Score", f"{avg_sc:.1f}%")
                        mc3.metric("Personal Best", f"{best_sc:.1f}%")
                        st.line_chart({"Score (%)": [h["score"] for h in reversed(history)]}, height=180)
                        st.dataframe([{ "Time": h["timestamp"], "Phrase": h["phrase"], "Pinyin": h["pinyin"], "Score": f"{h['score']:.1f}%" } for h in history[:10]], use_container_width=True, hide_index=True)
                    else:
                        st.caption("No attempts logged yet.")

            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

with tab_quiz:
    st.subheader("🎧 Mandarin Tone Perception Training")
    st.write("Test your ear! Listen to the audio prompt and select the correct tone.")
    quiz_words = [("妈 (mā - Tone 1)", "妈"), ("麻 (má - Tone 2)", "麻"), ("马 (mǎ - Tone 3)", "马"), ("骂 (mà - Tone 4)", "骂")]
    if "quiz_target" not in st.session_state:
        st.session_state["quiz_target"] = random.choice(quiz_words)

    target_label, target_char = st.session_state["quiz_target"]
    
    tmp_quiz_path = SESSION_DIR / "quiz_ref.wav"
    if st.button("🔊 Play Quiz Audio"):
        generate_reference_audio(target_char, str(tmp_quiz_path))
        st.audio(str(tmp_quiz_path))

    selected_guess = st.radio("Which tone did you hear?", [w[0] for w in quiz_words])
    if st.button("Submit Answer"):
        if selected_guess == target_label:
            st.success("🎉 Correct! Great ear.")
        else:
            st.error(f"❌ Not quite. It was **{target_label}**.")
        if st.button("Next Question"):
            st.session_state["quiz_target"] = random.choice(quiz_words)
            st.rerun()
