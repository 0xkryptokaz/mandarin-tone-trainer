import argparse
import sys
import numpy as np
import librosa
from pypinyin import lazy_pinyin

from dsp.features import extract_pitch_contour, to_utterance_semitones
from dsp.segmentation import split_into_n, warmup_segmentation
from dsp.scoring import grade_lesson
from core.generator import generate_reference_audio
from core.recorder import record_audio
from core.fake_student import FakeStudentGenerator
from core.visualizer import plot_tone_comparison
from wrappers import ASRWrapper

# Global ASR Instance & Warmup
asr = ASRWrapper(model_size="base")
warmup_segmentation()

USE_TEST_MODE = True
TEST_ERROR_MODE = 'tone_high'
TEST_TTS_ENGINE = 'gtts'

def run_lesson(word_to_practice, error_mode=None, tts_engine=None):
    print(f"\n--- Lesson: {word_to_practice} ---")
    current_error_mode = error_mode or TEST_ERROR_MODE
    current_tts_engine = tts_engine or TEST_TTS_ENGINE
    
    # 1. Setup Teacher
    teacher_file = "teacher_ref.wav"
    generate_reference_audio(word_to_practice, filename=teacher_file)
    t_y, t_sr = librosa.load(teacher_file, sr=None)
    t_times, t_f0 = extract_pitch_contour(y=t_y, sr=t_sr)
    t_st = to_utterance_semitones(t_f0)
    t_intervals = split_into_n(n=len(lazy_pinyin(word_to_practice)), y=t_y, sr=t_sr)
    
    # 2. Get Student Input
    student_file = "student_attempt.wav"
    if USE_TEST_MODE:
        print(f"🤖 Test Mode: {current_error_mode}")
        FakeStudentGenerator.generate(
            chinese_text=word_to_practice, filename=student_file,
            error_mode=current_error_mode, tts_engine=current_tts_engine
        )
    else:
        print("🎤 Speak now...")
        record_audio(student_file, duration=4)
    
    # 3. Process Student
    s_y, s_sr = librosa.load(student_file, sr=None)
    s_times, s_f0 = extract_pitch_contour(y=s_y, sr=s_sr)
    s_st = to_utterance_semitones(s_f0)
    s_intervals = split_into_n(n=len(lazy_pinyin(word_to_practice)), y=s_y, sr=s_sr)
    
    # 4. Grading
    assessment = grade_lesson(s_st, t_st, s_intervals, t_intervals, s_times, t_times)
    
    if not assessment.scored:
        print(f"❌ Assessment Failed: {assessment.reason}")
        return 0.0, []

    # 5. Report Card
    print(f"\n🏆 Final Score: {assessment.overall_score:.1f}%")
    for i, score in enumerate(assessment.syllable_scores):
        if score is not None:
            status = "✅" if score > 80 else "⚠️" if score > 50 else "❌"
            print(f"   Syllable {i+1}: {score:.1f}% {status}")
        else:
            print(f"   Syllable {i+1}: Unscored")
    
    # 6. Visualization
    fig = plot_tone_comparison(assessment.viz_t, assessment.viz_u, assessment.overall_score)
    fig.savefig("tone_comparison.png")
    
    return assessment.overall_score, [s for s in assessment.syllable_scores if s is not None]

# ... Keep the rest of your main.py (run_test_suite, run_engine_comparison, quick_test) identically here ...
