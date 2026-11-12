from dsp.segmentation import split_into_n
from pypinyin import lazy_pinyin
from dsp.segmentation import split_into_n
from pypinyin import lazy_pinyin
import numpy as np
import argparse
import sys
from core.generator import generate_reference_audio
from core.recorder import record_audio
from core.fake_student import FakeStudentGenerator
from dsp.features import extract_pitch_contour
from dsp.scoring import grade_lesson
from core.visualizer import plot_tone_comparison
from wrappers import ASRWrapper

# Global ASR Instance
asr = ASRWrapper(model_size="base")

# Configuration
USE_TEST_MODE = True
TEST_ERROR_MODE = 'tone_high'  # Change this to test different scenarios
TEST_TTS_ENGINE = 'gtts'       # Options: 'gtts', 'gtts-tw', 'gtts-en'

def run_lesson(word_to_practice, error_mode=None, tts_engine=None):
    """
    Run a single pronunciation lesson.
    
    Args:
        word_to_practice: str - Chinese characters to practice
        error_mode: str - Override TEST_ERROR_MODE (for batch testing)
        tts_engine: str - Override TEST_TTS_ENGINE
    
    Returns:
        tuple: (final_score, breakdown) - Score and per-syllable breakdown
    """
    print(f"\n--- Lesson: {word_to_practice} ---")
    
    # Use overrides if provided, otherwise use globals
    current_error_mode = error_mode if error_mode is not None else TEST_ERROR_MODE
    current_tts_engine = tts_engine if tts_engine is not None else TEST_TTS_ENGINE
    
    # 1. Setup Teacher (The "Golden Record")
    teacher_file = "teacher_ref.wav"
    generate_reference_audio(word_to_practice, filename=teacher_file)
    t_times, t_pitch = extract_pitch_contour(teacher_file)
    
    # New ASR-based intervals for teacher
    t_transcription = asr.transcribe(teacher_file)
    t_intervals = split_into_n(teacher_file, len(lazy_pinyin(word_to_practice)))
    
    # 2. Get Student Input
    student_file = "student_attempt.wav"
    if USE_TEST_MODE:
        print(f"🤖 Test Mode: {current_error_mode}")
        FakeStudentGenerator.generate(
            chinese_text=word_to_practice,
            filename=student_file,
            error_mode=current_error_mode,
            tts_engine=current_tts_engine
        )
    else:
        print("🎤 Speak now...")
        record_audio(student_file, duration=4)
    
    # 3. Process Student
    s_times, s_pitch = extract_pitch_contour(student_file)
    
    # New ASR-based intervals for student
    s_transcription = asr.transcribe(student_file)
    s_intervals = split_into_n(student_file, len(lazy_pinyin(word_to_practice)))
    
    # 4. Grading (The Production Logic)
    final_score, breakdown, viz_u, viz_t = grade_lesson(
        s_pitch, t_pitch,
        s_intervals, t_intervals,
        s_times, t_times
    )
    
    # 5. Report Card
    print(f"\n🏆 Final Score: {final_score:.1f}%")
    if breakdown:
        print("📊 Breakdown:")
        for i, score in enumerate(breakdown):
            status = "✅" if score > 80 else "⚠️" if score > 50 else "❌"
            print(f"   Syllable {i+1}: {score:.1f}% {status}")
    
    # 6. Visualization
    fig = plot_tone_comparison(viz_t, viz_u, final_score)
    fig.savefig("tone_comparison.png")
    print(f"📊 Graph saved to tone_comparison.png")
    
    return final_score, breakdown


def run_test_suite(word_to_practice):
    """
    Run comprehensive tests across all error modes.
    Generates a report showing how the scoring system handles different errors.
    """
    print("\n" + "="*70)
    print(f"🧪 COMPREHENSIVE TEST SUITE: {word_to_practice}")
    print("="*70)
    
    results = {}
    
    # Test each error mode
    for mode in FakeStudentGenerator.ERROR_MODES.keys():
        print(f"\n{'─'*70}")
        print(f"Testing Error Mode: {mode}")
        print(f"Description: {FakeStudentGenerator.ERROR_MODES[mode]['description']}")
        print('─'*70)
        
        try:
            score, breakdown = run_lesson(word_to_practice, error_mode=mode)
            results[mode] = {
                'score': score,
                'breakdown': breakdown,
                'status': 'SUCCESS'
            }
        except Exception as e:
            print(f"❌ Error during test: {e}")
            results[mode] = {
                'score': 0.0,
                'breakdown': [],
                'status': f'FAILED: {str(e)}'
            }
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("📊 TEST SUITE SUMMARY")
    print("="*70)
    print(f"{'Error Mode':<25} {'Score':<10} {'Status':<15} {'Avg Syllable':<15}")
    print("─"*70)
    
    for mode, data in results.items():
        avg_syllable = np.mean(data['breakdown']) if data['breakdown'] else 0.0
        status_icon = "✅" if data['status'] == 'SUCCESS' else "❌"
        
        print(f"{mode:<25} {data['score']:>6.1f}%   {status_icon} {data['status']:<13} {avg_syllable:>6.1f}%")
    
    # Analysis
    print("\n" + "="*70)
    print("🔍 ANALYSIS")
    print("="*70)
    
    perfect_score = results.get('perfect', {}).get('score', 0)
    print(f"✓ Perfect pronunciation score: {perfect_score:.1f}%")
    
    if perfect_score < 70:
        print("  ⚠️  WARNING: Perfect mode scored low. System may be too strict.")
    elif perfect_score > 95:
        print("  ✓ GOOD: System recognizes correct pronunciation well.")
    
    wrong_scores = [data['score'] for mode, data in results.items() 
                    if 'wrong' in mode or 'high' in mode or 'low' in mode]
    if wrong_scores:
        avg_wrong = np.mean(wrong_scores)
        print(f"✓ Average score for wrong tones: {avg_wrong:.1f}%")
        
        if avg_wrong > 60:
            print("  ⚠️  WARNING: System may not be detecting wrong tones effectively.")
        else:
            print("  ✓ GOOD: System penalizes incorrect tones appropriately.")
    
    print("\n" + "="*70)


def run_engine_comparison(word_to_practice):
    """
    Compare different TTS engines with the same error mode.
    Useful for understanding how different accents affect scoring.
    """
    print("\n" + "="*70)
    print(f"🌐 TTS ENGINE COMPARISON: {word_to_practice}")
    print("="*70)
    
    results = {}
    error_mode = 'perfect'  # Use perfect mode to test engine differences
    
    for engine in FakeStudentGenerator.TTS_ENGINES.keys():
        print(f"\n{'─'*70}")
        print(f"Testing TTS Engine: {engine}")
        print('─'*70)
        
        try:
            score, breakdown = run_lesson(word_to_practice, 
                                        error_mode=error_mode,
                                        tts_engine=engine)
            results[engine] = {
                'score': score,
                'breakdown': breakdown
            }
        except Exception as e:
            print(f"❌ Error: {e}")
            results[engine] = {'score': 0.0, 'breakdown': []}
    
    # Summary
    print("\n" + "="*70)
    print("📊 ENGINE COMPARISON SUMMARY")
    print("="*70)
    print(f"{'TTS Engine':<20} {'Score':<10} {'Notes'}")
    print("─"*70)
    
    for engine, data in results.items():
        lang = FakeStudentGenerator.TTS_ENGINES[engine]
        print(f"{engine:<20} {data['score']:>6.1f}%   ({lang})")
    
    print("\n" + "="*70)


def quick_test(word_to_practice, modes_to_test=None):
    """
    Quick test of specific error modes.
    
    Args:
        word_to_practice: str - Chinese characters
        modes_to_test: list - Specific modes to test (default: ['perfect', 'tone_high', 'tone_low'])
    """
    if modes_to_test is None:
        modes_to_test = ['perfect', 'tone_high', 'tone_low']
    
    print("\n" + "="*70)
    print(f"⚡ QUICK TEST: {word_to_practice}")
    print("="*70)
    
    for mode in modes_to_test:
        if mode not in FakeStudentGenerator.ERROR_MODES:
            print(f"⚠️  Skipping unknown mode: {mode}")
            continue
        
        print(f"\n--- {mode} ---")
        score, _ = run_lesson(word_to_practice, error_mode=mode)
        print(f"Result: {score:.1f}%")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mandarin Pronunciation Research Tool")
    
    # Define the "Modes" of operation
    parser.add_argument('--mode', type=str, default='single', 
                        choices=['single', 'suite', 'engine', 'quick'],
                        help="The test mode to run")
    
    # Define the input text
    parser.add_argument('--text', type=str, default="你好", 
                        help="The Chinese text to practice")
    
    # Define the error type (for single or quick mode)
    parser.add_argument('--error', type=str, default='perfect',
                        help="The error mode for the FakeStudent (e.g., tone_high, tone_low)")

    args = parser.parse_args()

    if args.mode == 'single':
        run_lesson(args.text, error_mode=args.error)
    
    elif args.mode == 'suite':
        run_test_suite(args.text)
        
    elif args.mode == 'engine':
        run_engine_comparison(args.text)
        
    elif args.mode == 'quick':
        quick_test(args.text)
