import numpy as np
import pytest

from dsp.features import extract_pitch_contour, to_utterance_semitones
from dsp.scoring import grade_lesson
from dsp.segmentation import split_into_n


def _make_two_syllable_chirp(sr: int = 16000, f_start: float = 160.0, f_end: float = 240.0) -> np.ndarray:
    """Synthesize two voiced bursts separated by a brief silence valley."""
    t1 = np.linspace(0, 0.35, int(sr * 0.35), endpoint=False)
    # Syllable 1: falling-rising contour (Tone 3-like)
    freq1 = f_start - 30.0 * np.sin(np.pi * t1 / 0.35)
    phase1 = 2 * np.pi * np.cumsum(freq1) / sr
    syl1 = 0.6 * np.sin(phase1)

    gap = np.zeros(int(sr * 0.12))

    t2 = np.linspace(0, 0.35, int(sr * 0.35), endpoint=False)
    # Syllable 2: rising contour (Tone 2-like)
    freq2 = np.linspace(f_start, f_end, t2.size)
    phase2 = 2 * np.pi * np.cumsum(freq2) / sr
    syl2 = 0.6 * np.sin(phase2)

    return np.concatenate([syl1, gap, syl2]).astype(np.float32)


def test_dsp_pipeline_identical_speaker_scores_high():
    sr = 16000
    teacher_y = _make_two_syllable_chirp(sr=sr, f_start=180.0, f_end=260.0)
    student_y = _make_two_syllable_chirp(sr=sr, f_start=180.0, f_end=260.0)

    t_times, t_f0 = extract_pitch_contour(y=teacher_y, sr=sr)
    s_times, s_f0 = extract_pitch_contour(y=student_y, sr=sr)

    t_st = to_utterance_semitones(t_f0)
    s_st = to_utterance_semitones(s_f0)

    t_intervals = split_into_n(n=2, y=teacher_y, sr=sr)
    s_intervals = split_into_n(n=2, y=student_y, sr=sr)

    assert len(t_intervals) == 2
    assert len(s_intervals) == 2

    assessment = grade_lesson(s_st, t_st, s_intervals, t_intervals, s_times, t_times)
    assert assessment.scored is True
    assert assessment.overall_score >= 85.0
    assert len(assessment.syllable_scores) == 2


def test_speaker_pitch_shift_invariance():
    """A lower-pitched voice speaking the exact same contour should still score well."""
    sr = 16000
    teacher_y = _make_two_syllable_chirp(sr=sr, f_start=210.0, f_end=290.0)
    student_y = _make_two_syllable_chirp(sr=sr, f_start=130.0, f_end=180.0)

    t_times, t_f0 = extract_pitch_contour(y=teacher_y, sr=sr)
    s_times, s_f0 = extract_pitch_contour(y=student_y, sr=sr)

    assessment = grade_lesson(
        to_utterance_semitones(s_f0),
        to_utterance_semitones(t_f0),
        split_into_n(n=2, y=student_y, sr=sr),
        split_into_n(n=2, y=teacher_y, sr=sr),
        s_times,
        t_times,
    )
    assert assessment.scored is True
    assert assessment.overall_score >= 75.0


def test_database_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    import importlib
    import database

    importlib.reload(database)
    database.init_db()

    user = database.get_or_create_user("CI Tester", "English")
    database.log_attempt(user["id"], "你好", "nǐ hǎo", 88.5, [85.0, 92.0], "你好")
    history = database.get_user_history(user["id"])

    assert len(history) == 1
    assert history[0]["phrase"] == "你好"
    assert history[0]["score"] == pytest.approx(88.5)
