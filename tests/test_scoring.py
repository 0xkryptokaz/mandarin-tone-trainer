import numpy as np
import pytest
from dsp.features import to_utterance_semitones
from dsp.scoring import calculate_segment_score, grade_lesson

# Frozen syllable-2 (hao3) contours from teacher_ref.wav / student_attempt.wav
T_BELOW = np.array([0.155, -0.243, -0.597, -0.837, -0.928, -0.986, -0.986, -1.044,
                    -1.086, -1.127, -1.181, -1.194, -1.086, -0.744, -0.425])
U_BELOW = np.array([0.605, 0.147, -0.163, -0.411, -0.675, -0.839, -0.948, -0.993,
                    -1.008, -1.052, -1.096, -1.139, -1.167, -1.209, -1.25, -1.222,
                    -1.139, -0.901, -0.641])


def test_below_mean_syllable_scores_high():
    score, _, _ = calculate_segment_score(U_BELOW, T_BELOW)
    assert score is not None and score > 80


def test_inverted_contour_scores_low():
    score, _, _ = calculate_segment_score(-U_BELOW, T_BELOW)
    assert score is not None and score < 40


def test_identical_recording_scores_100():
    f0 = np.array([np.nan, 180.0, 195.0, 215.0, 240.0, 260.0, np.nan])
    st = to_utterance_semitones(f0)
    score, _, _ = calculate_segment_score(st, st)
    assert score == pytest.approx(100.0)


@pytest.mark.parametrize("semitone_shift", [5.0, -5.0])
def test_speaker_register_invariance(semitone_shift):
    teacher_hz = np.array([np.nan, 150.0, 165.0, 185.0, 210.0, 230.0, np.nan])
    student_hz = teacher_hz * (2.0 ** (semitone_shift / 12.0))

    t_st = to_utterance_semitones(teacher_hz)
    u_st = to_utterance_semitones(student_hz)
    score, _, _ = calculate_segment_score(u_st, t_st)
    assert score == pytest.approx(100.0, abs=1e-6)


def test_rising_vs_flat_discriminates():
    rising = np.linspace(-3.0, 3.0, 12)
    flat = np.zeros(12)
    score, _, _ = calculate_segment_score(flat, rising)
    assert score is not None and score < 35


def test_rising_vs_falling_discriminates():
    rising = np.linspace(-3.0, 3.0, 12)
    falling = np.linspace(3.0, -3.0, 12)
    score, _, _ = calculate_segment_score(falling, rising)
    assert score is not None and score < 20


def test_internal_unvoiced_gap_handled():
    t_st = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    u_st = np.array([-3.0, -2.0, np.nan, 0.0, 1.0, 2.0, 3.0])
    score, _, _ = calculate_segment_score(u_st, t_st)
    assert score is not None and score > 95


def test_entirely_unvoiced_returns_unscored():
    unvoiced = np.full(10, np.nan)
    target = np.linspace(-2.0, 2.0, 10)
    score, _, _ = calculate_segment_score(unvoiced, target)
    assert score is None


def test_constant_f0_edge_case():
    const_hz = np.array([np.nan, 210.0, 210.0, 210.0, 210.0, 210.0, np.nan])
    st = to_utterance_semitones(const_hz)
    score, _, _ = calculate_segment_score(st, st)
    assert score == pytest.approx(100.0)

def test_interval_mismatch_returns_unscored():
        times = np.linspace(0.0, 1.0, 20)
        pitch = np.linspace(-2.0, 2.0, 20)
        res = grade_lesson(pitch, pitch, [(0.0, 0.4)], [(0.0, 0.4), (0.5, 0.9)], times, times)
        assert res.scored is False
        assert res.overall_score is None
        assert "interval mismatch" in (res.reason or "").lower()
