"""Checks on waveform pairs, especially the alignment that makes them usable."""

import numpy as np
import pytest

from mp3ml.waveform import (
    SilentAudioError,
    align,
    estimate_lag,
    make_pair,
    relative_error,
)

SAMPLERATE = 44100


@pytest.fixture(scope="module")
def music() -> np.ndarray:
    """Three seconds of broadband, transient-rich audio — enough to lock a lag onto."""
    rng = np.random.default_rng(0)
    t = np.arange(3 * SAMPLERATE) / SAMPLERATE
    x = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 1310 * t)
    x += 0.05 * rng.standard_normal(len(t))
    x[:: SAMPLERATE // 4] += 0.6
    return np.round(np.clip(x, -1, 1) * 32767).astype(np.int16)


@pytest.mark.parametrize("true_lag", [0, 1, 137, 2210, -50])
def test_estimate_lag_recovers_a_known_shift(music: np.ndarray, true_lag: int) -> None:
    pad = 5000
    padded = np.concatenate([np.zeros(pad, dtype=np.int16), music])
    shifted = padded[pad - true_lag :] if true_lag <= pad else padded
    assert estimate_lag(music, shifted) == true_lag


def test_align_produces_equal_length(music: np.ndarray) -> None:
    for lag in (0, 500, -500):
        out = align(music, music[100:], lag)
        assert len(out) == len(music)


def test_align_undoes_the_shift(music: np.ndarray) -> None:
    delayed = np.concatenate([np.zeros(300, dtype=np.int16), music])
    restored = align(music, delayed, 300)
    np.testing.assert_array_equal(restored, music)


@pytest.mark.parametrize("bitrate", [64, 32, 16])
def test_round_trip_is_aligned_and_sane(music: np.ndarray, bitrate: int, tmp_path) -> None:
    """The stored pair must be sample-aligned and much closer than a naive pairing.

    Below 32 kbit/s LAME resamples and the resampler's delay is not covered by the
    gapless tag, so this is exactly where a naive pairing silently breaks.
    """
    degraded, meta = make_pair(music, bitrate, tmp_path / f"a{bitrate}.mp3")
    assert len(degraded) == len(music)
    assert degraded.dtype == np.int16
    assert meta["rel_error_aligned"] < 0.8
    assert meta["rel_error_aligned"] <= meta["rel_error_naive"]


def test_low_bitrates_actually_need_the_alignment(music: np.ndarray, tmp_path) -> None:
    """At 32 kbit/s the naive pairing must be visibly worse — otherwise the guard is untested."""
    _, meta = make_pair(music, 32, tmp_path / "a32.mp3")
    assert meta["lag_samples"] != 0
    assert meta["rel_error_naive"] > 2 * meta["rel_error_aligned"]


def test_misalignment_is_refused(music: np.ndarray, monkeypatch, tmp_path) -> None:
    """A failed lag search must raise, not return uncorrelated pairs."""
    import mp3ml.waveform as wf

    monkeypatch.setattr(wf, "estimate_lag", lambda *a, **k: 30000)
    with pytest.raises(RuntimeError, match="alignment failed"):
        make_pair(music, 32, tmp_path / "bad.mp3")


def test_relative_error_is_zero_for_identical_signals(music: np.ndarray) -> None:
    assert relative_error(music, music) == pytest.approx(0.0)


def test_short_signals_do_not_return_a_boundary_artifact(music: np.ndarray) -> None:
    """A lag can only be measured against much more signal than the lag itself.

    Without clamping, argmax lands on the search boundary and returns max_lag,
    which looks like a measurement but is an artifact.
    """
    short = music[:500]
    lag = estimate_lag(short, short)
    assert lag == 0, lag
    assert abs(lag) <= 500 // 4


def test_silent_audio_raises_rather_than_guessing(music: np.ndarray) -> None:
    """Many musdb stems are wholly or partly silent; correlating silence is meaningless."""
    silence = np.zeros(44100, dtype=np.int16)
    with pytest.raises(SilentAudioError):
        estimate_lag(silence, silence)


def test_lag_is_measured_on_the_loudest_region_not_the_start(music: np.ndarray) -> None:
    """A silent lead-in must not throw off the estimate — this caused real off-by-one
    errors on musdb stems that open with digital silence."""
    lead_in = np.zeros(300_000, dtype=np.int16)
    clean = np.concatenate([lead_in, music])
    delayed = np.concatenate([np.zeros(137, dtype=np.int16), clean])
    assert estimate_lag(clean, delayed) == 137


def test_periodic_content_needs_a_constrained_search() -> None:
    """A sustained tone defeats an unconstrained cross-correlation.

    Real case: a 49 Hz bass stem has autocorrelation 0.984 at its 900-sample
    period, so neighbouring cycles give near-identical correlation peaks and
    argmax picks among them arbitrarily. Anchoring to the known lag resolves it,
    because the tolerance is far narrower than any musical period.
    """
    t = np.arange(400_000) / SAMPLERATE
    tone = np.round(0.5 * np.sin(2 * np.pi * 49 * t) * 32767).astype(np.int16)
    true_lag = 3046
    delayed = np.concatenate([np.zeros(true_lag, dtype=np.int16), tone])

    # Unconstrained, the answer is allowed to land on a neighbouring cycle.
    free = estimate_lag(tone, delayed)
    assert (free - true_lag) % 900 == 0, f"expected a whole-period error, got {free}"

    # Anchored, it must be exact.
    assert estimate_lag(tone, delayed, expected_lag=true_lag) == true_lag


def test_constrained_search_does_not_disturb_a_correct_lag(music: np.ndarray) -> None:
    delayed = np.concatenate([np.zeros(137, dtype=np.int16), music])
    assert estimate_lag(music, delayed, expected_lag=137) == 137
    assert estimate_lag(music, delayed, expected_lag=140, tolerance=50) == 137
