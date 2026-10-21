"""Checks on lag estimation — the two traps that defeat cross-correlation.

`mp3ml.apollo.verify_constant_delay` relies on `estimate_lag` to prove that a
constant priming-delay trim really aligns a pair. If this estimator is wrong,
that proof is worthless and we would be training on misaligned audio without
any metric showing it — the failure mode Apollo's own loader has for 7-15% of
its pairs.
"""

import numpy as np
import pytest

from mp3ml.align import SilentAudioError, estimate_lag

SAMPLERATE = 44100


@pytest.fixture(scope="module")
def music() -> np.ndarray:
    """Three seconds of broadband, transient-rich audio — enough to lock onto."""
    rng = np.random.default_rng(0)
    t = np.arange(3 * SAMPLERATE) / SAMPLERATE
    x = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 1310 * t)
    x += 0.05 * rng.standard_normal(len(t))
    x[:: SAMPLERATE // 4] += 0.6
    return np.round(np.clip(x, -1, 1) * 32767).astype(np.int16)


@pytest.mark.parametrize("true_lag", [0, 1, 137, 1105, 2210, -50])
def test_estimate_lag_recovers_a_known_shift(music: np.ndarray, true_lag: int) -> None:
    """Including 1105 and 2210 — MP3's priming delay unresampled and resampled."""
    pad = 5000
    padded = np.concatenate([np.zeros(pad, dtype=np.int16), music])
    shifted = padded[pad - true_lag :] if true_lag <= pad else padded
    assert estimate_lag(music, shifted) == true_lag


def test_silent_audio_raises_rather_than_guessing(music: np.ndarray) -> None:
    """Many stems are wholly or partly silent; correlating silence is meaningless."""
    silence = np.zeros_like(music)
    with pytest.raises(SilentAudioError):
        estimate_lag(silence, silence)


def test_lag_is_measured_on_the_loudest_region_not_the_start(music: np.ndarray) -> None:
    """A silent lead-in must not throw off the estimate.

    This caused real off-by-one errors before `_loudest_window` existed: the
    first seconds of many stems are digital silence, and a correlation taken
    there locks onto noise.
    """
    lead_in = np.zeros(SAMPLERATE, dtype=np.int16)
    clean = np.concatenate([lead_in, music])
    true_lag = 1105
    pad = 5000
    padded = np.concatenate([np.zeros(pad, dtype=np.int16), clean])
    assert estimate_lag(clean, padded[pad - true_lag :]) == true_lag


def test_periodic_content_needs_a_constrained_search() -> None:
    """Periodic material is ambiguous by construction, so the search is bounded.

    A sustained 49 Hz bass note has autocorrelation 0.984 at its ~900-sample
    period, so neighbouring cycles give near-identical correlation peaks and an
    unconstrained `argmax` picks between them arbitrarily. Passing
    `expected_lag` resolves it, since the tolerance is far smaller than any
    musical period. **GCC-PHAT does not help here** — whitening amplifies phase
    in the empty bands a bass stem mostly consists of.
    """
    t = np.arange(3 * SAMPLERATE) / SAMPLERATE
    tone = np.round(0.5 * np.sin(2 * np.pi * 49 * t) * 32767).astype(np.int16)
    true_lag = 1105
    pad = 5000
    padded = np.concatenate([np.zeros(pad, dtype=np.int16), tone])
    delayed = padded[pad - true_lag :]
    assert estimate_lag(tone, delayed, expected_lag=true_lag) == true_lag


def test_short_signals_do_not_return_a_boundary_artifact(music: np.ndarray) -> None:
    """`argmax` hitting the search boundary looks like an answer and is not.

    `max_lag` is capped against the window actually available, so a lag cannot
    be "measured" from less signal than the lag itself.
    """
    short = music[: SAMPLERATE // 2]
    lag = estimate_lag(short, short, max_lag=20000)
    assert lag == 0, f"identical signals gave lag {lag}"
