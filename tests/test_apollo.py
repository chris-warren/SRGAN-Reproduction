"""Checks on the Apollo corpus adapter, especially which bitrates are usable.

Synthetic audio only — the corpus lives on an external drive and these must run
without it. The two tests that need `lame`/`ffmpeg` follow the precedent in
`test_waveform.py`, which also round-trips through the real encoder.
"""

import shutil

import numpy as np
import pytest

from mp3ml.apollo import (
    APOLLO_BITRATES,
    COMPARABLE_BITRATES,
    MONO_OUTPUT_SAMPLERATE,
    MONO_SAFE_BITRATES,
    MP3_PRIMING_DELAY,
    degrade,
    verify_constant_delay,
)

SAMPLERATE = 44100
#: The paper's rates (§4.1), for the intersection argument below.
PAPER_BITRATES = (16, 32, 64)

needs_codec = pytest.mark.skipif(
    shutil.which("lame") is None or shutil.which("ffmpeg") is None,
    reason="needs lame and ffmpeg on PATH",
)


@pytest.fixture(scope="module")
def music() -> np.ndarray:
    """Four seconds of broadband, transient-rich audio as float32 in [-1, 1]."""
    rng = np.random.default_rng(0)
    t = np.arange(4 * SAMPLERATE) / SAMPLERATE
    x = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 1310 * t)
    x += 0.05 * rng.standard_normal(len(t))
    x[:: SAMPLERATE // 4] += 0.6
    return np.clip(x, -1, 1).astype(np.float32)


def test_only_mono_safe_rates_avoid_resampling() -> None:
    """Below 64 kbit/s mono LAME resamples, measured 2026-10-21 with ffprobe.

    This is the 64 kbit/s mono CBR floor CLAUDE.md warns about. A resampled
    encode is a different degradation from the one the paper measures, and it
    changes MP3's priming delay, so the tables must agree on which rates are
    safe.
    """
    for rate, sr in MONO_OUTPUT_SAMPLERATE.items():
        assert (sr == SAMPLERATE) == (rate in MONO_SAFE_BITRATES), rate
    assert set(MONO_SAFE_BITRATES) <= set(APOLLO_BITRATES)


def test_comparable_bitrates_are_the_only_defensible_choice() -> None:
    """64 kbit/s is the sole rate in Apollo's set, the paper's set, and safe.

    If this ever fails, the comparison's whole premise has moved: a run is only
    comparable in both directions at a rate both corpora contain and that does
    not resample.
    """
    both = set(APOLLO_BITRATES) & set(PAPER_BITRATES)
    assert both == {32, 64}, "the two papers' bitrate sets changed"
    assert set(COMPARABLE_BITRATES) == both & set(MONO_SAFE_BITRATES) == {64}
    assert 32 not in MONO_SAFE_BITRATES, "32 is in both sets but resamples in mono"


@needs_codec
def test_degrade_refuses_rates_that_resample(music: np.ndarray) -> None:
    for rate in (24, 32, 48):
        with pytest.raises(ValueError, match="resamples"):
            degrade(music, rate)


@needs_codec
def test_degrade_returns_an_aligned_signal_of_equal_length(music: np.ndarray) -> None:
    """The pair must line up sample-for-sample, with no measurable residual lag.

    `degrade` trims a hard-coded `MP3_PRIMING_DELAY`; this asserts the constant
    is actually right by measuring with the same estimator the main dataset
    builder uses. Apollo's own loader skips this check and leaves 7-15% of its
    pairs out of step (its README), which is precisely the silent corruption
    this guards against.
    """
    out = degrade(music, 64)
    assert out.shape == music.shape
    assert out.dtype == np.float32
    assert verify_constant_delay(music, (64,)) == {64: 0}


@needs_codec
def test_degradation_actually_degrades(music: np.ndarray) -> None:
    """A pass-through would make every later metric meaningless."""
    out = degrade(music, 64)
    residual = np.sqrt(((out - music) ** 2).mean()) / np.sqrt((music**2).mean())
    assert 0.01 < residual < 1.0, f"relative residual {residual:.3f} is implausible"


def test_priming_delay_is_the_documented_sum() -> None:
    """576 encoder + 529 decoder, the figure DATASET.md and CLAUDE.md both use."""
    assert MP3_PRIMING_DELAY == 576 + 529
