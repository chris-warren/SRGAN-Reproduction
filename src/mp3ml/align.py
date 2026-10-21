"""Sample-accurate lag estimation between a clean signal and its MP3.

Extracted from the waveform-pair builder on 2026-10-21, when that phase moved
to `/home/chris/thesis/mdct-granule-pairs`. Only this much survives in the
SR-GAN reproduction, because `mp3ml.apollo.verify_constant_delay` uses it to
prove that the constant priming-delay trim really does align a pair -- the
check that keeps us from repeating Apollo's own 7-15% misalignment.

Two traps live here, both found the hard way and both still worth knowing:

* **A silent lead-in defeats cross-correlation.** Many musdb and MoisesDB stems
  open with seconds of digital silence, and correlating silence against silence
  gives a meaningless peak. Hence `_loudest_window`.
* **Periodic material defeats it differently.** A sustained 49 Hz bass note has
  autocorrelation 0.984 at its 900-sample period, so neighbouring cycles give
  near-identical peaks and `argmax` picks between them arbitrarily. The fix is
  a search constrained around a known expected lag. **GCC-PHAT makes this
  worse**, not better: whitening amplifies phase in the empty bands a bass stem
  mostly consists of.
"""

from __future__ import annotations

import numpy as np


class SilentAudioError(RuntimeError):
    """The signal carries too little energy to align or to train on."""


def _loudest_window(x: np.ndarray, size: int) -> int:
    """Start index of the highest-energy window of `size` samples.

    Correlating a silent lead-in against a silent lead-in yields a meaningless
    peak, and many musdb stems open with seconds of digital silence — so the lag
    must be measured where there is actually signal.
    """
    if len(x) <= size:
        return 0
    cumulative = np.cumsum(np.concatenate([[0.0], x.astype(np.float64) ** 2]))
    starts = np.arange(0, len(x) - size + 1, max(size // 8, 1))
    return int(starts[np.argmax(cumulative[starts + size] - cumulative[starts])])


def estimate_lag(
    clean: np.ndarray,
    degraded: np.ndarray,
    max_lag: int = 20000,
    search_samples: int = 400_000,
    expected_lag: int | None = None,
    tolerance: int = 50,
) -> int:
    """Samples `degraded` lags `clean` by, via FFT cross-correlation.

    Positive means `degraded` starts later, so `degraded[lag:]` lines up with
    `clean[0:]`. The correlation is taken over the loudest region rather than the
    first samples, so a silent lead-in cannot produce a spurious lag.

    Pass `expected_lag` to constrain the search to +/-`tolerance` around it, which
    is the only reliable way to align strongly periodic material such as a
    sustained bass note.

    Raises `SilentAudioError` if even the loudest region is essentially silent.
    """
    size = min(search_samples, len(clean), len(degraded))
    # A lag can only be measured against far more signal than the lag itself;
    # otherwise the correlation peak runs into the search boundary and argmax
    # returns max_lag, which looks like an answer but is an artifact.
    max_lag = min(max_lag, max(size // 4, 1))
    start = _loudest_window(clean, size)

    a = clean[start : start + size].astype(np.float64)
    b = degraded[start : start + size].astype(np.float64)
    if np.sqrt((a**2).mean()) < 1.0:  # int16 scale: below 1 LSB RMS
        raise SilentAudioError(
            f"loudest {size / 44100:.1f}s window has RMS "
            f"{np.sqrt((a**2).mean()):.3e} — no signal to align on"
        )

    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    xc = np.fft.irfft(np.fft.rfft(a - a.mean(), n) * np.conj(np.fft.rfft(b - b.mean(), n)), n)

    if expected_lag is None:
        window = np.concatenate([xc[-max_lag:], xc[: max_lag + 1]])
        return int(max_lag - np.argmax(window))

    # Constrained search. Periodic content defeats an unconstrained correlation:
    # a sustained 49 Hz bass note has autocorrelation 0.984 at its 900-sample
    # period, so neighbouring cycles give near-identical peaks and argmax picks
    # among them arbitrarily. (GCC-PHAT does not help here — whitening amplifies
    # phase in the empty bands a bass stem mostly consists of.) The lag is a
    # property of the decoder's resampler, identical for every file at a given
    # bitrate, so searching a narrow window around the known value resolves the
    # ambiguity by construction: the tolerance is far smaller than any musical
    # period.
    offsets = np.arange(expected_lag - tolerance, expected_lag + tolerance + 1)
    return int(offsets[np.argmax(xc[-offsets % n])])
