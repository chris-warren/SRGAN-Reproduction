"""Waveform pairs: original audio against its MP3-decoded self.

This is the representation used by Lattner & Nistal, *Stochastic Restoration of
Heavily Compressed Musical Audio Using Generative Adversarial Networks*
(Electronics 2021, 10, 1349) — a complex STFT of the decoded waveform, not MP3's
internal coefficients. Granule pairs (`mp3ml.oracle`) cannot serve that task: at
16 and 32 kbit/s LAME resamples internally, so `xr` and `xr_hat` both live at the
reduced rate and a model trained on them could never restore the lost bandwidth.

Alignment is the trap here. MP3 has an inherent round-trip delay of **1105
samples** (576 encoder + 529 decoder). LAME normally records it in a Xing/Info +
LAME gapless tag, and a decoder that honours the tag trims it: at 44.1 kHz ffmpeg
reports `start_time = 0.025057` s, exactly 1105/44100, and the round trip comes
back at lag 0.

Below 32 kbit/s LAME switches to MPEG-2/2.5 and **writes no Xing/Info/LAME tag at
all**, so nothing tells the decoder to trim. The full 1105-sample delay survives,
measured at the codec's native rate, and resampling back to 44.1 kHz scales it:
1105 x 44100/22050 = 2210 (32 kbit/s) and 1105 x 44100/16000 = 3045.7 ~ 3046
(16 kbit/s). The resampler itself contributes essentially nothing (measured 0.0
and 0.3 samples) -- ffmpeg compensates its own filter delay.

Pairing naively then gives relative error above 1.0, i.e. worse than pairing the
clean signal against silence, with nothing to signal the mistake. So the lag is
measured per file here and the degraded signal is stored **already aligned**,
sample for sample, with the lag recorded for provenance.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

#: Above this, treat the pairing as failed. Aligned relative error measured on
#: real music is ~0.07 (64k), ~0.11 (32k), ~0.15 (16k); a misaligned pair lands
#: above 1.0, so anything near that means the lag search failed.
MAX_ALIGNED_ERROR = 0.8


def encoder_settings(bitrate_kbps: int) -> dict:
    """The exact encode/decode configuration, for recording alongside the data.

    The reference paper (Electronics 2021, 10, 1349 section 3.3) fixes only the
    bitrate, mono, LAME 3.100 and a 44.1 kHz source; everything here beyond that
    is our choice. See DATASET.md.
    """
    return {
        "encoder": "lame",
        "encoder_version": _tool_version(["lame", "--version"]),
        "encode_argv": [
            "lame",
            "--quiet",
            "-m",
            "m",
            "--cbr",
            "-b",
            str(bitrate_kbps),
            "<in.wav>",
            "<out.mp3>",
        ],
        "rate_control": "cbr",
        "channels": 1,
        "downmix": "arithmetic mean of L/R before encoding",
        "decoder": "ffmpeg",
        "decoder_version": _tool_version(["ffmpeg", "-version"]),
        "decode_argv": [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            "<in.mp3>",
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            "44100",
            "-",
        ],
        "output_samplerate": 44100,
        "alignment": "FFT cross-correlation; degraded stored pre-shifted",
    }


def _tool_version(argv: list[str]) -> str:
    try:
        out = subprocess.run(argv, capture_output=True, check=False, text=True)
        return (out.stdout or out.stderr).splitlines()[0].strip()
    except (OSError, IndexError):
        return "unknown"


def encode_mp3(pcm: np.ndarray, bitrate_kbps: int, out_path: Path, samplerate: int = 44100) -> Path:
    """Encode mono int16 `pcm` to MP3 with the stock `lame` CLI.

    Unlike `mp3ml.encode.reference_encode`, the LAME/Xing tag is **kept** — it
    carries the gapless delay/padding that lets a decoder line the output back
    up with the input.
    """
    if shutil.which("lame") is None:
        raise FileNotFoundError("`lame` not found on PATH")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.wav"
        sf.write(src, pcm, samplerate, subtype="PCM_16")
        subprocess.run(
            [
                "lame",
                "--quiet",
                "-m",
                "m",
                "--cbr",
                "-b",
                str(bitrate_kbps),
                str(src),
                str(out_path),
            ],
            check=True,
        )
    return out_path


def decode_mp3(mp3_path: Path, samplerate: int = 44100) -> np.ndarray:
    """Decode to mono int16 at `samplerate`, resampling up from LAME's internal rate."""
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("`ffmpeg` not found on PATH")
    out = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(mp3_path),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            str(samplerate),
            "-",
        ],
        check=True,
        capture_output=True,
    ).stdout
    return np.frombuffer(out, dtype="<i2").copy()


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


def align(clean: np.ndarray, degraded: np.ndarray, lag: int) -> np.ndarray:
    """Shift `degraded` by `lag` and pad/trim it to exactly `len(clean)`.

    The result is sample-aligned with `clean`, so a training loop can slice both
    at the same offsets with no index arithmetic.
    """
    if lag >= 0:
        shifted = degraded[lag:]
    else:
        shifted = np.concatenate([np.zeros(-lag, dtype=degraded.dtype), degraded])
    if len(shifted) < len(clean):
        shifted = np.concatenate(
            [shifted, np.zeros(len(clean) - len(shifted), dtype=degraded.dtype)]
        )
    return shifted[: len(clean)]


def relative_error(clean: np.ndarray, degraded: np.ndarray) -> float:
    a = clean.astype(np.float64)
    b = degraded.astype(np.float64)
    denom = np.linalg.norm(a)
    return float(np.linalg.norm(b - a) / denom) if denom else float("nan")


def make_pair(
    pcm: np.ndarray,
    bitrate_kbps: int,
    mp3_path: Path,
    samplerate: int = 44100,
    expected_lag: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Encode, decode, resample and align. Returns (degraded, meta).

    Raises `SilentAudioError` if the source has no usable signal (many musdb
    stems are wholly or partly silent), and `RuntimeError` if the aligned pair is
    still implausibly far apart, meaning the lag search failed.
    """
    encode_mp3(pcm, bitrate_kbps, mp3_path, samplerate)
    decoded = decode_mp3(mp3_path, samplerate)
    lag = estimate_lag(pcm, decoded, expected_lag=expected_lag)
    degraded = align(pcm, decoded, lag)

    err_aligned = relative_error(pcm, degraded)
    err_naive = relative_error(pcm, align(pcm, decoded, 0))
    if not np.isfinite(err_aligned) or err_aligned > MAX_ALIGNED_ERROR:
        raise RuntimeError(
            f"alignment failed at {bitrate_kbps}k: relative error {err_aligned:.3f} "
            f"after shifting by {lag} samples (naive {err_naive:.3f}). A misaligned "
            "pair trains on essentially uncorrelated signals."
        )
    return degraded, {
        "bitrate_kbps": bitrate_kbps,
        "samplerate": samplerate,
        "lag_samples": lag,
        "rel_error_aligned": round(err_aligned, 6),
        "rel_error_naive": round(err_naive, 6),
        "n_samples": int(len(degraded)),
        "lag_constrained": expected_lag is not None,
        "mp3_bytes": int(mp3_path.stat().st_size),
    }
