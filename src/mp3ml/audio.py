"""Loading source audio into the mono int16 PCM the dumper expects."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def load_pcm(
    path: Path,
    seconds: float | None = None,
    channels: int = 1,
    samplerate: int = 44100,
) -> np.ndarray:
    """Load `path` as int16 PCM: shape (n,) for mono, (n, 2) interleaved for stereo.

    Raises if the file's sample rate is not `samplerate`; this pipeline does not
    resample, because resampling would add its own artifacts to the clean side of
    every training pair.
    """
    if channels not in (1, 2):
        raise ValueError(f"channels must be 1 or 2, got {channels}")

    data, sr = sf.read(str(path), dtype="float64", always_2d=True)
    if sr != samplerate:
        raise ValueError(f"{path} is {sr} Hz, expected {samplerate} Hz (no resampling here)")

    if channels == 1:
        out = data.mean(axis=1)
    elif data.shape[1] == 1:
        out = np.repeat(data, 2, axis=1)  # mono source, stereo output
    else:
        out = data[:, :2]

    if seconds is not None:
        out = out[: int(seconds * sr)]

    peak = np.abs(out).max()
    if peak > 1.0:
        logger.warning("%s peaks at %.3f; clipping to [-1, 1]", path, peak)
        out = np.clip(out, -1.0, 1.0)

    return np.round(out * 32767.0).astype(np.int16)


def load_mono(path: Path, seconds: float | None = None, samplerate: int = 44100) -> np.ndarray:
    """Mono int16 PCM, shape (n,). Thin wrapper over `load_pcm`."""
    return load_pcm(path, seconds=seconds, channels=1, samplerate=samplerate)
