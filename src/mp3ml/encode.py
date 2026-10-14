"""Reference MP3 encode and decode, for validating the dumper against stock tools."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np


def reference_encode(
    pcm: np.ndarray, bitrate_kbps: int, out_path: Path, samplerate: int = 44100
) -> Path:
    """Encode mono int16 `pcm` with the stock `lame` CLI, matching mp3dump's settings.

    ``-t`` disables the LAME Tag (the Xing/Info header frame), matching the
    dumper's ``lame_set_bWriteVbrTag(0)``. Everything else is left at LAME's
    defaults, exactly as the dumper leaves it, so the two bitstreams are
    comparable byte for byte — see check 1 in ``scripts/validate_oracle.py``.
    """
    if shutil.which("lame") is None:
        raise FileNotFoundError("`lame` not found on PATH")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "lame",
        "--quiet",
        "-t",
        "-r",  # raw PCM input
        "-s",
        str(samplerate / 1000.0),
        "--bitwidth",
        "16",
        "--signed",
        "--little-endian",
        "-m",
        "m",
        "--cbr",
        "-b",
        str(bitrate_kbps),
        "-",
        str(out_path),
    ]
    subprocess.run(cmd, input=pcm.astype("<i2").tobytes(), check=True, stdout=subprocess.DEVNULL)
    return out_path


def decode_to_pcm(mp3_path: Path, samplerate: int = 44100) -> np.ndarray:
    """Decode `mp3_path` to mono float64 with ffmpeg (there is no mpg123 here)."""
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("`ffmpeg` not found on PATH")
    cmd = [
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
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True)
    return np.frombuffer(proc.stdout, dtype="<i2").astype(np.float64) / 32768.0
