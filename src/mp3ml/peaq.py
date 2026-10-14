"""ODG and Distortion Index via PEAQ (§3.4.1).

The paper uses `github.com/akinori-ito/peaqb-fast`, "an openly available
implementation of the basic version (as defined in the ITU recommendation) of
PEAQ" (ITU-R BS.1387-1). Named in footnote 3, so the *tool* is [P], not just
the metric — worth keeping, since PEAQ implementations disagree with each other
and matching the paper's numbers means matching its implementation.

The binary is not bundled: it is a separate C project under its own licence.
`available()` reports whether it is on PATH, and the harness skips ODG/DI with a
warning rather than failing, so evaluation works without it.

To install::

    git clone https://github.com/akinori-ito/peaqb-fast
    cd peaqb-fast && ./configure && make
    sudo cp src/peaqb /usr/local/bin/      # or put it on PATH

PEAQ compares a *reference* against a *test* signal and expects real files, so
this writes temporary WAVs.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

BINARY = "peaqb"

#: peaqb-fast prints one "ODG: <x>" / "DI: <x>" line per analysed frame.
_ODG = re.compile(r"ODG:\s*(-?\d+\.?\d*)")
_DI = re.compile(r"DI:\s*(-?\d+\.?\d*)")


def available() -> bool:
    return shutil.which(BINARY) is not None


def odg_di(
    reference: np.ndarray,
    test: np.ndarray,
    samplerate: int = 44100,
) -> tuple[float, float]:
    """Mean ODG and DI comparing `test` against `reference`.

    ODG runs 0 (imperceptible) to -4 (very annoying); DI is differently scaled
    but correlated. Higher is better for both (§4.1).

    Raises `FileNotFoundError` if the binary is missing — call `available()`
    first if you want to skip instead.
    """
    if not available():
        raise FileNotFoundError(
            f"{BINARY!r} not on PATH. PEAQ is an external tool; see this module's "
            "docstring for the four-line build, or skip ODG/DI."
        )
    with tempfile.TemporaryDirectory() as tmp:
        ref_path, test_path = Path(tmp) / "ref.wav", Path(tmp) / "test.wav"
        for path, data in ((ref_path, reference), (test_path, test)):
            sf.write(path, np.asarray(data, dtype=np.float32), samplerate, subtype="PCM_16")
        proc = subprocess.run(
            [BINARY, "-r", str(ref_path), "-t", str(test_path)],
            capture_output=True,
            text=True,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"{BINARY} failed ({proc.returncode}): {proc.stderr.strip()[:400]}")

    odgs = [float(m) for m in _ODG.findall(proc.stdout)]
    dis = [float(m) for m in _DI.findall(proc.stdout)]
    if not odgs:
        raise RuntimeError(
            f"{BINARY} produced no ODG lines — output format may differ from the "
            f"expected 'ODG: <value>'. First 400 chars:\n{proc.stdout[:400]}"
        )
    # peaqb emits per-frame values; the paper reports one number per excerpt.
    return float(np.mean(odgs)), float(np.mean(dis)) if dis else float("nan")
