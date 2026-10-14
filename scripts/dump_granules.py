#!/usr/bin/env python
"""Dump granule-aligned (xr, xr_hat) pairs for one audio file at one bitrate.

python scripts/dump_granules.py INPUT.wav --bitrate 320 --seconds 30
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from mp3ml.audio import load_mono
from mp3ml.oracle import dump_granules


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="source audio (44.1 kHz)")
    parser.add_argument("--bitrate", type=int, default=320, help="CBR bitrate in kbps")
    parser.add_argument("--seconds", type=float, default=None, help="cap input duration")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/granules"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("dump_granules")

    pcm = load_mono(args.input, seconds=args.seconds)
    log.info("loaded %s: %d samples (%.1f s) mono", args.input.name, len(pcm), len(pcm) / 44100)

    prefix = args.out_dir / f"{args.input.stem}_{args.bitrate}k"
    granules = dump_granules(pcm, bitrate_kbps=args.bitrate, out_prefix=prefix)

    xr = granules.xr.astype(np.float64)
    xr_hat = granules.xr_hat.astype(np.float64)
    rms = np.linalg.norm(xr - xr_hat) / np.linalg.norm(xr)

    expected = -(-len(pcm) // 1152) * granules.meta["mode_gr"]
    print(f"\n{args.input.name} @ {args.bitrate} kbps")
    print(f"  granules      {len(granules.records)} (upper bound from input length: {expected})")
    print(
        f"  frames        {granules.meta['emitted_frames']} emitted, "
        f"{granules.meta['encoder_frames']} encoded, {granules.meta['decoded_frames']} decoded"
    )
    print(f"  block types   {granules.block_type_counts()}")
    print(f"  RMS(xr-xr_hat)/RMS(xr)  {rms:.6e}")

    npz = prefix.with_suffix(".npz")
    granules.save_npz(npz)
    print(f"  wrote         {npz}")


if __name__ == "__main__":
    main()
