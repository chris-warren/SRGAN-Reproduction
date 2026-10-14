#!/usr/bin/env python
"""Validate the LAME oracle before trusting any granule data it produces.

Five checks:

1. Encoder settings — with LAME's analysis path OFF, mp3dump's bitstream must be
   byte-identical to the stock `lame` CLI, proving our encoder configuration
   matches stock LAME. Also reports how much enabling analysis perturbs the
   bitstream (a known, documented effect at high bitrates — see README).
2. Decoder agreement — mpglib's decode (which produces xr_hat) must match
   ffmpeg's decode of the same file. There is no mpg123 on this box.
3. Degradation trend — RMS(xr - xr_hat) must rise as bitrate falls.
4. Granule alignment — encoder- and decoder-side side info must agree for every
   granule. The sharpest check: the two come from different code paths, so a
   one-frame error in the join collapses it immediately.
5. xr consistency across bitrates — after dividing out LAME's per-bitrate input
   scale, xr must agree across bitrates wherever the block type and lowpass
   agree, confirming xr is the transform of the input and nothing else.

    python scripts/validate_oracle.py [INPUT.wav] [--seconds 10]
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

import numpy as np

from mp3ml.audio import load_mono
from mp3ml.encode import decode_to_pcm, reference_encode
from mp3ml.oracle import check_alignment, dump_granules

BITRATES = (320, 192, 128, 64)


def synthetic_pcm(seconds: float, samplerate: int = 44100) -> np.ndarray:
    """A tone pair plus a click train, so long, start, short and stop blocks all occur."""
    t = np.arange(int(seconds * samplerate)) / samplerate
    x = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.15 * np.sin(2 * np.pi * 3011 * t)
    x[:: samplerate // 4] += 0.5  # transients -> short blocks
    return np.round(np.clip(x, -1, 1) * 32767).astype(np.int16)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", type=Path, nargs="?", help="source audio; synthetic if omitted")
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if args.input is not None:
        pcm = load_mono(args.input, seconds=args.seconds)
        source = args.input.name
    else:
        pcm = synthetic_pcm(args.seconds)
        source = f"synthetic tone + clicks ({args.seconds:g}s)"

    print(f"source: {source}  ({len(pcm)} samples mono @ 44100 Hz)\n")
    failures: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        sets = {b: dump_granules(pcm, b, tmpdir / f"v{b}", write_pcm=True) for b in BITRATES}

        # -- Check 1: encoder settings match stock LAME ------------------------
        print("1. encoder settings (analysis OFF) vs stock `lame` CLI")
        for bitrate in BITRATES:
            plain = dump_granules(pcm, bitrate, tmpdir / f"na{bitrate}", analysis=False)
            ref = reference_encode(pcm, bitrate, tmpdir / f"ref{bitrate}.mp3")
            ours, theirs = plain.mp3_path.read_bytes(), ref.read_bytes()
            with_analysis = sets[bitrate].mp3_path.read_bytes()
            perturb = sum(1 for a, b in zip(with_analysis, theirs, strict=False) if a != b) + abs(
                len(with_analysis) - len(theirs)
            )
            if ours == theirs:
                note = "identical" if perturb == 0 else f"identical; analysis perturbs {perturb} B"
                print(f"   {bitrate:>4} kbps  {note}")
            else:
                print(
                    f"   {bitrate:>4} kbps  DIFFER even with analysis off "
                    f"(ours={len(ours)} ref={len(theirs)})"
                )
                failures.append(f"encoder settings at {bitrate} kbps")

        # -- Check 2: decoder agreement ---------------------------------------
        print("\n2. decoder agreement (mpglib via mp3dump vs ffmpeg, same .mp3)")
        for bitrate, gs in sets.items():
            ours = np.fromfile(gs.pcm_path, dtype="<i2").astype(np.float64) / 32768.0
            ffmpeg = decode_to_pcm(gs.mp3_path)
            n = min(len(ours), len(ffmpeg))
            # ffmpeg drops the decoder delay that mpglib emits; align on best lag.
            best = min(
                (
                    np.linalg.norm(ours[lag : lag + n - 1200] - ffmpeg[: n - 1200])
                    / np.linalg.norm(ffmpeg[: n - 1200]),
                    lag,
                )
                for lag in range(0, 1200)
            )
            status = "ok" if best[0] < 1e-3 else "HIGH"
            print(f"   {bitrate:>4} kbps  normalized RMS {best[0]:.4e} at lag {best[1]}  {status}")
            if best[0] >= 1e-3:
                failures.append(f"decoder disagreement at {bitrate} kbps")

        # -- Check 3: degradation trend ---------------------------------------
        print("\n3. degradation rises as bitrate falls")
        rms = {}
        for bitrate, gs in sets.items():
            xr = gs.xr.astype(np.float64)
            xr_hat = gs.xr_hat.astype(np.float64)
            rms[bitrate] = np.linalg.norm(xr - xr_hat) / np.linalg.norm(xr)
            print(f"   {bitrate:>4} kbps  RMS(xr-xr_hat)/RMS(xr) = {rms[bitrate]:.6e}")
        ordered = [rms[b] for b in sorted(BITRATES, reverse=True)]
        if all(a < b for a, b in zip(ordered, ordered[1:], strict=False)):
            print("   monotonic: yes")
        else:
            print("   monotonic: NO")
            failures.append("degradation is not monotonic in bitrate")

        # -- Check 4: granule alignment ---------------------------------------
        print("\n4. granule alignment (encoder vs decoder side info)")
        for bitrate, gs in sets.items():
            agree = check_alignment(gs.records)
            xr = gs.xr.astype(np.float64)
            xr_hat = gs.xr_hat.astype(np.float64)
            ok = (xr.std(axis=1) > 0) & (xr_hat.std(axis=1) > 0)
            corr = np.array(
                [np.corrcoef(a, b)[0, 1] for a, b in zip(xr[ok], xr_hat[ok], strict=True)]
            )
            print(
                f"   {bitrate:>4} kbps  block_type {agree['block_type']:.4f}  "
                f"global_gain {agree['global_gain']:.4f}  "
                f"corr min {corr.min():.6f}  blocks {gs.block_type_counts()}"
            )
            if agree["block_type"] < 1.0 or agree["global_gain"] < 1.0:
                failures.append(f"side-info misalignment at {bitrate} kbps")

        # -- Check 5: xr consistency across bitrates ---------------------------
        print("\n5. xr across bitrates, after dividing out LAME's input scale")
        base = sets[320]
        for bitrate, gs in sets.items():
            if bitrate == 320:
                continue
            n = min(len(base.records), len(gs.records))
            same_bt = base.records["block_type"][:n] == gs.records["block_type"][:n]
            a = base.xr[:n][same_bt].astype(np.float64) / base.meta["input_scale"]
            b = gs.xr[:n][same_bt].astype(np.float64) / gs.meta["input_scale"]
            rel = np.linalg.norm(a - b) / np.linalg.norm(a)
            lp = gs.meta["lowpass_hz"]
            same_lp = lp == base.meta["lowpass_hz"]
            note = "" if same_lp else f" (lowpass differs: {lp} vs {base.meta['lowpass_hz']} Hz)"
            print(
                f"   320 vs {bitrate:>3} kbps  scale {gs.meta['input_scale']:.4f}  "
                f"rel diff {rel:.3e}{note}"
            )
            if same_lp and rel > 1e-5:
                failures.append(f"xr inconsistent between 320 and {bitrate} kbps")

    print()
    if failures:
        print("FAILED: " + "; ".join(failures))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
