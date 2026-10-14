#!/usr/bin/env python
"""Build the granule-pair dataset from a corpus of 44.1 kHz wavs.

    python scripts/build_dataset.py --tracks 10                    # pilot
    python scripts/build_dataset.py                                # full train split

Writes one directory per (bitrate, track) under --out, alongside the source
corpus rather than inside it. Safe to re-run: existing track directories are
skipped, so an interrupted build resumes.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from mp3ml.audio import load_pcm
from mp3ml.dataset import TrackEntry, write_index, write_track

# Stereo at 44.1 kHz cannot go below 112 kbps: LAME resamples (96k -> 32 kHz,
# 64k -> 24 kHz), which changes what xr means. Mono is safe down to 64.
DEFAULT_BITRATES_STEREO = (320, 192, 128, 112)
DEFAULT_BITRATES_MONO = (320, 192, 128, 64)

logger = logging.getLogger("build_dataset")


def find_tracks(source: Path, pattern: str) -> list[Path]:
    """Every track audio file under `source`, sorted for a deterministic split."""
    return sorted(source.glob(pattern))


def assign_split(index: int, val_every: int) -> str:
    """Split by track, never by chunk — chunks from one song must not straddle splits."""
    return "val" if val_every and index % val_every == val_every - 1 else "train"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source", type=Path, default=Path("/home/chris/thesis/data/musdb18hq/train")
    )
    parser.add_argument(
        "--pattern", default="*/mixture.wav", help="glob for track audio under --source"
    )
    parser.add_argument("--out", type=Path, default=Path("/home/chris/thesis/data/mp3ml-granules"))
    parser.add_argument("--bitrates", type=int, nargs="+", default=None)
    parser.add_argument("--channels", type=int, default=2, choices=(1, 2))
    parser.add_argument("--tracks", type=int, default=None, help="cap track count (pilot runs)")
    parser.add_argument("--seconds", type=float, default=None, help="cap each track's duration")
    parser.add_argument(
        "--val-every", type=int, default=10, help="every Nth track goes to val; 0 disables"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    bitrates = args.bitrates or (
        DEFAULT_BITRATES_STEREO if args.channels == 2 else DEFAULT_BITRATES_MONO
    )
    tracks = find_tracks(args.source, args.pattern)
    if not tracks:
        logger.error("no tracks matched %s under %s", args.pattern, args.source)
        return 1
    if args.tracks:
        tracks = tracks[: args.tracks]

    logger.info(
        "%d tracks, bitrates %s, %d channel(s) -> %s",
        len(tracks),
        list(bitrates),
        args.channels,
        args.out,
    )

    entries: list[TrackEntry] = []
    t_start = time.time()
    total_bytes = 0

    for i, wav in enumerate(tracks):
        name = wav.parent.name if wav.name == "mixture.wav" else wav.stem
        split = assign_split(i, args.val_every)
        try:
            pcm = load_pcm(wav, seconds=args.seconds, channels=args.channels)
        except ValueError as exc:
            logger.warning("skipping %s: %s", name, exc)
            continue
        duration = len(pcm) / 44100.0

        for bitrate in bitrates:
            out_dir = args.out / f"bitrate={bitrate}" / name
            if (out_dir / "meta.json").exists():
                logger.info(
                    "[%d/%d] %s @ %dk already built, skipping", i + 1, len(tracks), name, bitrate
                )
                meta = None
            else:
                t0 = time.time()
                meta = write_track(pcm, out_dir, bitrate, args.channels)
                size = sum(f.stat().st_size for f in out_dir.iterdir())
                total_bytes += size
                logger.info(
                    "[%d/%d] %s @ %dk: %d granules, %.1f MB, %.1fs  %s",
                    i + 1,
                    len(tracks),
                    name,
                    bitrate,
                    meta["n_granules"],
                    size / 1e6,
                    time.time() - t0,
                    meta["block_types"],
                )

            meta = meta or json.loads((out_dir / "meta.json").read_text())
            entries.append(
                TrackEntry(
                    track=name,
                    bitrate_kbps=bitrate,
                    channels=args.channels,
                    n_granules=meta["n_granules"],
                    n_samples=meta["input_samples"],
                    split=split,
                    path=str(out_dir.relative_to(args.out)),
                    source=str(wav),
                    duration_s=round(duration, 3),
                )
            )

    index_path = write_index(args.out, entries)
    n_train = sum(1 for e in entries if e.split == "train")
    gran = sum(e.n_granules for e in entries)

    print(f"\nbuilt {len(entries)} (track, bitrate) entries in {time.time() - t_start:.0f}s")
    print(f"  splits        {n_train} train / {len(entries) - n_train} val (split by track)")
    print(f"  granules      {gran:,} total ({gran * args.channels:,} channel-granules)")
    print(f"  written       {total_bytes / 1e9:.2f} GB this run")
    print(f"  index         {index_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
