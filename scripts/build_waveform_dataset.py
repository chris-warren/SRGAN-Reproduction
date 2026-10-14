#!/usr/bin/env python
"""Build waveform pairs for the Lattner & Nistal GAN reproduction.

Original 44.1 kHz mono against its MP3-decoded self at 16/32/64 kbit/s, stored
sample-aligned so a training loop can slice both at identical offsets.

    python scripts/build_waveform_dataset.py --tracks 3 --splits train   # pilot
    python scripts/build_waveform_dataset.py --stems --workers 12        # full, 47 h

Layout::

    <root>/
      index.json
      clean/<track>__<stem>.npy          44.1 kHz mono int16, canonical
      bitrate=16/<track>__<stem>/
        degraded.npy                     same length as clean, pre-aligned
        audio.mp3
        meta.json                        lag + alignment residual

Safe to re-run: existing entries are skipped, so an interrupted build resumes.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from mp3ml.audio import load_pcm
from mp3ml.waveform import SilentAudioError, encoder_settings, make_pair

#: The paper's bitrates (Electronics 2021, 10, 1349, section 3.3). All mono.
DEFAULT_BITRATES = (16, 32, 64)
STEM_NAMES = ("mixture", "bass", "drums", "other", "vocals")

logger = logging.getLogger("build_waveform")


@dataclass(frozen=True)
class WaveEntry:
    track: str
    stem: str
    bitrate_kbps: int
    n_samples: int
    lag_samples: int
    rel_error: float
    split: str
    clean_path: str
    degraded_path: str
    source: str


def build_one(job: dict) -> dict:
    """Build every bitrate for one (track, stem). Runs in a worker process."""
    wav = Path(job["wav"])
    out = Path(job["out"])
    name = job["name"]
    clean_path = out / "clean" / f"{name}.npy"
    entries: list[dict] = []
    failures: list[str] = []
    skipped: list[str] = []
    written = 0

    try:
        pcm = load_pcm(wav, seconds=job["seconds"], channels=1)
    except (ValueError, RuntimeError) as exc:
        return {"entries": [], "failures": [f"{name}: {exc}"], "bytes": 0, "name": name}

    if not clean_path.exists():
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(clean_path, pcm)
        written += clean_path.stat().st_size

    for bitrate in job["bitrates"]:
        out_dir = out / f"bitrate={bitrate}" / name
        meta_path = out_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                degraded, meta = make_pair(pcm, bitrate, out_dir / "audio.mp3")
            except SilentAudioError as exc:
                # Not a failure: a silent stem carries no training signal, so
                # there is nothing to align and nothing to learn. Drop it.
                skipped.append(f"{name}@{bitrate}k: {exc}")
                continue
            except RuntimeError as exc:
                failures.append(f"{name}@{bitrate}k: {exc}")
                continue
            np.save(out_dir / "degraded.npy", degraded)
            meta_path.write_text(json.dumps(meta, indent=2))
            written += sum(f.stat().st_size for f in out_dir.iterdir())

        entries.append(
            asdict(
                WaveEntry(
                    track=job["track"],
                    stem=job["stem"],
                    bitrate_kbps=bitrate,
                    n_samples=meta["n_samples"],
                    lag_samples=meta["lag_samples"],
                    rel_error=meta["rel_error_aligned"],
                    split=job["split"],
                    clean_path=str(clean_path.relative_to(out)),
                    degraded_path=str((out_dir / "degraded.npy").relative_to(out)),
                    source=str(wav),
                )
            )
        )
    return {
        "entries": entries,
        "failures": failures,
        "skipped": skipped,
        "bytes": written,
        "name": name,
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--source", type=Path, default=Path("/home/chris/thesis/data/musdb18hq"))
    p.add_argument("--out", type=Path, default=Path("/home/chris/thesis/data/mp3ml-waveform"))
    p.add_argument("--bitrates", type=int, nargs="+", default=list(DEFAULT_BITRATES))
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument(
        "--stems", action="store_true", help="also build the 4 stems, not just mixtures (5x data)"
    )
    p.add_argument(
        "--val-every",
        type=int,
        default=10,
        help="every Nth train track becomes val; 0 disables. musdb ships only "
        "train/ and test/, so a validation split has to be carved out of train.",
    )
    p.add_argument("--tracks", type=int, default=None, help="cap tracks per split (pilot runs)")
    p.add_argument("--seconds", type=float, default=None, help="cap each track's duration")
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel worker processes; each handles one (track, stem)",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    stems = STEM_NAMES if args.stems else ("mixture",)

    jobs = []
    for split in args.splits:
        track_dirs = sorted(d for d in (args.source / split).iterdir() if d.is_dir())
        if args.tracks:
            track_dirs = track_dirs[: args.tracks]
        for t_i, tdir in enumerate(track_dirs):
            # Split by TRACK: every stem of a track shares its split, since stems
            # are derived from the same mixture and separating them would leak.
            if split == "train" and args.val_every:
                track_split = "val" if t_i % args.val_every == args.val_every - 1 else "train"
            else:
                track_split = split
            for stem in stems:
                wav = tdir / f"{stem}.wav"
                if not wav.exists():
                    logger.warning("missing %s for %s", stem, tdir.name)
                    continue
                jobs.append(
                    {
                        "wav": str(wav),
                        "out": str(args.out),
                        "track": tdir.name,
                        "stem": stem,
                        "name": f"{tdir.name}__{stem}",
                        "split": track_split,
                        "bitrates": args.bitrates,
                        "seconds": args.seconds,
                    }
                )

    logger.info(
        "%d files x %d bitrates, %d worker(s) -> %s",
        len(jobs),
        len(args.bitrates),
        args.workers,
        args.out,
    )

    entries: list[dict] = []
    failures: list[str] = []
    skipped: list[str] = []
    total_bytes = 0
    t_start = time.time()

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(build_one, j): j for j in jobs}
            for i, fut in enumerate(as_completed(futures), 1):
                r = fut.result()
                entries += r["entries"]
                failures += r["failures"]
                skipped += r["skipped"]
                total_bytes += r["bytes"]
                logger.info("[%d/%d] %s", i, len(jobs), r["name"])
    else:
        for i, j in enumerate(jobs, 1):
            r = build_one(j)
            entries += r["entries"]
            failures += r["failures"]
            skipped += r["skipped"]
            total_bytes += r["bytes"]
            logger.info("[%d/%d] %s", i, len(jobs), r["name"])

    entries.sort(key=lambda e: (e["split"], e["track"], e["stem"], e["bitrate_kbps"]))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "index.json").write_text(
        json.dumps(
            {
                "n_entries": len(entries),
                "bitrates": args.bitrates,
                "samplerate": 44100,
                "channels": 1,
                "stems": list(stems),
                "encoder_settings": {str(b): encoder_settings(b) for b in args.bitrates},
                "provenance": "see DATASET.md for what the reference paper "
                "specifies and what was chosen here",
                "entries": entries,
            },
            indent=2,
        )
    )

    hours = sum(e["n_samples"] for e in entries) / 44100 / 3600 / max(len(args.bitrates), 1)
    errs = {b: [e["rel_error"] for e in entries if e["bitrate_kbps"] == b] for b in args.bitrates}
    print(f"\nbuilt {len(entries)} entries in {time.time() - t_start:.0f}s")
    for split in sorted({e["split"] for e in entries}):
        n_tracks = len({e["track"] for e in entries if e["split"] == split})
        print(
            f"  {split:<6} {sum(1 for e in entries if e['split'] == split):>6} entries "
            f"({n_tracks} tracks)"
        )
    print(f"  audio         {hours:.2f} h per bitrate")
    for b in sorted(errs):
        v = errs[b]
        if v:
            print(f"  {b:>3}k rel err  mean {np.mean(v):.4f}  range {min(v):.4f}-{max(v):.4f}")
    print(f"  written       {total_bytes / 1e9:.2f} GB this run")
    if skipped:
        print(f"  skipped       {len(skipped)} silent sources (no signal to learn)")
    if failures:
        print(f"  FAILURES      {len(failures)}")
        for f in failures[:5]:
            print(f"    {f}")
    print(f"  index         {args.out / 'index.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
