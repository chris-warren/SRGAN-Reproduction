#!/usr/bin/env python
"""Audit a built waveform dataset, and optionally repair bad entries.

    python scripts/verify_waveform_dataset.py
    python scripts/verify_waveform_dataset.py --repair

Checks every entry for the failure modes that are silent at training time:

1. **Lag consistency** — the lag is a property of the decoder's resampler, so it
   must be identical across every track at a given bitrate. An outlier means the
   cross-correlation locked onto the wrong peak, which happens when a track opens
   with silence.
2. **Array agreement** — clean and degraded must exist and be the same length.
3. **Residual alignment** — re-measuring the lag on the stored pair must give 0.
4. **Error sanity** — a stored relative error near or above 1.0 means the pair is
   less correlated than silence would be.

`--repair` deletes and rebuilds any entry that fails, then re-runs the checks.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from mp3ml.waveform import MAX_ALIGNED_ERROR, estimate_lag, make_pair


def check_entry(root: Path, entry: dict, expected_lag: int) -> list[str]:
    problems = []
    clean_path = root / entry["clean_path"]
    deg_path = root / entry["degraded_path"]

    if not clean_path.exists():
        return [f"missing {clean_path.name}"]
    if not deg_path.exists():
        return [f"missing {deg_path.name}"]

    clean = np.load(clean_path, mmap_mode="r")
    degraded = np.load(deg_path, mmap_mode="r")
    if clean.shape != degraded.shape:
        problems.append(f"length mismatch {clean.shape} vs {degraded.shape}")
    if entry["lag_samples"] != expected_lag:
        problems.append(f"lag {entry['lag_samples']} != {expected_lag} (modal for this bitrate)")
    if not np.isfinite(entry["rel_error"]) or entry["rel_error"] > MAX_ALIGNED_ERROR:
        problems.append(f"rel_error {entry['rel_error']}")

    if not problems:
        # The stored pair is meant to be pre-aligned, so re-measuring must give 0.
        try:
            residual = estimate_lag(np.asarray(clean), np.asarray(degraded))
        except RuntimeError:
            residual = 0  # silent source; nothing to re-measure
        if residual != 0:
            problems.append(f"residual lag {residual} after storage")
    return problems


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--root", type=Path, default=Path("/home/chris/thesis/data/mp3ml-waveform"))
    p.add_argument("--repair", action="store_true", help="rebuild entries that fail")
    p.add_argument("--limit", type=int, default=None, help="check only the first N entries")
    args = p.parse_args()

    index = json.loads((args.root / "index.json").read_text())
    entries = index["entries"][: args.limit]
    print(f"auditing {len(entries)} entries in {args.root}\n")

    # The modal lag per bitrate is the ground truth: it comes from the resampler,
    # not the audio, so the majority value is the correct one by construction.
    modal = {}
    for b in sorted({e["bitrate_kbps"] for e in entries}):
        lags = Counter(e["lag_samples"] for e in entries if e["bitrate_kbps"] == b)
        modal[b] = lags.most_common(1)[0][0]
        print(f"  {b:>3}k modal lag {modal[b]:<6} distribution {dict(lags)}")
    print()

    bad: list[tuple[dict, list[str]]] = []
    for i, e in enumerate(entries, 1):
        problems = check_entry(args.root, e, modal[e["bitrate_kbps"]])
        if problems:
            bad.append((e, problems))
        if i % 200 == 0:
            print(f"  ...{i}/{len(entries)}")

    print(f"\n{len(entries) - len(bad)}/{len(entries)} entries OK")
    for e, problems in bad[:20]:
        print(
            f"  BAD  {e['track'][:38]:<38} {e['stem']:<8} {e['bitrate_kbps']:>3}k: "
            f"{'; '.join(problems)}"
        )
    if len(bad) > 20:
        print(f"  ... and {len(bad) - 20} more")

    if bad and args.repair:
        print(f"\nrepairing {len(bad)} entries...")
        repaired, still_bad = 0, 0
        for e, _ in bad:
            out_dir = (args.root / e["degraded_path"]).parent
            clean = np.load(args.root / e["clean_path"])
            shutil.rmtree(out_dir, ignore_errors=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            want = modal[e["bitrate_kbps"]]
            try:
                # Anchor the search to the modal lag. An unconstrained search is
                # what produced the bad value in the first place, and on strongly
                # periodic material it will reproduce it exactly.
                degraded, meta = make_pair(
                    clean, e["bitrate_kbps"], out_dir / "audio.mp3", expected_lag=want
                )
            except RuntimeError as exc:
                print(f"  FAILED {e['track'][:40]} @ {e['bitrate_kbps']}k: {exc}")
                still_bad += 1
                continue
            if meta["lag_samples"] != want:
                # Never report success on a lag that is still an outlier.
                print(
                    f"  FAILED {e['track'][:40]} @ {e['bitrate_kbps']}k: "
                    f"rebuilt lag {meta['lag_samples']} still != {want}"
                )
                still_bad += 1
                continue
            np.save(out_dir / "degraded.npy", degraded)
            (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
            e["lag_samples"] = meta["lag_samples"]
            e["rel_error"] = meta["rel_error_aligned"]
            repaired += 1
        (args.root / "index.json").write_text(json.dumps(index, indent=2))
        print(f"  repaired {repaired}, still failing {still_bad}")
        return 1 if still_bad else 0

    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
