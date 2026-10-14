---
name: dump-granules
description: Extract (xr, xr_hat) granule pairs from an audio file at a given bitrate using the LAME oracle, and sanity-check the result. Use when generating or inspecting granule-pair training data.
argument-hint: [audio-file] [bitrate]
---

Dump granule pairs for $ARGUMENTS:

1. If `native/lame_dump/mp3dump` doesn't exist, build it with
   `bash native/lame_dump/build.sh` first.
2. Run `./.venv/bin/python scripts/dump_granules.py <file> --bitrate <kbps>`.
   **Always pass `--seconds`** unless the user explicitly asked for a whole
   track: `xr` is one float per input sample, so a clean+degraded float32 pair is
   ~1.27 GB per audio-hour per bitrate. Default to `--seconds 30` while iterating.
3. Report the granule count, the block-type histogram, and `RMS(xr-xr_hat)`.

Sanity-check the numbers before calling it done:

- Granule count should be close to `ceil(samples / 1152) * 2`. A shortfall of one
  or two frames is expected — the trailing flush frames are dropped on purpose
  (see `native/lame_dump/README.md`).
- `RMS(xr-xr_hat)` should be roughly 1e-3 (320 kbps) to 1e-1 (64 kbps) on real
  music. Near zero means the pair is degenerate; above ~0.5 means something is
  misaligned.
- If the block-type histogram is all `long`, say so — the material has no
  transients and won't exercise the short/start/stop path that the eventual
  model has to handle.

If anything looks off, don't diagnose it from the summary alone — run the
`validate-oracle` skill, which checks granule alignment exactly.

Source audio must be 44.1 kHz; the loader refuses anything else rather than
resampling, because resampling would add artifacts to the clean side of every
pair. `data/musdb18hq/train` holds 94 complete tracks, each with a `mixture.wav`.

Only CBR at >= 64 kbps is supported. Below that LAME silently resamples, and
`dump_granules` raises rather than returning data that no longer lines up with
the input.
