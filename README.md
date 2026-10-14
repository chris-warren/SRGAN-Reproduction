# MDCT-ML — MP3 compression-reversal dataset prep

Preparing training data for a model that reverses MP3 compression: given lossy
MP3 output, reconstruct something closer to the uncompressed original. The
research plan is `mp3_dataset_plan.md`; this repo is the current phase, which is
dataset preparation and evaluation only — **no model training yet**.

## Two datasets, two purposes

| dataset | builder | for |
|---|---|---|
| **waveform pairs** | `scripts/build_waveform_dataset.py` | reproducing Lattner & Nistal's GAN (STFT domain) |
| **granule pairs** | `scripts/build_dataset.py` | the MP3 coefficient-domain track (`xr`/`xr_hat`) |

They are not interchangeable. The paper operates on a **complex STFT of the
decoded waveform**, and at 16/32 kbit/s LAME resamples internally (to 22.05 and
16 kHz), so `xr` and `xr_hat` both live at the reduced rate — a model trained on
granule pairs could never restore the lost bandwidth, which is the paper's main
result. 64 kbit/s mono is the one bitrate where both representations are valid,
making it the natural point to compare them.

## The two kinds of pair

**Granule pairs** — the training data. 576 coefficients per granule, in MP3's own
hybrid transform domain:

| side | what it is |
|---|---|
| `xr` | clean target: the encoder's **pre-quantization** coefficients |
| `xr_hat` | degraded input: the decoder's **dequantized** coefficients |

Both sit at the same point in the MP3 signal chain — post-alias-reduction,
pre-quantization — so they differ only by the quantization and Huffman round
trip. Read straight out of LAME's built-in analysis path by
`native/lame_dump/mp3dump.c`, rather than reimplementing the hybrid transform and
bitstream parser. See `native/lame_dump/README.md`.

**Waveform pairs** — original PCM against MP3-decoded PCM. Not training data;
needed because `xr` is the transform of LAME's *pre-processed* input, so a
perfect `xr_hat -> xr` model still does not reconstruct the original audio. Final
evaluation has to return to the waveform and compare against the true original.
These are free: you already have both files.

A third option was considered and dropped — a plain single-stage MDCT of the two
waveforms. It is the waveform pair under a fixed invertible transform, so it
carries no extra information, and measurement showed it represents the
degradation *worse* than granule pairs (relative RMS at 320k: 5.6e-3 for
granule pairs vs 2.8e-2), its error being inflated by the filterbank round-trip
and encoder delay on top of the actual quantization.

## Setup

```sh
python3 -m venv .venv && ./.venv/bin/pip install -e ".[dev]"
bash native/lame_dump/setup_lame.sh     # fetch + checksum + build pinned LAME
bash native/lame_dump/build.sh          # build the dumper against it
```

Requires `ffmpeg` and `lame` on PATH.

`setup_lame.sh` downloads upstream **lame-3.100**, verifies its MD5
(`83e260acbe4389b54fe08e0bdbf7cddb`), and builds it into `third_party/`
(gitignored). The oracle is deliberately **not** built against a hand-edited
LAME checkout: `xr`/`xr_hat` are only meaningful relative to a known encoder, and
the result has to be reproducible from a verifiable source. Override with
`LAME_DIR` only if you know why.

## Commands

```sh
# Validate the oracle — run this before trusting any generated data
python scripts/validate_oracle.py                      # synthetic signal
python scripts/validate_oracle.py path/to/audio.wav --seconds 20

# Dump granule pairs for one file at one bitrate
python scripts/dump_granules.py INPUT.wav --bitrate 320 --seconds 30

# Build the granule-pair dataset
python scripts/build_dataset.py --tracks 10            # pilot
python scripts/build_dataset.py                        # full train split

# Build + audit the waveform-pair dataset (the GAN reproduction)
python scripts/build_waveform_dataset.py --stems --workers 12
python scripts/verify_waveform_dataset.py [--repair]

pytest tests/ -q
ruff check . && ruff format .
```

## Waveform-pair layout

For the GAN reproduction: original 44.1 kHz mono against its MP3-decoded self at
16/32/64 kbit/s, matching the paper's setup.

```
<root>/
  index.json
  clean/<track>__<stem>.npy        44.1 kHz mono int16, canonical
  bitrate=16/<track>__<stem>/
    degraded.npy                   same length as clean, PRE-ALIGNED
    audio.mp3
    meta.json                      lag + alignment residual
```

Exact MP3 settings, what the paper specifies, and every choice it left open are
recorded in **[DATASET.md](DATASET.md)**, and embedded in each build's
`index.json` under `encoder_settings`.

**Alignment is resolved at build time**, so a training loop slices both arrays at
identical offsets with no index arithmetic. This matters more than it sounds:
MP3 carries an inherent 1105-sample round-trip delay (576 encoder + 529 decoder).
LAME records it in a Xing/Info+LAME gapless tag and ffmpeg trims it — but below
32 kbit/s LAME emits MPEG-2/2.5 and **writes no tag at all**, so the delay
survives at the native rate and scales on resampling (2210 at 32k, 3046 at 16k). Pairing
naively there gives relative error above 1.0, i.e. worse than pairing the clean
signal against silence, and nothing crashes to tell you:

| bitrate | lag | naive pairing | aligned | penalty |
|---|---|---|---|---|
| 64k | 0 | 0.072 | 0.072 | 1.0x |
| 32k | 2210 | 1.518 | 0.113 | **13.5x** |
| 16k | 3046 | 1.121 | 0.154 | **7.3x** |

The lag is measured on the **loudest** region of each file, not the first
samples: many musdb stems open with digital silence, and correlating silence
against silence produces a spurious peak — this caused real off-by-one lags
before it was fixed. `make_pair` raises if the aligned pair is still implausibly
far apart, and raises `SilentAudioError` for sources with no usable signal (a
number of musdb stems are wholly silent), so neither can reach the dataset.

`scripts/verify_waveform_dataset.py` audits a finished build: it checks that the
lag is identical across every track at a bitrate, that clean and degraded agree
in length, that re-measuring the stored pair gives residual lag 0, and that no
stored error is near 1.0. `--repair` rebuilds any entry that fails.

STFT (window 2048, hop 512 in the paper) is computed at train time, so those stay
hyperparameters.

## Granule-pair layout

Built by `scripts/build_dataset.py` into a directory **alongside** the source
corpus, one folder per bitrate and one per track:

```
<root>/
  index.json                         every (track, bitrate) entry + split
  bitrate=128/
    <track name>/
      xr.npy        (n_granules, n_channels, 576) float32   clean target
      xr_hat.npy    (n_granules, n_channels, 576) float32   degraded input
      sideinfo.npy  (n_granules, n_channels)      structured
      audio.mp3     the compressed audio itself
      meta.json     encoder settings + build-time alignment check
```

Arrays are stored **whole per track, not pre-chunked**, and read with
`np.load(path, mmap_mode="r")`. That keeps chunk length a training
hyperparameter rather than a property of the data, and lets a DataLoader sample
random offsets so chunk boundaries differ every epoch — fixed cuts would give
only ~21,500 distinct examples per bitrate per channel and would systematically
under-represent transients landing near a boundary.

`write_track` refuses to write a track whose granule join is misaligned, so a
bad pair cannot reach the dataset silently. Splits are assigned **by track**,
never by chunk, so no song appears in both train and val.

### Reading chunks

`TrackReader` addresses chunks by number, or by arbitrary offset:

```python
from mp3ml.dataset import TrackReader, build_chunk_index, read_index

r = TrackReader(track_dir, chunk_granules=64)  # 64 granules = 0.84 s
r.n_chunks  # 284
r.chunk(100)  # -> xr (64, 2, 576)
r.chunks([100, 101, 250])  # -> xr (3, 64, 2, 576)
r.window(offset=12345, length=64)  # any offset, not chunk-aligned
```

`chunk_granules` is a read-time argument, so the same files serve any chunk
length — 1 granule (13 ms) through 256 (3.3 s) — with no regeneration.

For a Dataset, `build_chunk_index(root, chunk_granules, split=..., bitrates=...)`
returns every addressable chunk as a structured array of
`(entry, chunk, bitrate_kbps)`, so `__getitem__(i)` is a direct lookup with no
directory scan. Reads run ~3 ms per 0.84 s stereo chunk with readers cached.

Because `sideinfo.npy` carries `block_type` per granule, chunks can be filtered
or oversampled by content — e.g. the ones containing transients (short blocks),
which are only ~4% of granules but are the hard case.

**Storage:** `xr` is one float per input sample per channel, and float32 is
required — float16 corrupts 32% of the `xr - xr_hat` residual at 320 kbps, which
is the very signal being learned. Stereo at 4 bitrates costs ~1.4 GB per
audio-minute; the full 94-track musdb train split is ~61 GB.

**Bitrates:** mono is safe down to 64 kbps, but **stereo only down to 112 kbps** —
below that LAME resamples (96k -> 32 kHz, 64k -> 24 kHz) and `xr` stops
corresponding to the input. Defaults are 320/192/128/112 for stereo and
320/192/128/64 for mono.

## What is verified

`scripts/validate_oracle.py` runs five checks; all pass on both synthetic signals
and real music:

1. **Encoder settings** — with the analysis path off, the dumper's bitstream is
   byte-identical to the stock `lame` CLI at every bitrate.
2. **Decoder agreement** — mpglib's decode matches ffmpeg's to ~1.5e-4.
3. **Degradation trend** — `RMS(xr - xr_hat)` rises monotonically as bitrate
   falls (on real music: 4.6e-3 at 320k → 1.0e-1 at 64k).
4. **Granule alignment** — encoder- and decoder-side side info agree exactly
   (1.0000) for every granule. This is the sharpest check available: the two
   come from different code paths, so a one-frame join error collapses it.
5. **`xr` consistency across bitrates** — after dividing out LAME's per-bitrate
   input scale, `xr` agrees across bitrates to ~4e-7 (float32 precision).

### Constraints on leaning fully on LAME

- **Stay at CBR >= 64 kbps.** Below that LAME silently resamples (48k -> 32 kHz;
  32k -> 22.05 kHz and MPEG-2 LSF with 576-sample frames and 1 granule), which
  changes what `xr` even means. `dump_granules` now raises rather than returning
  such data; pass `allow_resample=True` only deliberately.
- **VBR works** — alignment stays exact (1.0000) at every `-V` level and VBR
  applies no input attenuation — but `-V7` and below resample too, so the same
  constraint applies.
- **Mono only.** Stereo (MS/intensity) is not handled yet.
- **LAME-only data.** Every pair comes from LAME, so a model trained on this may
  not generalize to Fraunhofer/iTunes/Shine-encoded MP3s. Nothing in this
  approach can fix that — other encoders expose no equivalent hook.
- Enabling the analysis path changes bitstream *bytes* at 320/192 kbps but has
  no meaningful effect on actual distortion (measured ratios 0.993 / 1.001 /
  1.000 vs analysis off), so there is no train/test mismatch.

Two properties of the data that follow from this, both documented in detail in
`native/lame_dump/README.md` and recorded in every dump's `.json`:

- LAME **attenuates and lowpasses its input per bitrate** (0.95/0.97/1.00, and
  16.5 kHz at 64 kbps), so `xr` is the transform of LAME's pre-processed input,
  not of the pristine original. Pairs are internally consistent; cross-bitrate
  comparisons must divide `input_scale` out.
- Enabling LAME's analysis path **changes the bitstream at 320 and 192 kbps**
  (not at 128/64). Shown by experiment to be a behavioural effect, not memory
  corruption — a guard-band canary test found no out-of-bounds writes. The
  output is still a real LAME MP3 and the pair is still internally consistent.

## Layout

```
src/mp3ml/         audio loading, the oracle wrapper, reference encode/decode
native/lame_dump/  the C dumper that reads LAME's analysis path
scripts/           thin CLI entry points
tests/             fast unit tests — synthetic audio only, no corpus
outputs/           generated data (gitignored)
```

## Not done yet

Stereo (mono only so far), the multi-file dataset builder and manifest, and the
clean NumPy reimplementation of the hybrid transform validated bin-for-bin
against the oracle (plan Phase 2). `is[576]`, the raw quantized integers, is not
in `plotting_data` — exposing it needs one field added to LAME's struct plus one
`memcpy`.
