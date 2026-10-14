# `mp3dump` — the LAME oracle

Extracts genuine, granule-aligned `(xr, xr_hat)` pairs directly out of LAME,
without reimplementing the MP3 hybrid transform, Huffman decoding or
dequantization. (`mp3_dataset_plan.md` calls these "Representation C"; the code
and docs here just call them granule pairs.)

## Why this works

LAME compiles in an "analysis" path (used by its `mp3x` GTK frame analyzer) that
exports two arrays into `struct plotting_data` (`libmp3lame/lame-analysis.h`):

| field | written at | what it is |
|---|---|---|
| `xr[gr][ch][576]` | `libmp3lame/encoder.c:472` | encoder's **pre-quantization** coefficients |
| `mpg123xr[gr][ch][576]` | `mpglib/layer3.c:1945` | decoder's **dequantized** coefficients |

The encoder copy happens after Stage 2 (`mdct_sub48`) and before Stage 4 (the
quantization loop). The decoder copy happens immediately before `III_antialias`.
Those are **the same point in the signal chain** — post-alias-reduction,
pre-quantization — so the two differ only by the quantization and Huffman round
trip. That is exactly the clean/degraded pair the project needs.

Side info also comes from both sides, which gives a free exact-equality check:
`block_type`/`enc_global_gain` are encoder-side, `dec_block_type`/`global_gain`
are decoder-side and read back out of the real bitstream.

## Build

```sh
bash setup_lame.sh                 # once: fetch + checksum + build pinned LAME
bash build.sh                      # build the dumper against third_party/lame-3.100
LAME_DIR=/path/to/other-lame bash build.sh
```

`setup_lame.sh` pins upstream **lame-3.100** (MD5
`83e260acbe4389b54fe08e0bdbf7cddb`) and asserts `NOANALYSIS` stays undefined —
if it were defined, the analysis path that exports `xr`/`xr_hat` would be
compiled out and the dumper would silently produce nothing useful.

Build against a pinned, unmodified tree rather than a working checkout: `xr` and
`xr_hat` only mean something relative to a known encoder. A local edit to the
MDCT or quantizer would change the data without changing any test.

LAME source is fetched into gitignored `third_party/`, never committed here.
`mp3dump.c` includes LAME's internal headers and statically links
`libmp3lame.a`, so it is a derived work under LGPL-2.1 and is kept isolated in
this directory. Static linking is required because `hip_set_pinfo` is declared
in the internal header `libmp3lame/util.h` and is not exported from the shared
library.

## Usage

```sh
mp3dump --bitrate 320 --samplerate 44100 --out-prefix OUT < mono_s16le.pcm
```

Writes `OUT.bin` (fixed-size records, see `RECORD_DTYPE` in `src/mp3ml/oracle.py`),
`OUT.json` (metadata) and `OUT.mp3`. Options: `--pcm-out` also writes mpglib's
decoded PCM; `--no-analysis` encodes with the analysis path off and emits no
records (a diagnostic for comparing bitstreams).

## Three traps this code exists to avoid

**1. `pinfo->xr` is declared `double[576]` but holds `float`s.** `encoder.c:472`
does `memcpy(..., sizeof(FLOAT) * 576)` where `FLOAT` is `float`
(`machine.h:131`), so only the first 2304 bytes of a 4608-byte array are filled,
with floats. Reading it as `double` yields near-zero garbage. Stock LAME's own
`mp3x` analyzer has this bug (`gtkanal.c:562`). `mpg123xr` is *not* affected —
`layer3.c` assigns it element-wise as `double`.

**2. The encoder/decoder frame lag.** `gtkanal.c` interleaves encoding and
decoding, so in any one pass `pinfo->xr` is frame `N` while `pinfo->mpg123xr` is
frame `N - mpglag`. Pairing them naively silently misaligns every record. This
program sidesteps it: encode everything first, then decode the finished
bitstream from frame 0, so decoded frame `N` is encoder frame `N` by
construction. The Xing/Info header frame is disabled (`bWriteVbrTag(0)`) so the
bitstream holds audio frames only and indices line up.

**3. `lame_encode_flush` can encode several frames at once.** `pinfo->xr` is a
single buffer that each frame overwrites, so the earlier of those is
unrecoverable. Stashing during flush would mislabel a frame's `xr`. Those
trailing frames are encoder padding rather than real audio, so they are dropped:
only frames stashed before the flush are emitted.

## Three properties of the data you must know

**LAME pre-processes its input, per bitrate.** It attenuates by `scale` and
applies a lowpass *before* the transform, so `xr` is the transform of LAME's
pre-processed input, not of the pristine original:

| bitrate | input scale | lowpass |
|---|---|---|
| 320, 256 | 1.00 | 20500 Hz |
| 192 | 0.97 | 20500 Hz |
| ≤ 160 | 0.95 | 20500 Hz |
| 64 | 0.95 | 16500 Hz |

(`libmp3lame/presets.c:294`.) Both are recorded in every `.json` as
`input_scale` and `lowpass_hz`. Within one bitrate the pair is consistent, so
training pairs are unaffected; comparing `xr` *across* bitrates requires dividing
the scale back out. Verified by check 5 in `scripts/validate_oracle.py`: after
rescaling, `xr` agrees across bitrates to ~4e-7 (float32 precision).

**LAME silently resamples below 64 kbps, and can switch to MPEG-2 LSF.** The
output sample rate is not necessarily the one you asked for:

The safe floor depends on the **channel count**, since what matters is bits per
channel:

| CBR | mono out_samplerate | stereo out_samplerate |
|---|---|---|
| 320 / 256 / 192 / 160 / 128 / 112 | 44100 | 44100 |
| 96 / 80 | 44100 | **32000** |
| 64 | 44100 | **24000** |
| 48 | **32000** | — |
| 32 | **22050** (MPEG-2 LSF: 576-sample frames, 1 granule) | — |

**Mono is safe to 64 kbps; stereo only to 112 kbps.**

VBR does the same at low quality (`-V7` -> 32 kHz, `-V9` -> MPEG-2 LSF), though
VBR applies no input attenuation at any level. When this happens `xr` covers a
different frequency range and no longer corresponds 1:1 to the input samples, so
such pairs must not be mixed into a 44.1 kHz dataset unnoticed. `out_samplerate`
is recorded in every `.json`, and `dump_granules` raises unless
`allow_resample=True`. **Stay at CBR >= 64 kbps.**

**Enabling the analysis path changes the bitstream at high bitrates.** With
analysis off, `mp3dump` is byte-identical to the stock `lame` CLI at every
bitrate. With it on, the bitstream differs at 320 and 192 kbps (identical at 128
and 64). Established by experiment: the change appears only when `cfg->analysis`
is set *and* `gfc->pinfo` is non-NULL — that is, only when LAME actually writes
into `pinfo`. A 1 MB guard-band canary test around the `plotting_data` object
showed **no out-of-bounds writes**, so this is not memory corruption; the extra
per-frame work `set_frame_pinfo` does perturbs an encoder decision somewhere.

This does not invalidate the data: the output is still a real LAME-encoded MP3,
and `xr_hat` is still the dequantization of the quantization of that same `xr`
(confirmed by exact side-info agreement and per-granule correlation). It does
mean the MP3 produced in analysis mode is not bit-identical to what stock LAME
would produce at 320/192 kbps. Check 1 in `validate_oracle.py` reports the
magnitude of the perturbation on every run. Measured separately: the perturbation
has no meaningful effect on actual distortion — decoded-vs-original error ratios
of 0.993 / 1.001 / 1.000 at 320 / 192 / 128 kbps versus analysis off — so there
is no train/test mismatch.

Alignment is exact (1.0000) under VBR at every `-V` level too, so the frame join
is not CBR-specific.

**Stereo needs no M/S handling.** Under joint stereo both sides are L/R: the
decoder converts M/S -> L/R at `mpglib/layer3.c:1839`, *before* the `mpg123xr`
dump at 1945, and the encoder's `ms_convert` runs inside the Stage 4 iteration
loop (`quantize.c:1412`), *after* the `pinfo->xr` copy at `encoder.c:472`.
Verified empirically: stereo alignment is 1.0000 and per-granule correlation
>= 0.96 at every bitrate from 320 down to 112 kbps.
