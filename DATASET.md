# Waveform-pair dataset: provenance

How the MP3 pairs were generated, what the reference paper specifies, and every
place we had to make a choice it left open.

Reference: Lattner, S.; Nistal, J. *Stochastic Restoration of Heavily Compressed
Musical Audio Using Generative Adversarial Networks.* **Electronics 2021**, 10,
1349. <https://doi.org/10.3390/electronics10111349>

## What the paper specifies

The entirety of its encoding description is §3.3:

> The model is trained on pairs of audio data, where one part is the MP3 version,
> the other part is a high-quality (44.1 kHz) version of the signal. We use a
> dataset of approximately 64 hours of Nr 1 songs of the US charts between 1950
> and 2020. The high-quality data is then compressed to 16kbit/s, 32kbit/s and
> 64kbit/s mono MP3 using the LAME MP3 codec, version 3.100.

So it fixes four things: **16/32/64 kbit/s**, **mono**, **LAME 3.100**, and a
**44.1 kHz** reference. Everything else — rate-control mode, quality setting,
lowpass, downmix method, decoder, resampling policy, alignment — is unspecified.

Its representation (§3.3.1) is the **complex STFT, window 2048, hop 512**, with
signed-square-root scaling applied to each component, on 4-second segments with
50% overlap. That is why this dataset stores waveforms and not MP3 coefficients:
see "Why not granule pairs" below.

## What we do

| step | exact command / method |
|---|---|
| source | musdb18hq `train/` + `test/`, mixture + 4 stems, 44.1 kHz stereo WAV |
| downmix | arithmetic mean of L/R → mono, rounded to int16 (`mp3ml.audio.load_pcm`) |
| encode | `lame --quiet -m m --cbr -b {16,32,64} in.wav out.mp3` |
| decode | `ffmpeg -v error -i out.mp3 -f s16le -acodec pcm_s16le -ac 1 -ar 44100 -` |
| align | FFT cross-correlation, stored pre-aligned (`mp3ml.waveform.make_pair`) |
| store | `clean.npy` + `degraded.npy`, int16, identical length |

Versions on this machine: **LAME 3.100** (matching the paper), **ffmpeg 6.1.1**.

The downmix is computed **once** and used as both the clean target *and* the
encoder input, so the pair cannot drift apart through differing downmix paths.

### What LAME resolves those settings to

Reported by LAME itself. Note the **lowpass sits well below the resampled Nyquist
rate**, so the usable bandwidth is narrower than the output sample rate suggests:

| bitrate | output rate | MPEG version | lowpass transition | quality |
|---|---|---|---|---|
| 16 kbit/s | 16 kHz | MPEG-2 Layer III | 5484 – 5677 Hz | qval=3 |
| 32 kbit/s | 22.05 kHz | MPEG-2 Layer III | 8269 – 8535 Hz | qval=3 |
| 64 kbit/s | 44.1 kHz | MPEG-1 Layer III | 16538 – 17071 Hz | qval=3 |

At 16 and 32 kbit/s LAME **resamples internally**; the decode is then resampled
back to 44.1 kHz so the pair lines up with the reference. This is implied by the
paper (its model outputs 44.1 kHz) but not described there.

## Choices the paper left open

Each of these could differ from what Lattner & Nistal did. Recorded so the gap is
visible rather than buried.

1. **CBR.** We pass `--cbr` explicitly. "compressed to 16kbit/s" reads as constant
   bitrate, and LAME's `-b` defaults to CBR anyway, so this is the natural reading
   — but ABR (`--abr`) would also satisfy the wording.
2. **Default quality, `qval=3`.** LAME's CBR default. The paper names no `-q`/`-V`.
3. **Downmix by arithmetic mean** of L/R before encoding. The paper says the MP3s
   are mono but not how stereo sources were reduced, nor whether its references
   were mono to begin with.
4. **ffmpeg as the decoder.** The paper names no decoder. ffmpeg honours LAME's
   gapless tag, which matters for alignment (below).
5. **Resampling back to 44.1 kHz** with ffmpeg's default resampler (swresample).
6. **Alignment** — see below. The paper does not mention it at all.
7. **Corpus.** musdb18hq (9.4 h of mixtures, 47 h including stems) instead of
   their ~64 h of US chart #1 songs, which is proprietary and unavailable.

## Alignment: our addition

> **READ THIS FIRST IF THE MODEL UNDERPERFORMS.**
>
> The degraded audio in this dataset is **time-shifted** to strip MP3's codec
> priming delay. This is deliberate and correct, but it is invisible in the data
> and it is the single most likely thing to be blamed — or wrongly exonerated —
> when results look off. Three things to check, in order:
>
> 1. **Is the evaluation input trimmed the same way as training?** If not, a
>    phase-sensitive metric gives relative error above 1.0 — worse than
>    predicting silence.
> 2. **Is the plain-MP3 baseline sane?** Unaligned, it scores 1.435 against the
>    reference. A broken-looking baseline means the alignment is wrong, not the
>    model.
> 3. **LSD will not show you any of this.** It is nearly shift-invariant
>    (1.02x inflation from a full 3046-sample misalignment), so a misalignment
>    hides behind normal-looking LSD numbers. Test with a phase-sensitive metric.
>
> The trim needs **no reference audio**: read the MP3's sample rate, check for
> the gapless tag, then `lag = 0 if tag else round(1105 * 44100 / native_rate)`.
> Verified to match the measured lag on every file tested, so aligning at
> evaluation time is not "peeking at the ground truth".

The paper never mentions aligning the pair, and it matters a great deal.

MP3 has an inherent round-trip delay of **1105 samples** (576 encoder + 529
decoder). LAME normally records it in a Xing/Info + LAME gapless tag, and ffmpeg
honours it — at 44.1 kHz it reports `start_time = 0.025057` s, exactly 1105/44100,
and trims, giving lag 0.

Below 32 kbit/s LAME switches to MPEG-2/2.5 and **writes no Xing/Info/LAME tag at
all**. Nothing tells the decoder to trim, so the full 1105-sample delay survives
at the codec's native rate, and resampling back to 44.1 kHz scales it:

| bitrate | native rate | tag | lag at native rate | scaled to 44.1 kHz |
|---|---|---|---|---|
| 64 kbit/s | 44100 | `Info`+`LAME` | 0 (trimmed) | 0 |
| 32 kbit/s | 22050 | **none** | 1105 | 1105 x 2 = **2210** |
| 16 kbit/s | 16000 | **none** | 1105 | 1105 x 2.756 = **3046** |

The resampler itself adds essentially nothing (measured 0.0 and 0.3 samples);
ffmpeg compensates its own filter delay. Measured on musdb:

| bitrate | lag (samples) | naive pairing | aligned | penalty |
|---|---|---|---|---|
| 64 kbit/s | 0 | 0.072 | 0.072 | 1.0x |
| 32 kbit/s | 2210 | 1.518 | 0.113 | **13.5x** |
| 16 kbit/s | 3046 | 1.121 | 0.154 | **7.3x** |

A relative error above 1.0 means the "degraded" signal is *less* correlated with
the reference than silence would be. Training on that learns nothing, and nothing
crashes to announce it.

The lag proved constant per bitrate across every track tested (a property of the
resampler, not the content), but it is **measured per file anyway** and the
degraded signal is stored already shifted, so a training loop slices both arrays
at identical offsets. `make_pair` raises if the aligned pair is still implausibly
far apart, so a failed lag search cannot silently enter the dataset.

### Is the shift a modification of the audio?

No, and this is worth being able to state precisely, because it is the obvious
objection to the whole approach:

- It is a **pure index shift** — `stored[:n] == raw[lag:lag+n]` holds exactly.
  No sample is filtered, resampled or rescaled.
- What it discards is **digital silence** (measured RMS 0.000 over the 3046
  trimmed samples) — the codec's priming, not music.
- It is **what a correct decoder already does**. At 64 kbit/s ffmpeg performs
  this exact trim itself from the gapless tag. We apply it by hand at 16 and
  32 kbit/s only because LAME writes no tag there. The result is that all three
  bitrates are treated identically.
- MP3 adds junk at **both** ends: for one 9,675,225-sample track the decode is
  9,679,598 samples — 3046 of priming at the head and 1327 of frame padding at
  the tail.

A restoration model is shift-equivariant, so an untrimmed input simply yields a
correspondingly shifted output; verified at rel diff 9.6e-05 against the trimmed
path. Trimming costs nothing at inference and removes a nuisance parameter from
training, where the lag otherwise differs per bitrate (3046 / 2210 / 0) and the
model would have to infer the bitrate to undo it.

If you change decoder or resampler, re-verify — though among quality resamplers
it barely matters: soxr and ffmpeg's swresample differ by 3e-05, against an MP3
degradation of 0.127. (A deliberately poor resampler such as `filter_type=cubic`
differs by 0.59, so the check is still worth running.)

### Generalization caveats

What the model will and will not have seen, for anyone evaluating on real files.
None of these are caused by the alignment; they follow from the paper's setup
and the corpus:

1. **Mono only.** Real MP3s are overwhelmingly stereo, and joint-stereo coding
   produces artifacts (M/S coupling, stereo image collapse) absent from this
   data. The largest gap.
2. **LAME only.** Fraunhofer, iTunes and Shine have different psychoacoustic
   models and leave different artifacts.
3. **CBR only.** Much real-world MP3 is VBR, where bit allocation varies per
   frame.
4. **First generation only.** Sources here are lossless; many MP3s in the wild
   are transcodes of other lossy files, with compounded artifacts.
5. **44.1 kHz sources only**, and musdb's genre distribution is indie/rock-heavy.

## Why not granule pairs

The other dataset in this repo (`scripts/build_dataset.py`) stores MP3's internal
pre/post-quantization coefficients (`xr`/`xr_hat`). That cannot serve this paper.

At 16 and 32 kbit/s LAME resamples, so **both** `xr` and `xr_hat` live at the
reduced rate. A model trained on them could only undo quantization inside the
surviving band; it could never restore the 5.6–22 kHz that the lowpass and
resampling removed, which is precisely what the paper's GAN is for.

**64 kbit/s mono is the one bitrate where both representations are valid** — no
resampling, and the paper tested it — making it the natural point to compare
STFT-domain against coefficient-domain restoration.

## Corpus and splits

- musdb18hq `train/` (94 tracks) and `test/` (50 tracks), official splits.
- Mixture plus the 4 stems (bass, drums, other, vocals) per track: **719 files**
  (one track is missing its `bass.wav` and is skipped with a warning).
- Splits are honoured **by track**, so no song appears in two splits. Stems stay
  with their own mixture's split — they are correlated with it, and separating
  them would leak.
- Stems are training augmentation, not pretend mixtures: isolated stems have very
  different spectra from full mixes. **Evaluate on mixtures only.**

## Known deviations from the paper

| | paper | here |
|---|---|---|
| corpus | ~64 h, US chart #1 songs (proprietary) | musdb18hq, 9.4 h mixtures / 47 h with stems |
| splits | 80/10/10 by song | musdb official train/test, by track |
| segments | 4 s, 50% overlap | chunk length chosen at train time |
| everything in "Choices the paper left open" | unspecified | as recorded above |
