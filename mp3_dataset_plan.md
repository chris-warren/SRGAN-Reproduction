# MP3 Compression-Reversal: Dataset Preparation Plan

## 1. Context

**End goal (later phase):** train a model that reverses MP3 compression —
i.e. given lossy MP3-decoded audio, reconstruct something closer to the
original uncompressed audio.

**This phase's goal:** prepare training data, in multiple candidate input
representations, and empirically determine which representation(s) are
worth carrying into model training. This is explicitly a dataset-prep +
evaluation phase, not a modeling phase — no model training happens yet.

**Prior context (see chat history for full derivation):**
- MP3's real pre-quantization format is not a single MDCT. It's a hybrid
  two-stage transform: a 32-band polyphase analysis filter, followed by a
  per-subband MDCT (36-pt long / 12-pt short ×3), followed by an 8-butterfly
  alias-reduction step across subband boundaries. The result, `xr[576]` per
  granule per channel, is what the encoder quantizes.
- The polyphase filter alone is not perfectly lossless (~1e-6 RMS residual
  even without quantization), but this is negligible next to quantization
  noise at any real bitrate. It is not worth dedicated modeling effort.
- **Quantization is the only stage that destroys information.** Every other
  decode stage (Huffman decode, dequantization's *formula*, reorder, stereo
  processing, alias-reduction reversal, IMDCT, synthesis filter) is exact,
  deterministic, and derivable from the bitstream's side info. None of them
  need to be learned.
- `xr` (encoder, pre-quantization) and `xr_hat` (decoder, dequantized) are
  the *same representation* — same 576-value granule/channel array, same
  point in the signal chain — differing only by quantization + Huffman
  round-trip. This makes them a valid (clean, degraded) training pair.
- Getting genuine `xr` for arbitrary original audio does **not** require
  patching a real encoder. The only heuristic/encoder-specific choice in
  the whole pipeline is the block-type (long/short/start/stop/mixed)
  transient-detection schedule — and that's transmitted in every MP3's
  side info, so it can be read rather than guessed. Given that schedule,
  analysis filter → per-subband MDCT → alias-reduction is fixed, spec-
  defined math, appliable directly to the original uncompressed audio.
- Quantization itself, given the scalefactors + global_gain (also sitting
  in the side info), is deterministic and invertible — encoder heuristics
  only decide *which* scalefactors/gain to pick, not the is<->xr mapping.
  This gives a strong, no-encoder-required self-consistency validation
  method (Section 5).
- A promising existing reference for the exact stage-by-stage math: the
  Rust crate [`oxideav-mp3`](https://github.com/OxideAV/oxideav-mp3)
  (MIT, `cargo add oxideav-mp3`). It implements Huffman decode, dequantize,
  reorder, stereo, alias-reduce, IMDCT, and polyphase synthesis as
  separate, spec-cited, independently tested functions, plus an in-progress
  encoder-side analysis filter + forward MDCT. It's young (mono/MPEG-1
  only, "clean-room rebuild in progress" as of mid-2026) — treat it as a
  reference / potential PyO3-bindable dependency to validate against, not
  a black box to trust blindly.

## 2. Representations to generate (the comparison set)

Build all three. The point of this phase is to compare them, not commit
to one yet.

| # | Representation | Clean (target) | Degraded (input) |
|---|---|---|---|
| A | Waveform (baseline) | original PCM | real-MP3-decoded PCM |
| B | Plain single-stage MDCT | `mdct()` of original PCM (reuse `mdct_utils.py` from earlier) | `mdct()` of real-MP3-decoded PCM |
| C | True hybrid coefficient domain | reconstructed `xr[576]` from original PCM (Section 4) | `xr_hat[576]` extracted from the real MP3 bitstream (Section 3) |

A is the standard/simple baseline. B is a cheap intermediate step (already
have working, verified code for it). C is the fidelity-first option and
the most work — Sections 3–5 are all in service of building and validating
C correctly.

## 3. Extracting `is[576]` / `xr_hat[576]` from a real MP3 file

Investigate, in this order, and record effort/reliability for each:

1. **`oxideav-mp3` via PyO3 bindings, or a small Rust CLI wrapper.** It
   already has `huffman::decode_huffman` → `is[576]` and
   `requantize::requantize` → `xr_hat[576]` implemented and tested at the
   granule level. Cheapest path if it's stable enough to depend on. A thin
   Rust binary that walks a file's frames and dumps `is`/`xr_hat` +
   side-info fields (block_type, scalefactors, global_gain, subblock_gain,
   stereo mode) to `.npy`/JSON per granule is likely the lowest-effort
   integration — avoids a full PyO3 build if that's friction.
2. **Existing Python-native or Python-bound decoder exposing internals.**
   Search current PyPI/GitHub before assuming none exists — the standard
   ones (pydub, librosa/audioread, pymad) go straight through ffmpeg/libmad
   to PCM and do **not** expose this, but it's worth a real search rather
   than assuming.
3. **Fallback: implement the parser directly.** Frame header + side info
   parsing is straightforward (~1 page of bit-level parsing per ISO/IEC
   11172-3 §2.4.1). Huffman decode needs the full codebook table set
   (§2.4.2.7, Annex B Table B.7) — tedious but well documented, and
   `oxideav-mp3`'s `huffman` module or the classic reference decoders
   (dist10, mpg123) are usable as a **structural** reference (do not
   copy code with unclear licensing — reimplement from the ISO tables).

**Validate whichever option is chosen** by running the *full* decode chain
(dequantize → reorder → stereo → alias-reduce → IMDCT → synthesis filter)
on its extracted `is[576]` and comparing the resulting PCM against a
trusted reference decoder (`ffmpeg` or `mpg123` CLI) on the same file.
Match within a tight numerical tolerance (e.g. <1e-4 normalized) before
trusting the intermediate `xr_hat` values for anything downstream.

## 4. Reconstructing genuine pre-quantization `xr` from original audio

For a given MP3 file and its known original uncompressed source:

1. Parse the MP3's side info to get, per granule/channel: block_type,
   mixed_block_flag, subblock_gain (if short), scalefactors, global_gain,
   stereo mode.
2. Run the polyphase analysis filter (512-tap prototype, ISO Annex C
   Table C.1) on the *original* PCM to get 32×18 subband-domain samples
   per granule.
3. Apply the per-subband MDCT using the block type read in step 1 (36-pt
   long, or 12-pt ×3 short, or the mixed split) — **must match the real
   file's actual block-type schedule**, not an independently-run transient
   detector.
4. Apply the alias-reduction butterfly (8 pairs, Annex B Table B.9
   coefficients) to get the final `xr[576]`.

This should be implementable by porting `oxideav-mp3`'s `analysis`, `mdct`
(forward path), and `alias` modules, or reimplementing from the same ISO
sections directly in Python/NumPy (the plain-MDCT code from
`mdct_utils.py` is a useful sanity-check reference for the MDCT step, but
note it's a *different, simplified, single-stage* transform — it is not a
drop-in substitute for the real per-subband hybrid transform).

## 5. Validation methodology — do not skip this

Two independent checks, both encoder-source-free:

- **Decode round-trip check** (validates Section 3's extraction):
  `is[576]` (from the real file) → your dequantize → alias-restore →
  IMDCT → synthesis filter chain → compare to a trusted reference
  decoder's PCM for the same file.
- **Quantization self-consistency check** (validates Section 4's
  reconstruction): take your reconstructed `xr` from Section 4, quantize
  it using the *real file's actual* scalefactors + global_gain (spec-
  defined, deterministic formula — see `oxideav-mp3`'s `quantize` module
  or ISO §2.4.3.4.7.1), and check whether the result matches the real
  file's actual `is[576]` bin-for-bin. An exact or near-exact match
  confirms your Section 4 pipeline is correct — this is possible without
  ever seeing the real encoder's source, because quantization is
  deterministic once the scalefactors/gain are known.

Track pass rate of both checks across the corpus and bitrates below —
this is itself a useful data point on which representation is trustworthy
enough to train on.

## 6. Corpus & experiment matrix

- **Start from uncompressed sources**, not found-in-the-wild MP3s — you
  need the true original for every file. A license-clear corpus (e.g.
  LibriSpeech, VCTK, MUSDB18, or FMA) resampled to a single fixed sample
  rate.
- **Mono first.** Matches `oxideav-mp3`'s current scope and halves the
  early complexity; add stereo (MS/intensity stereo decode) once mono is
  validated end-to-end.
- **Multiple bitrates** (e.g. 320 / 192 / 128 / 64 kbps) — quantization
  loss and block-type behavior vary a lot with bitrate, and you want a
  difficulty range for the eventual model, plus a way to see whether
  Section 5's checks hold up equally well at low bitrates (more
  aggressive quantization, more short blocks).
- Encode every source file yourself (e.g. via `ffmpeg -codec:a
  libmp3lame` or the `lameenc` Python wrapper) at each bitrate, so the
  ground-truth original is always known and paired.
- **Complexity staging:** get one file, one bitrate, long-blocks-only
  working and validated end-to-end before generalizing to short/mixed
  blocks and the full corpus.

## 7. Phased checklist

- [ ] **Phase 0** — environment: ffmpeg/lame for encoding, a reference
      decoder (ffmpeg/mpg123) for validation, Rust toolchain if pursuing
      the `oxideav-mp3` route.
- [ ] **Phase 1** — MP3 parsing: frame header + side info + `is[576]`
      extraction. Validate via full decode round-trip (Section 5).
- [ ] **Phase 2** — dequantization: `xr_hat[576]` extraction from `is[576]`.
- [ ] **Phase 3** — clean-`xr` reconstruction pipeline (Section 4) +
      quantization self-consistency validation (Section 5). Start
      long-blocks/mono only.
- [ ] **Phase 4** — generalize Phase 1–3 to short/mixed blocks, then
      stereo.
- [ ] **Phase 5** — waveform-pair (A) and plain-MDCT-pair (B) generation
      — the simpler tracks, mostly reusing existing `mdct_utils.py` code.
- [ ] **Phase 6** — unify into one dataset builder: a manifest (source
      file, bitrate, block-type stats) + cached per-file arrays for all
      three representations, so downstream experiments can swap
      representations without regenerating data.
- [ ] **Phase 7** — comparison report: for each representation, report
      generation cost (time/storage), Section 5 validation pass rates
      across bitrates, and any structural caveats — to decide which
      representation(s) proceed to model training.

## 8. Open questions to revisit later (not blocking this phase)

- Loss domain for eventual model training (waveform/perceptual vs.
  coefficient-space) — decide once representations B and C exist and can
  be compared empirically.
- Whether to also retain `is[576]` itself (pre-dequantization integers)
  as a fourth representation, for a "predict the quantization index"
  framing rather than a continuous-coefficient framing.
- Stereo handling details (MS vs intensity stereo) once mono is solid.
