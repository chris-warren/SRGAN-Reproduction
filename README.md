# SRGAN-Reproduction

A PyTorch reproduction of:

> Lattner, S.; Nistal, J. **Stochastic Restoration of Heavily Compressed Musical
> Audio Using Generative Adversarial Networks.** *Electronics* 2021, 10, 1349.
> <https://doi.org/10.3390/electronics10111349>

The paper restores detail destroyed by heavy MP3 compression with a conditional
WGAN-GP: a generator sees the MP3's complex STFT and emits a restored one, and a
critic sees both the candidate and the MP3 it came from, so the restoration has
to stay faithful to its input rather than merely sound plausible.

There is **no official implementation**. The Data Availability Statement covers
audio examples only. The authors answered our open questions by email on
2026-09-29, and those answers are recorded verbatim in
[`configs/REPRODUCTION.md`](configs/REPRODUCTION.md) §6.2.

## Status

| | |
|---|---|
| model, losses, training loop | implemented, asserted against Table 1 |
| the paper's §3.4 metrics (eq. 5–7) | implemented and audited against the paper text |
| §4.1 best-of-20 evaluation | implemented |
| **a full training run** | **not done.** `faithful.yaml` has never been executed at any scale |
| Table 2 | needs six models (3 rates x {stochastic, deterministic}); only 64 kbit/s is reachable from this corpus |
| PEAQ (ODG, DI) | binary not installed — 2 of Table 2's 5 columns unobtainable |
| MOS listening test | not run |

A long run on the reduced config diverged at step 4609. The diagnosis, including
**two measurement bugs of our own** that invalidated earlier conclusions, is in
`REPRODUCTION.md` §4a–§4f. That file is the honest record: every claim is marked
**[P]** paper / **[I]** inferred / **[A]** assumption / **[D]** deviation, and
retracted findings are kept rather than quietly deleted.

## The data, and why only one bitrate works

Training reads the **Apollo** preprocessed corpus, not a corpus of our own,
because the goal is a like-for-like comparison against Apollo
([Li et al.](https://github.com/JusperLee/Apollo)) — another codec-restoration
model — and both must see identical audio.

Apollo stores **originals only** (`hdf5_train/<stem>/*.hdf5`, shape
`(n_windows, 2, 264600)` float32: 6 s stereo at 44.1 kHz) and applies MP3 inside
its own loader, so this reproduction degrades at training time too.

**Only 64 kbit/s is usable, and that was measured rather than chosen:**

| kbit/s (mono) | MP3 output rate | in Apollo | in the paper |
|---|---|---|---|
| 24 | 16 000 Hz — resampled | yes | no |
| 32 | 22 050 Hz — resampled | yes | **yes** |
| 48 | 32 000 Hz — resampled | yes | no |
| **64** | **44 100 Hz — clean** | **yes** | **yes** |
| 96, 128 | 44 100 Hz — clean | yes | no |

This generator is single-channel, and in mono LAME silently resamples below
64 kbit/s, which makes the pair a different degradation from the paper's and
doubles MP3's priming delay. `apollo.verify_constant_delay` caught 32 kbit/s as
a +1105 residual lag after the usual trim. So 32 kbit/s is in both papers'
bitrate sets and still unusable, and **16 kbit/s — the paper's headline case —
is absent from this corpus entirely.** `apollo.degrade` raises on the unsafe
rates rather than quietly producing misaligned pairs.

Apollo's own `codec_simu` leaves 7–15% of its pairs out of step with their
targets (its README says so). Ours trims MP3's constant priming delay and
verifies the residual is zero, and the evaluation clips carry a per-clip
`residual_lag_samples` that `ApolloEvalPairs` **corrects** rather than
discarding — recovering the whole set instead of dropping 2% of it.

## Setup

```sh
python3 -m venv .venv
./.venv/bin/pip install -e ".[apollo,dev]"      # apollo extra = h5py
```

`ffmpeg` and `lame` must be on PATH. Optional extras: `tracking` for Weights &
Biases. Put `WANDB_API_KEY=...` in a gitignored `.env` at the repo root, or set
`WANDB_MODE=offline`; metrics always land in `outputs/<run>/metrics.jsonl`
regardless, and that JSONL is the durable record.

## Commands

```sh
pytest tests/ -q                                 # synthetic audio only, no corpus
ruff check . --fix && ruff format .

# Size the hardware BEFORE any long run. Reports step time, peak memory, and
# whether the GPU or the data pipeline is the bottleneck.
python scripts/bench_cluster.py configs/faithful.yaml --workers 16

python scripts/train.py configs/smoke.yaml --max-steps 4          # plumbing
python scripts/train.py configs/faithful.yaml --bitrate 64 --tracker wandb
python scripts/evaluate.py CKPT --split test                      # best-of-20
```

Pass `--bitrate` — the paper trains **one model per rate**, and omitting it is a
different experiment.

## Why the data pipeline may be the bottleneck

On-the-fly MP3 costs about **0.8 s per item** (measured), so a batch of 12 needs
~9.5 CPU-seconds. Below roughly 12 dataloader workers the GPU waits, and a
faster GPU does not help:

| workers | s/step from data | 40 000 steps |
|---|---|---|
| 8 | 1.5 | 16.7 h |
| 16 | 0.75 | 8.3 h |
| 32 | 0.38 | 4.2 h |

Pre-building the degraded audio once (~31 CPU-hours, 82 GB) removes this
entirely. `scripts/bench_cluster.py` measures which side is binding.

## Layout

- `src/mp3ml/` — `apollo.py` (corpus reader and the bitrate reasoning),
  `models.py` (Table 1), `losses.py` (eq. 1, eq. 3, gradient and drift
  penalties), `train.py`, `stft.py` (the paper's signed-sqrt complex STFT),
  `metrics.py` (eq. 5–7 plus modern additions), `evaluate.py`, `align.py`,
  `config.py`, `tracking.py`, `peaq.py`, `fad.py`
- `configs/` — `faithful.yaml` (Table 1 exactly), `local.yaml` (8 GB card),
  `smoke.yaml` (toy), plus `REPRODUCTION.md` and `COVERAGE.md`
- `scripts/` — thin CLI entry points
- `.claude/rules/` — the checks that shaped this work, notably
  `faithfulness.md` (four bugs it caught that running the code would not have)

## A note on what this repo used to be

It began as an MDCT experiment: pairing LAME's pre-quantization `xr[576]` against
the decoder's dequantized `xr_hat[576]`, read straight out of the encoder's
analysis path. That work — the C dumper, the oracle, and the waveform-pair
builder that fed this reproduction before it moved to Apollo's corpus — lives at
`/home/chris/thesis/mdct-granule-pairs` with its own README and history. None of
it was defective. The last commit here containing it is tagged
`mdct-granule-phase`.
