# Project

A PyTorch reproduction of Lattner & Nistal, *Stochastic Restoration of Heavily
Compressed Musical Audio using GANs* (Electronics 2021, 10, 1349) — model,
losses and training loop. `configs/REPRODUCTION.md` is the living record of
every ambiguity, assumption and deviation and is **required reading before
changing anything under `src/mp3ml/{models,losses,stft,train,config}.py` or
`configs/`**.

**The authors answered our open questions by email on 2026-09-29** — the critic
conditioning, the grouped convolutions, θ, and all four training coefficients
are **[P]**, recorded verbatim in `configs/REPRODUCTION.md` §6.2. Several of our
assumptions were wrong; the critic was rebuilt as a grouped-convolution stack
with an 8-channel input.

**No full training run has been done yet**, and `faithful.yaml` has never been
executed at any scale. A long run on the reduced config diverged at step 4609
(§4a–§4f). Beware short measurement windows: five "stable" conclusions in this
repo's history were artefacts of 25–40 step runs.

## The corpus is Apollo's, and only 64 kbit/s is usable

Training reads the **Apollo** preprocessed corpus
(`/media/chris/Seagate Portable Drive/data/processed/apollo_v1`), because the
point is a like-for-like comparison against Apollo (Li et al.) and both models
must see identical audio. `src/mp3ml/apollo.py` reads it.

Apollo stores **originals only** — `hdf5_train/<stem>/*.hdf5` holds
`"data": (n_windows, 2, 264600)` float32, 6 s stereo at 44.1 kHz — and applies
MP3 in its own loader, so `ApolloPairs` degrades at training time too.

- IMPORTANT: **64 kbit/s is the only usable rate.** Apollo generates
  24/32/48/64/96/128 and the paper uses 16/32/64, but this generator is
  single-channel and in **mono** LAME silently resamples below 64 (24 → 16 kHz,
  32 → 22.05 kHz, 48 → 32 kHz), which changes the degradation and doubles MP3's
  priming delay. `apollo.verify_constant_delay` caught 32 kbit/s as a +1105
  residual lag. 32 is in both papers' sets and still unusable.
  `apollo.degrade` raises on the rest. **16 kbit/s, the paper's headline case,
  is absent from this corpus entirely.** See REPRODUCTION.md §4f.
- IMPORTANT: a spectrogram's leading dimension of 2 is **real/imag**, not audio
  channels. Apollo is stereo; left and right are indexed as **separate items**,
  which keeps Table 1 untouched and doubles the data. Do not add a channel
  dimension to the model.
- Apollo's `codec_simu` leaves **7–15% of its pairs misaligned** (its own
  README). Ours trims MP3's constant priming delay and verifies the residual is
  zero; the evaluation clips carry a per-clip `residual_lag_samples` that
  `ApolloEvalPairs` **corrects** rather than discarding.
- Split datasets **by track, never by chunk** (`.claude/rules/data.md`).

## Setup & commands

- Install: `python3 -m venv .venv && ./.venv/bin/pip install -e ".[apollo,dev]"`
  (the `apollo` extra is h5py, needed to read the corpus). **Nothing is
  compiled** — the only build step this project ever had was the C granule
  dumper, which moved to the archive.
- **`ffmpeg` and `lame` must be on PATH**: `apollo.degrade` shells out to them
  **per training item**, so they are runtime dependencies, not build tools. On a
  cluster, `module load ffmpeg`. `apollo.check_tools` fails at Dataset
  construction rather than at the first batch.
- **Point the corpus somewhere real**: the configs carry the workstation's
  external-drive path. Set `$APOLLO_DATA` (expanded, so `$SCRATCH/apollo_v1`
  works) or pass `--data-root`. A missing root raises at startup.
- Run tests: `pytest tests/ -q` — synthetic audio only, no corpus, no network
- Lint + format: `ruff check . --fix && ruff format .`
- Check a config's size: `python scripts/check_config.py configs/faithful.yaml --measure`
  (`--measure` runs a real step; the analytic estimate under-reports by ~5x)
- Smoke-test the loop: `python scripts/train.py configs/smoke.yaml --max-steps 4`
- **Size the hardware before any long run:**
  `python scripts/bench_cluster.py configs/faithful.yaml --workers 16`
  It reports step time, peak memory, and whether the GPU or the data pipeline is
  binding. Never extrapolate from a development GPU.
- Train: `python scripts/train.py configs/faithful.yaml --bitrate 64 [--resume]`
  (**pass `--bitrate`** — the paper trains one model per rate, §1.3)
- Live tracking: add `--tracker wandb`. The key is read from a **gitignored
  `.env`** at the repo root (`WANDB_API_KEY=...`); a shell export overrides it,
  and `WANDB_MODE=offline` is respected. Metrics always land in
  `outputs/<run>/metrics.jsonl` regardless, and that JSONL is the durable record
- Evaluate a checkpoint: `python scripts/evaluate.py CKPT --split test`
  (paper's best-of-20 protocol on Apollo's packed eval clips, so the numbers are
  comparable to Apollo's as well; `--samples 1` is NOT comparable to Table 2)
- Experiment results go in **`outputs/experiments/`** — gitignored but durable.
  Do not write them to a scratch directory; two sets have already been lost to
  /tmp clears

External tools assumed on PATH: `ffmpeg`, `lame`.

## Repo layout

- `src/mp3ml/` — `apollo.py` (the corpus reader, `ApolloPairs`, `ApolloEvalPairs`
  and the bitrate reasoning), `align.py` (lag estimation, used to prove the
  delay trim is correct), `config.py` (experiment config + architecture
  derivation), `stft.py` (the paper's signed-sqrt complex STFT), `data.py` (the
  shape contract), `models.py` (generator + critic, Table 1), `losses.py`
  (eq. 1, eq. 3, gradient and drift penalties), `train.py` (the training loop),
  `metrics.py` (eq. 5–7, plus SI-SDR / multi-resolution LSD / Frechet
  distance), `evaluate.py` (the §4.1 best-of-20 harness), `peaq.py` (ODG/DI via
  the external `peaqb` binary), `fad.py` (VGGish embeddings), `tracking.py`
  (JSONL metric log, optional W&B mirror)
- `configs/` — `faithful.yaml` (the paper, Table 1 exactly, cloud-sized),
  `local.yaml` (reduced, for an 8 GB card), `smoke.yaml` (toy),
  `REPRODUCTION.md` (why each choice was made) and `COVERAGE.md` (proof that
  nothing in the paper was missed — read this one before a training run)
- `scripts/` — thin CLI entry points that call into `src/`
- `outputs/` — generated data (gitignored, never commit)

**The granule-pair and waveform-pair data phases moved to
`/home/chris/thesis/mdct-granule-pairs` on 2026-10-21** (the LAME dumper,
`oracle.py`, `dataset.py`, `waveform.py` and their scripts). Nothing there was
defective; the comparison against Apollo simply needs Apollo's corpus. That
folder has its own README and git history, and the last commit here containing
it is tagged `mdct-granule-phase`.

## Workflow rules

- Run `pytest tests/ -q` before considering any change to `src/` complete.
- IMPORTANT: **update documentation in the same turn as the code, never after.**
  Docs here quote measured numbers and paper claims; a stale figure is a false
  claim, not untidiness. `tests/test_docs.py` asserts config-derived figures in
  `configs/REPRODUCTION.md` against the code, so drift fails the suite — but it
  cannot catch prose. After changing a quoted value, `grep -rn` the old value
  across `*.md` before moving on. See `.claude/rules/docs.md`.
- IMPORTANT: this repo reproduces a published model. **Every addition to the
  model, losses, STFT or configs gets a faithfulness check in the same turn**:
  locate the paper's words for it, compare Table 1 column by column (including
  the non-linearity column), assert it in a test, and update
  `configs/REPRODUCTION.md` §1.2 plus the changelog. See
  `.claude/rules/faithfulness.md` — it lists four real bugs this check caught
  that running the code would not have.
- IMPORTANT: `configs/REPRODUCTION.md` marks every claim **[P]** paper /
  **[I]** inferred / **[A]** assumption / **[D]** deviation. Never promote an
  inference to a fact silently — change the marker and add a changelog row. The
  record of having been wrong is what makes the reproduction auditable.
- IMPORTANT: **validate a metric before trusting a model's number on it.** Score
  the trivial pass-through on every metric, in the same run, and print it first;
  a baseline that shares a bug's blind spot proves nothing. Two days were lost
  to an `istft` boundary that made every waveform metric wrong by up to 46 dB,
  and a day to two metrics that saturate. See `.claude/rules/experiments.md` and
  REPRODUCTION.md §4c, §5.
- Prefer capping corpus work with `--max-steps` while iterating: on-the-fly MP3
  costs ~0.8 s per item, so a batch of 12 needs ~9.5 CPU-seconds and the **data
  pipeline, not the GPU, can set the floor** (§4f).

## Code style

- Type-hint public function signatures. Skip hints on obvious local variables.
- Docstrings only where behavior isn't obvious from the name and signature —
  but do document *why* where MP3 or LAME behaviour is surprising.
- Prefer plain functions over classes; use a class only to hold state across calls.
- Use `@dataclass` for config objects, not dicts with string keys.
- No bare `except:` — catch specific exceptions or let them propagate.
- Log with `logging`, not `print`, inside `src/`. Scripts may `print` their report.

## Avoid bloat

- Don't add an abstraction (base class, interface, config flag) until it's used in
  2+ places. Duplicate a small function once rather than generalizing on first use.
- Don't add a config option "for flexibility" — add it when an experiment needs it.
- Delete dead code instead of commenting it out.
- Prefer editing an existing file over creating a new one for a small addition.
