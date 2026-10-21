# Project

Dataset preparation for MP3 compression reversal: building (clean, degraded)
training pairs so a later phase can train a model to undo MP3 quantization. The
research plan of record is `mp3_dataset_plan.md`; dataset provenance and the
exact MP3 settings are in `DATASET.md`.

The repo now covers **two phases**. The dataset is built and validated, and on
top of it sits a reproduction of Lattner & Nistal, *Stochastic Restoration of
Heavily Compressed Musical Audio using GANs* (Electronics 2021, 10, 1349) —
model, losses and training loop, in PyTorch. `configs/REPRODUCTION.md` is the
living record of every ambiguity, assumption and deviation and is **required
reading before changing anything under `src/mp3ml/{models,losses,stft,train,
config}.py` or `configs/`**.

**The authors answered our open questions by email on 2026-09-29** — the critic
conditioning, the grouped convolutions, θ, and all four training coefficients
are now **[P]**, recorded verbatim in `configs/REPRODUCTION.md` §6.2. Several of
our assumptions were wrong; the critic was rebuilt as a grouped-convolution
stack with an 8-channel input.

**Training is stable as of 2026-10-04.** A long-standing divergence was traced
to a missing **equalized learning rate** — He's constant was baked into the
weights rather than applied at runtime, leaving per-layer weight scales spanning
35x (§4.10). All four survival cells now run 250 steps with the gradient penalty
at ~1, including the authors' original `recreate_optimizers: true`. **No full
training run has been done yet.** Beware short measurement windows: five
"stable" conclusions in this repo's history were artefacts of 25-40 step runs.

The central artifact is **granule pairs**: the encoder's pre-quantization
`xr[576]` (clean target) paired with the decoder's dequantized `xr_hat[576]`
(degraded input), read directly out of LAME's analysis path by
`native/lame_dump/mp3dump.c`. **Waveform pairs** (original vs MP3-decoded PCM)
are kept separately as evaluation ground truth, because `xr` is the transform of
LAME's pre-processed input rather than of the pristine original. Read
`native/lame_dump/README.md` before touching anything in `native/` or
`src/mp3ml/oracle.py` — it documents three traps that silently corrupt the data.

# Setup & commands

- Install: `python3 -m venv .venv && ./.venv/bin/pip install -e ".[dev]"`
- Set up LAME once: `bash native/lame_dump/setup_lame.sh` (fetches upstream
  lame-3.100, verifies its MD5, builds into `third_party/`, gitignored)
- Build the dumper: `bash native/lame_dump/build.sh`
- Run tests: `pytest tests/ -q`
- Lint + format: `ruff check . --fix && ruff format .`
- Validate the oracle: `python scripts/validate_oracle.py [INPUT.wav]`
- Audit waveform pairs: `python scripts/verify_waveform_dataset.py [--repair]`
- Dump one file: `python scripts/dump_granules.py INPUT.wav --bitrate 320`
- Build granule pairs: `python scripts/build_dataset.py --tracks 10` (pilot)
- Build waveform pairs: `python scripts/build_waveform_dataset.py --stems --workers 12`
- Check a config's size: `python scripts/check_config.py configs/local.yaml --measure`
  (`--measure` runs a real step; the analytic estimate under-reports by ~5x)
- Smoke-test the training loop: `python scripts/train.py configs/smoke.yaml --max-steps 4`
- Train: `python scripts/train.py configs/local.yaml --bitrate 16 [--resume]`
  (**pass `--bitrate`** — the paper trains one model per rate, REPRODUCTION.md §1.3;
  without it the run mixes all three, which is a different experiment)
- Live tracking: add `--tracker wandb`. The key is read from a **gitignored
  `.env`** at the repo root (`WANDB_API_KEY=...`, from <https://wandb.ai/authorize>);
  a shell export overrides it. Metrics always land in
  `outputs/<run>/metrics.jsonl` regardless, and that JSONL is the durable record
- Experiment results go in **`outputs/experiments/`** — gitignored but durable.
  Do not write them to a scratch directory; two sets have already been lost to
  /tmp clears
- Diagnose the section 4.6 divergence: `python scripts/diagnose_divergence.py`
  (the open blocker: training diverges with `recreate_optimizers: true`)
- Evaluate a checkpoint: `python scripts/evaluate.py CKPT --split test`
  (uses the paper's best-of-20 protocol; `--samples 1` is NOT comparable to Table 2)
- Size the hardware before a long run: `python scripts/bench_cluster.py configs/apollo64.yaml --workers 16`
  (step time, peak memory, and whether the GPU or the data pipeline is binding —
  run it on the target card, never extrapolate from a development GPU)
- Train on **Apollo's corpus** for the Apollo-vs-SRGAN comparison:
  `python scripts/train.py configs/apollo64.yaml --bitrate 64 --tracker wandb --workers 16`
  Needs the `apollo` extra (`pip install -e ".[apollo,dev]"`) for h5py.

External tools assumed on PATH: `ffmpeg`, `lame`. There is no `mpg123` on this
machine — use `ffmpeg` as the reference decoder.

# Repo layout

- `src/mp3ml/` — the library: `audio.py` (loading/downmix), `oracle.py` (the
  dumper wrapper and `RECORD_DTYPE`), `encode.py` (reference encode/decode),
  `dataset.py` (granule-pair layout, memmap loading, `TrackReader` chunk access),
  `waveform.py` (waveform pairs for the GAN reproduction, with lag alignment),
  `config.py` (experiment config + architecture derivation), `stft.py` (the
  paper's signed-sqrt complex STFT), `data.py` (training Dataset),
  `models.py` (generator + critic, Table 1), `losses.py` (eq. 1, eq. 3,
  gradient and drift penalties), `train.py` (the training loop),
  `metrics.py` (eq. 5-7, plus SI-SDR / multi-resolution LSD / Frechet
  distance), `evaluate.py` (the §4.1 best-of-20 harness), `peaq.py` (ODG/DI via
  the external `peaqb` binary), `fad.py` (VGGish embeddings for FAD),
  `tracking.py` (JSONL metric log, optional W&B mirror)
- `configs/` — `faithful.yaml` (the paper, cloud-sized), `local.yaml` (reduced,
  for an 8 GB card), `smoke.yaml` (toy, exercises the loop in seconds),
  `REPRODUCTION.md` (why each choice was made) and `COVERAGE.md` (proof that
  nothing in the paper was missed — read this one before a training run)
- `native/lame_dump/` — the C dumper. LGPL-2.1, links LAME's internal headers.
  Never *commit* LAME source; `setup_lame.sh` fetches it into gitignored
  `third_party/`.
- `scripts/` — thin CLI entry points that call into `src/`
- `tests/` — fast unit tests: synthetic audio only, no corpus, no network
- `outputs/` — generated data (gitignored, never commit)

# Workflow rules

- IMPORTANT: after changing anything in `native/lame_dump/` or `src/mp3ml/oracle.py`,
  run `python scripts/validate_oracle.py` before calling the change done. Its five
  checks are the only thing standing between a subtle misalignment and a corpus of
  silently wrong training pairs.
- `RECORD_DTYPE` in `src/mp3ml/oracle.py` must match `record_t` in `mp3dump.c`
  field for field. The C program writes `record_bytes` into its JSON and
  `dump_granules` refuses to load a mismatch — keep that check working.
- Run `pytest tests/ -q` before considering any change to `src/` complete.
- IMPORTANT: **update documentation in the same turn as the code, never after.**
  Docs here quote measured numbers and paper claims; a stale figure is a false
  claim, not untidiness. `tests/test_docs.py` asserts config-derived figures in
  `configs/REPRODUCTION.md` against the code, so drift fails the suite — but it
  cannot catch prose. After changing a quoted value, `grep -rn` the old value
  across `*.md` before moving on. See `.claude/rules/docs.md` for the
  code-to-doc mapping.
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
- IMPORTANT: when an experiment reveals a non-obvious fact about LAME's behaviour
  — a hidden preprocessing step, a silent mode switch, a field that doesn't hold
  what its type says — record it in `native/lame_dump/README.md` with the
  evidence, in the same turn you find it. This is the trigger `.claude/rules/docs.md`
  cannot catch, because no file edit fires it. Findings that live only in the
  conversation are lost at the end of the session.
- IMPORTANT: **64 kbps is the only bitrate usable for the Apollo comparison.**
  Apollo generates 24/32/48/64/96/128 and the paper uses 16/32/64, but in mono
  LAME resamples below 64 (24 -> 16 kHz, 32 -> 22.05 kHz, 48 -> 32 kHz), which
  changes the degradation and doubles the priming delay. 32 kbps is in both sets
  and still unusable. `mp3ml.apollo.degrade` raises on the rest; see
  REPRODUCTION.md §4f.
- IMPORTANT: the safe CBR floor depends on channel count — **64 kbps mono, but
  112 kbps stereo**. Below it LAME silently resamples (96k stereo -> 32 kHz, 64k
  stereo -> 24 kHz) and can switch to MPEG-2 LSF, changing what `xr` means.
  `dump_granules` raises on this; don't pass `allow_resample=True` to silence it
  without recording `out_samplerate` alongside the data.
- IMPORTANT: after building waveform pairs, run
  `python scripts/verify_waveform_dataset.py`. The lag is a property of the
  decoder's resampler, so it must be identical across every track at a bitrate —
  an outlier means the cross-correlation locked onto the wrong peak, which is what
  a silent lead-in causes. `--repair` rebuilds the offenders.
- IMPORTANT: waveform pairs must be **sample-aligned before storage**. LAME's
  gapless tag records MP3's inherent 1105-sample delay (576 enc + 529 dec), but
  LAME writes **no tag at all** below 32 kbit/s (MPEG-2/2.5), so it survives and
  scales on resampling: 2210 at 32k, 3046 at 16k. A naive pairing there has
  relative error > 1.0 — worse than pairing against silence — and fails silently.
  `mp3ml.waveform.make_pair` measures the lag per file and raises if it fails.
- Store coefficients as **float32, never float16** — fp16 corrupts 32% of the
  `xr - xr_hat` residual at 320 kbps, which is exactly what the model learns.
- Split datasets **by track, never by chunk**: chunks from one song in both train
  and val is leakage (see `.claude/rules/data.md`).
- Prefer capping corpus work with `--seconds` while iterating: `xr` is one float
  per input sample **per channel**, so a clean+degraded float32 pair costs
  ~1.27 GB per audio-hour per bitrate in mono, ~2.54 GB in stereo. The block-type
  schedule differs per bitrate, so clean `xr` cannot be shared across bitrates.

# Working with LAME

- Build against the **pinned upstream tree in `third_party/`**, never against a
  working checkout. `/home/chris/thesis/lame-3.100` is Chris's experiment tree and
  is locally modified: `newmdct.c` carries a `printf` and a per-band `fopen`/
  `fprintf` dump appending to `data/sample_data/mdct_dump.txt`, which reached
  337 MB in one afternoon of validation runs (~35 GB for a full 94-track,
  4-bitrate corpus build). A same-version diff against pristine 3.100 confirmed
  those two hunks are the *only* source differences and the MDCT maths is
  untouched — swapping to the pinned tree reproduced every validation number
  bit-identically. `/home/chris/thesis/libmp3lame` is **LAME 3.99.5**, a
  different release; don't use it as a reference.
- Don't assume a `plotting_data` field holds what its C type says. `pinfo->xr`
  is declared `double[576]` but is filled with 576 `float`s; see the dumper README.

# Code style

- Type-hint public function signatures. Skip hints on obvious local variables.
- Docstrings only where behavior isn't obvious from the name and signature —
  but do document *why* where LAME's behaviour is surprising.
- Prefer plain functions over classes; use a class only to hold state across calls.
- Use `@dataclass` for config objects, not dicts with string keys.
- No bare `except:` — catch specific exceptions or let them propagate.
- Log with `logging`, not `print`, inside `src/`. Scripts may `print` their report.

# Avoid bloat

- Don't add an abstraction (base class, interface, config flag) until it's used in
  2+ places. Duplicate a small function once rather than generalizing on first use.
- Don't add a config option "for flexibility" — add it when an experiment needs it.
- Delete dead code instead of commenting it out.
- Prefer editing an existing file over creating a new one for a small addition.
