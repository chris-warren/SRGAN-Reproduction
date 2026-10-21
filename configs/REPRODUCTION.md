# GAN reproduction: what the paper left open, and how it was resolved

Reproduction of:

> Lattner, S.; Nistal, J. *Stochastic Restoration of Heavily Compressed Musical
> Audio Using Generative Adversarial Networks.* **Electronics 2021**, 10, 1349.

**This file records only what was not obvious from the paper.** Anything Table 1
or §3 states plainly is implemented and asserted in `tests/`, not restated here.
What is here is the gaps: what the paper left unspecified, where the answer
eventually came from, and what we chose to do differently on purpose.

* Completeness — proof that nothing in the paper was *missed* — is
  `COVERAGE.md`, which maps all 89 normative statements onto code or a recorded
  decision.
* Dataset-side provenance is §4f, and the Apollo corpus's own README
  beside the data. The earlier granule-pair and waveform-pair phases, with
  their `DATASET.md`, moved to `/home/chris/thesis/mdct-granule-pairs` on
  2026-10-21 (tag `mdct-granule-phase`).
* There is **no official implementation**. The Data Availability Statement
  covers audio examples only, and the arXiv preprint is textually identical to
  the journal version.

The paper's own training setup (§3.2) needed no interpretation and is
implemented as stated: **40,000** iterations, batch size **12**, Adam at 1e-3,
gradient penalty, a drift penalty on D's output for real data, He initialisation.
Table 1's layer-by-layer architecture likewise — every row is asserted against
the config in `tests/test_config.py` and `tests/test_models.py`.

| mark | meaning |
|---|---|
| **[P]** | stated in the paper, or answered directly by the authors |
| **[I]** | inferred from the authors' *other* published code |
| **[A]** | our assumption — no evidence either way |
| **[D]** | deliberate deviation |

---

## 1. [P] Gaps that are now closed

Everything here was unknown or wrong at some point. The right-hand column is the
part worth keeping: **where the answer actually came from.**

| what was missing | resolution | source |
|---|---|---|
| How the MP3 conditioning reaches the critic | channel-wise concat, **8 channels**: MP3 complex (2), candidate complex (2), each magnitude widened to 2 | authors' email, §1.1 |
| What "two convolutional groups" means | PyTorch **`groups=2`** throughout the critic, joined only at the output layer | authors' email, §1.1 |
| θ in eq. 3 | **two** weights, not one: **1e-4** frequency (p=1.3), **1e-5** rhythm (p=1.6) | authors' email, §1.1 |
| Gradient-penalty weight | **5** | authors' email, §1.1 |
| Drift-penalty weight | **0.5** | authors' email, §1.1 |
| Adam betas | **(0, 0.99)** | authors' email; independently confirmed in their 2020 code |
| Critic:generator update ratio | **1:1** | authors' email; we had assumed 5 |
| Optimiser state across updates | **recreated each update** in the original code | authors' email — see §3.3 |
| Gradient-penalty reduction | **summed** over the batch, not averaged | authors' email; their code confirms |
| Drift-penalty reduction | summed over the batch | their code; and measured to suppress the drift ~4x — §1.2a |
| Why training diverged for weeks | **equalized learning rate** was missing | their code defaults to it — §1.2 |

### 1.1 [P] The authors' reply, 2026-09-29

Stefan Lattner answered by email. **This is the single most valuable source in
the project** and settled seven of the rows above at once.

> "I checked the original implementation and training settings—some details
> were unfortunately not made fully explicit in the paper.
>
> The critic receives the MP3 by **direct channel-wise concatenation**. Its
> effective input is **eight channels**: the two complex channels of the MP3,
> the two complex channels of the original/restored candidate, then two-channel
> versions of the MP3 magnitude and candidate magnitude. The two convolutional
> groups are indeed **grouped convolutions** throughout the critic: one group
> processes the complex MP3/candidate pair and the other processes the
> magnitude MP3/candidate pair. They are joined only in the final output layer.
> So the table's two-channel input description is incomplete/ambiguous rather
> than indicating that the MP3 enters elsewhere.
>
> For the profile regularizer, the final stochastic 32 kbit/s setup used
> **separate weights rather than a single shared θ: 1e-5 for the rhythm/time
> profile and 1e-4 for the frequency profile**. The exponents were 1.6 and 1.3,
> respectively. So θ = 1 would indeed be far too large.
>
> The settings I recovered for the stochastic runs were:
>
> - Gradient-penalty weight: **5**
> - Drift penalty weight: **0.5**
> - Adam betas: **(0, 0.99)**
> - Critic-to-generator update ratio: **1:1**
>
> One implementation detail worth noting: **the optimizer objects were
> recreated for each update** in the original code, so their Adam moment
> estimates were not retained across batches. Also, **the gradient penalty was
> summed over the batch rather than averaged**. If your goal is a close
> behavioral reproduction, these details may matter; if your goal is a robust
> modern baseline, retaining optimizer state and averaging the penalty would be
> reasonable improvements."

One caveat he gave: these are the settings of the **stochastic 32 kbit/s** run
specifically, recovered from code years later. They may differ for 16 and
64 kbit/s and for the deterministic models.

### 1.2 [I] Equalized learning rate — the cause of the divergence

Not in the paper, and not in the email. Found by reading
[`SonyCSLParis/Comparing-Representations-for-Audio-Synthesis-using-GANs`](https://github.com/SonyCSLParis/Comparing-Representations-for-Audio-Synthesis-using-GANs)
— **reference [67] of the paper**, by the same authors, cited for the complex-STFT
representation we use. Its `ProgressiveGAN` defaults to `equalizedlR=True`:

```python
self.module.weight.data.normal_(0, 1)  # weights start at N(0, 1)
self.weight = sqrt(2 / fan_in)  # He's constant...


def forward(self, x):
    return self.module(x) * self.weight  # ...applied to the OUTPUT
```

The paper says only "He's initialization is performed for all the layers"
(§3.2), which is literally true of both forms. Because this is an
initialisation detail rather than a layer, Table 1's silence does not rule it
out — see §3.6.

**Why it mattered so much,** measured on `local.yaml`:

| | He baked into weights | equalized |
|---|---|---|
| weight-scale spread across layers | **35.2x** | 1.1x |
| worst layer's step as a fraction of its own scale | **10.2%** | 0.10% |

A single step at lr 1e-3 moved Conv4 by 10.2% of its own weight scale while
moving Conv1 by 0.3%. Conv4 is the frequency-aggregation layer the paper's whole
§3.1.3 idea rests on, and it has by far the largest fan-in.

**Result.** Four survival cells, 250-step horizon, which previously died at
steps 27, 16, 4 and 3:

| `recreate_optimizers` | seed | result |
|---|---|---|
| false | 0 | survived, median gradient penalty **0.93** |
| false | 1 | survived, median **1.09** |
| true | 0 | survived, median **1.17** |
| true | 1 | survived, median **1.30** |

All sit at ~1, where a healthy WGAN-GP's penalty belongs. This includes
`recreate_optimizers: true`, the authors' original behaviour, so there is no
longer a conflict between a faithful configuration and one that trains.

Treat as strong evidence, not proof: 250 steps is not 40,000. **Cost:** a
forward hook applies the constant, so autograd retains the pre-scale activation
— peak memory rose ~5.7 → 7.3 GB and the step rate fell ~4x.

### 1.2a [I] Drift-penalty reduction — summed, not averaged

The paper names the drift term (§3.2: "a loss term that penalizes the magnitudes
of the output of D for real input data, **preventing the loss from drifting**")
and the authors gave us its weight (0.5), but not how it is reduced over the
batch. Their code sums it, exactly as it does the gradient penalty:

```python
lossEpsilon = (predRealD[:, -1] ** 2).sum() * self.config.epsilonD
```

We averaged until 2026-10-04, making ours `batch_size` times weaker — 4x at
`local.yaml`'s batch 4, 12x at the paper's 12.

**This showed up as a symptom first.** The 2000-step run's drift climbed from
3.88 to 269 (70x), with the Wasserstein and adversarial terms growing alongside
it — which is precisely the drifting the term exists to prevent. Measured over
400 steps, one seed:

| `drift_reduction` | drift @50 | drift @end | max | gp median |
|---|---|---|---|---|
| mean | 14.12 | 38.88 | 230.8 | 0.843 |
| **sum** | **3.66** | **20.90** | **174.2** | 2.120 |

At step 50 the drift is **3.9x lower**, almost exactly the 4x the batch size
predicts, and stays about half through 400 steps. `sum` is now the default.

Two honest caveats: one seed over 400 steps, which this project has learned not
to over-read; and `sum` only *slows* the growth (3.66 → 20.9) rather than
stopping it, so the drift term may still be weaker than intended. At the paper's
batch 12 it would be 3x stronger again.

### 1.3 [P] Stated in the paper, but easy to get wrong

These are in the text. They are listed because each was implemented incorrectly
first, and nothing failed when it was.

| trap | the paper's words | what goes wrong |
|---|---|---|
| Self-gating is **not** `nn.GLU` | eq. 2 uses **PReLU**; §3.1.2: linear units "resulted in unstable training" | shapes match either way, so it trains — just not as published |
| **Two different roots** of the spectrum | eq. 3 wants the "Hadamard root of the **power** spectrum" (`sqrt(P)`); §3.1.1 wants "the square-root of the **magnitude** spectrum" (`P^(1/4)`) | using one for both silently changes `L_profile`'s scale and the critic's second view |
| `z` is a **vector**, not a map | "simply repeated in the two convolutional dimensions"; §3.2.1 says "noise **vectors**" | a per-cell sample passes every shape check and inflates `‖z_i − z_j‖` ~100x |
| Skip connections in **D only** | "skip connections for D in the convolutional layers Conv6 - Conv14" | G is unpadded in time, so nothing is shape-compatible to add there |
| Target is **centre**-cropped | G is unpadded, so `output[k]` is centred on `input[k + sum(dilations)]` | a front crop misaligns target and prediction by 62 frames with no error |
| Evaluation is **best-of-20, per metric** | §4.1: "we sample 20 times ... for each objective metric, we take the best value of that set" | a single-draw score is not comparable to Table 2 |
| **One model per bitrate** | §4.1: "for each compression rate, we train a stochastic generator and a deterministic generator" | a mixed-bitrate model is a different experiment |
| "Deterministic" = **no noise input** | §4.1: "a deterministic generator (without noise input)" | not a stochastic model evaluated at z = 0 |

Three independent reimplementations (ours and both unofficial ones) made the
`z`-as-spatial-map misreading. That sentence is genuinely ambiguous.

---

## 2. [A] Still assumed — no answer exists

Nothing below is settled by the paper, the authors, or their code.

| parameter | our value | note |
|---|---|---|
| critic's condition **time window** | `y` centre-cropped to G's 212 frames | the authors confirmed *channel-wise* concatenation but said nothing about the time axis |
| noise draws seen by D | one | §3.2.1 ties the two draws to G's update only |
| batches per step | critic and generator draw **separate** batches | as in Gulrajani's reference implementation |
| STFT **window function** | Hann | §3.3.1 gives "window size 2048" but never names the window |
| `SQRT_FLOOR` | 1e-24 | numerical floor under `sqrt`; was 1e-12, which destroyed the signal |
| `LOG_FLOOR` | 1e-10 | floor under eq. 5's log; the paper states none |
| frequency axis | 1024 bins (Nyquist dropped) | `n_fft = 2048` gives 1025; Table 1 uses 1024 |
| segment length | **336 frames**, not 4.000 s | §3.3 says "4-second-long", §3.1.1 says "336 frames"; at hop 512 those differ (4 s = 341). The architecture's time chain is stated in frames |

---

## 3. [D] Deliberate deviations

### 3.1 [D] `local.yaml` — development only

The faithful model does not fit an 8 GB card.

| config | params | Adam state (fp32) | verdict on 8 GB |
|---|---|---|---|
| `faithful.yaml` | 404.3 M | 6.47 GB | does not fit |
| `local.yaml` | 50.8 M | 0.81 GB | fits |

`local.yaml` deviates **only in frequency resolution and width**:

| | faithful | local | reason |
|---|---|---|---|
| `n_bins` | 1024 | 512 | Conv4 scales with bins |
| `conv4_out` | 4096 | 1024 | the dominant layer |
| `latent_ch` | 128 | 32 | takes the whole `conv4_out` reduction (below) |
| `latent_freq` | 32 | **32** | *unchanged* — it is load-bearing (below) |
| `stack_ch` | 256 | 128 | activation memory |
| `noise_ch` | 64 | 32 | follows stack width |
| `batch_size` | 12 | 4 | measured OOM above 6 on 8 GB |
| `segment_frames` | 336 | **336** | *unchanged* |
| dilations | (1,2,4,8,16)x2 | **same** | *unchanged* |
| output frames | 212 | **212** | *unchanged* |

The time structure and dilation schedule are the paper's, so the receptive field
and the 336 → 212 crop behave identically. **Results from it are not comparable
to the paper.**

#### `latent_freq` cannot be reduced — found 2026-10-07

`Reshape1` splits `conv4_out` into `latent_ch × latent_freq`, so a reduced
config can take the reduction out of either factor and every shape, parameter
count and test still passes. Taking it out of `latent_freq` is **not**
equivalent. The stack's convolutions are 3-tap in frequency and padded by their
dilation `d`, so a layer at dilation `d ≥ latent_freq` has both outer taps on
padding for **every** row: it is a 1-tap conv in frequency, and the dilation is
wasted.

`local.yaml` carried `latent_freq = 16` against the paper's `(1,2,4,8,16)×2`
schedule, so the two dilation-16 layers were degenerate — 16 of 48 taps on real
data (33%) where faithful has 64 of 96 (67%). `smoke.yaml` was worse at
`latent_freq = 8`, degenerating four layers, which means every shape test in the
suite was asserting against a structurally different model from the one
`faithful.yaml` builds.

Both are fixed by moving the reduction to `latent_ch` at the same `conv4_out`
(local 51.3 M → 50.8 M params, smoke 0.27 M), and `ModelConfig.__post_init__`
now raises when `max(dilations) ≥ latent_freq` so no future config can
reintroduce it. Asserted by `test_latent_freq_must_exceed_the_widest_dilation`.

This was found while diagnosing the generator's fidelity failure (§4b) and is a
real defect in the reduced configs, but it is **not** the cause of that failure:
`faithful.yaml` always had `latent_freq = 32`, and the §4b measurements were
made on `local.yaml`, so they need re-running before anything is concluded from
them.

#### `batch_size` silently rescales two of the authors' four coefficients

Found 2026-10-13 (`audit_local_deviations.py`). The Wasserstein term is a batch
**mean** (`losses.py:31`); the gradient and drift penalties are batch **sums**,
which is the authors' own convention (`gp_reduction: sum`, §3.3). Since
`X.sum() == batch * X.mean()`, the penalty-to-Wasserstein ratio is
**proportional to the batch size**, so `local.yaml`'s reduction from the paper's
batch 12 to batch 4 divides both penalties by three:

| | batch | `gp_weight` | effective | `drift_weight` | effective |
|---|---|---|---|---|---|
| `faithful.yaml` | 12 | 5.0 | **60.0** | 0.5 | **6.0** |
| `local.yaml` | 4 | 5.0 | **20.0** | 0.5 | **2.0** |

Both of the authors' [P] values are therefore running **3x weaker** in every
local run, including the one that diverged at step 4609 — which used batch 4
with `gp_weight 5.0` and `drift_weight 0.5`. Restoring their balance at batch 4
needs `gp_weight: 15.0` and `drift_weight: 1.5`.

This is arithmetic, not a hypothesis. What is **not** established is that it
caused the divergence. It points the right way — a 3x weaker drift penalty
permits exactly the unchecked growth of critic scores that §4a logged (drift
11 → 1024, i.e. |score| ≈ 16, while the gradient penalty stayed near 1) — but
the mechanism is unproven. A literature check also found **no** support for the
related "small batches make the penalty estimate noisy" story, and one thesis
reports WGAN-GP getting *less* stable at *larger* batch sizes, so that argument
is dropped.

A second, smaller consequence of the width reductions: with the equalized
learning rate (§1.2) the runtime scale is `sqrt(2/fan_in)`, so narrower layers
are **more** sensitive per optimizer step of fixed size `lr` — 1.41x for the
3x3 stack convolutions and 2.00x for the latent→stack 1x1. Also pointing the
destabilising way, also unproven.

Ruled out: the critic/generator capacity ratio is preserved almost exactly
(0.253 faithful vs 0.254 local), so the reductions did not unbalance the two
networks.

**All three vanish at `faithful.yaml`.** They are consequences of the reduction,
not bugs in the code, so the fix is to stop reducing rather than to tune the
coefficients — tuning them would compensate for a deviation we intend to delete,
and would then be wrong at full size.

### 3.2 [D] Corpus, split and segment sampling

| | paper | ours | why |
|---|---|---|---|
| corpus | ~64 h of US chart #1 songs | musdb18hq, 47 h | the chart dataset is proprietary |
| split | 80 / 10 / 10 | 59 / 6 / 35 | musdb18hq's **canonical** test split is 50 of 150 tracks; keeping it makes our numbers comparable to other musdb work |
| segments | 4 s, 50% overlap | random offsets per access | a superset — every 50%-overlap window is reachable. **Evaluation uses fixed crops**, or no result would be reproducible |
| bitrate | one model per rate | `--bitrate` selects one; configs list all three | the mixed-bitrate model is a useful variant but is not what Table 2 measures |

### 3.3 [D] The two fidelity switches

The authors volunteered two details of the original implementation and framed
them explicitly as a choice. Both are config fields, so one codebase produces
either:

| field | original behaviour | modern alternative |
|---|---|---|
| `recreate_optimizers` | Adam rebuilt every update | state carried across steps |
| `gp_reduction` | penalty **summed** over the batch | averaged |

`recreate_optimizers: true` makes Adam behave as **sign-SGD with step size
exactly `lr`** — a fresh step from zeroed moments gives `m_hat = g`, `v_hat = g²`.
Measured: a parameter moves exactly 1e-3 for gradients from 1e-3 to 1e6.

At batch `B` the summed penalty is `B` times the averaged one, so `gp_weight: 5`
with `"sum"` at batch 12 behaves like an effective weight of 60. **Runs in the
two modes are not comparable**; `Trainer` writes the mode into
`outputs/<run>/notes.md` for that reason.

### 3.4 [D] Mixed precision and throughput

Neither is mentioned in the paper, which trained on 2x Titan RTX. Measured on an
RTX 2070 Super, `local.yaml`, batch 4:

| `amp` | `grad_checkpoint` | s/step | peak GB |
|---|---|---|---|
| true | true | 19.59 | 7.79 |
| **false** | **true** | **12.35** | **5.71** |
| false | false | 14.12 | 5.81 |

**AMP is a 1.6x pessimisation here** — we force the gradient penalty and the
generator loss to fp32 for correctness, so autocast pays casting costs without
buying fp16 throughput, and Conv4's 38 input channels are not a multiple of 8 so
tensor cores likely never engage. `local.yaml` therefore sets `amp: false`;
`faithful.yaml` leaves it on with a note to **re-benchmark on the cloud GPU**,
where Ampere/Hopper behave differently.

The gradient penalty stays fp32 regardless: it is a double-backward, and a
second derivative on fp16 activations is where AMP most often fails quietly.

### 3.5 [D] Checkpoint size

A `local.yaml` checkpoint is **992 MB** (G + D + both Adam states);
`faithful.yaml` will be ~8x larger. At `--checkpoint-every 1000` over 40k steps
that is ~40 GB per run, ~240 GB for the six runs Table 2 needs, and ~1.9 TB at
faithful scale. Not addressed. Cheapest fixes: keep only the last N numbered
checkpoints, or save optimiser state only for the newest.

### 3.6 [D] Which standard stabilisers are admissible

**The rule:** Table 1 enumerates *every layer* with its non-linearity column, so
anything that would appear as a layer is ruled out by its absence. Anything that
is an initialisation or scaling detail does not appear in a layer table at all.

| stabiliser | status | why |
|---|---|---|
| Equalized learning rate | **adopted** | not a layer; the authors' own code defaults to it (§1.2) |
| LayerNorm in the critic | rejected | Gulrajani recommends it, but it *is* a layer and Table 1 shows none |
| Minibatch stddev, PixelNorm | rejected | layers; Table 1 shows none |
| Two-timescale update (TTUR) | rejected | §3.2 gives a single learning rate |
| Gradient clipping | rejected | mentioned nowhere; would interact with the gradient penalty |
| Spectral normalisation | rejected | an alternative to GP; the paper uses GP |

If more stabilisation is ever needed, **LayerNorm in the critic** is the next
candidate and must be labelled a **[D]**.

---

## 4. [A] Open before this is a reproduction

1. **Only a 2000-step run has been done** — 5% of the paper's 40,000, on
   `local.yaml`. It completed without diverging, which is the first end-to-end
   confirmation that §1.2's fix holds. Evaluated on the test split with the
   best-of-20 protocol:

   | metric | MP3 baseline | model | wanted |
   |---|---|---|---|
   | LSD | 42.84 | **22.69** | lower ✓ |
   | multi-resolution LSD | 43.47 | **21.65** | lower ✓ |
   | spectral MSE | 0.209 | 4.229 | lower ✗ |
   | SNR | 12.59 | −35.15 | higher ✗ |
   | SI-SDR | 12.52 | −49.64 | higher ✗ |

   > **The SNR and SI-SDR rows are invalid** (`istft` boundary, §4c), and the
   > paragraph that followed them — reasoning from "SI-SDR at −49.6 dB is
   > scale-invariant, so the output is uncorrelated with the target" to a
   > generator that makes envelopes rather than content — is **retracted**. The
   > whole-number cause was in the ruler. The LSD rows are spectrogram-domain and
   > stand; the eval harness must be re-run for the rest.
   >
   > Note what the MP3-baseline column did here. Its SNR of **12.59** against the
   > paper's 13.69 was recorded as a sanity check that *passed*, and used to
   > validate the pipeline end to end. It did pass, and it was measuring a real
   > thing — but it could not have caught this bug, because `clean` and `y` carry
   > identical edge artefacts that cancel in their difference. **A baseline that
   > shares the bug's blind spot cannot validate the metric.** The check that
   > would have caught it is a *perturbed target*, where the error is independent
   > of both signals.

   LSD halving is the right direction and roughly the right size (the paper
   gets 66% at 16 kbit/s; this is 47%), and is the expected signature of a model
   adding high-frequency content — §3.4.3 says so explicitly.

   **Do not compare the spectral absolutes to Table 2.** `local.yaml` computes
   them on `n_fft` 1024 / 512 bins where the paper uses 2048 / 1024.
2. **PEAQ binary not installed** — ODG and DI are two of Table 2's five columns
   and cannot currently be produced at all. Cheapest remaining fix.
3. **MOS listening test not run** — 15 participants. The paper's headline claim
   (stochastic ≈ deterministic in perceived quality) cannot be checked without it.
4. **Six models needed for Table 2** — 3 bitrates x {stochastic, deterministic}.
5. Everything in §2 is still an assumption.

---

## 4a. [A] The 2026-10-05 long run — what it diagnosed

10,000 steps requested, **diverged at step 4609** (~5.1 h) with all four fixes
active. The guard caught it; `diverged.pt` and four numbered checkpoints are in
`outputs/local_16k/`, raw numbers in `outputs/experiments/`.

**Equalized learning rate did not "fix" the divergence — it delayed it ~170x**,
from step 27 to step 4609. That is a real improvement and it is why 2000 steps
completed earlier, but the instability is still there.

The validation curve is the important part (MP3 baseline: LSD 42.8, SI-SDR 12.2,
SNR 12.5, MSE 0.25):

| step | LSD | SI-SDR | SNR | spectral MSE |
|---|---|---|---|---|
| 500 | 72.3 | −53.2 | 0.0 | 6.26 |
| 1500 | 42.7 | −54.1 | −0.7 | 6.10 |
| 2500 | 30.9 | −54.8 | −10.1 | 5.75 |
| 3500 | 27.1 | −55.8 | −21.1 | 5.31 |
| 4500 | **19.9** | **−53.9** | **−32.7** | 5.25 |

**LSD improves monotonically and crosses below the MP3 baseline around step
1500. SI-SDR never moves — it sits at −54 for the entire run. SNR gets steadily
worse.**

> **The SI-SDR and SNR columns above are invalid** — the `istft` boundary bug,
> §4c. And the flat −54 was the tell we missed: a metric that does not budge
> across 4,000 steps while every other metric moves is not reporting on the
> model at all. It was pinned by exploded edge samples the generator does not
> control. **A metric frozen to two significant figures while its neighbours
> move is a broken metric, not a stuck model.** The LSD and spectral-MSE columns
> are spectrogram-domain and stand.
>
> The conclusion drawn from it — *"the generator is learning to synthesise
> plausible-sounding audio, not to restore its input"* — rested on SI-SDR being
> flat and is **retracted**. The LSD improvement in this table is real, and it is
> the opposite kind of evidence: the model was getting better and we had no
> working instrument to see whether that reached the waveform.

The logged critic accuracy was **1.00 at every step** until the blow-up, when it
flipped to 0.00. The critic separated real from fake perfectly from step 1 —
the generator never once fooled it.

**Where that points.** The paper's fidelity mechanism is the conditioning:
"In addition to the true/fake data, D also receives the MP3 versions ... That
way, it is ensured that the information present in the MP3 data is faithfully
preserved in the output of G" (§3.1). There is deliberately **no reconstruction
loss**. If SI-SDR never improves, that mechanism is not exerting pressure in our
implementation — plausibly because a critic winning 100% of the time has no need
to consult the conditioning to tell real from fake, so nothing pushes G toward
fidelity.

> That last inference is **void**, since its premise ("SI-SDR never improves")
> was the broken metric. The critic-accuracy observation it rests on is not: 1.00
> at every step until the blow-up is computed from critic scores, not waveforms,
> and a critic that is never fooled remains worth explaining.

**Ruled out: the conditioning works.** Measured on `step004000.pt` by feeding
the critic a mismatched `y` (rolled by one within the batch):

| | score |
|---|---|
| D(real, correct y) | −24.3 |
| D(real, **wrong** y) | −102.8 |
| D(fake, correct y) | −108.7 |

Mismatching the condition costs **78.6**, against **84.4** for swapping real for
fake — a ratio of 0.93. The critic consults `y` almost as heavily as it consults
the candidate, so §3.1's fidelity mechanism is intact and G *is* receiving a
fidelity signal. The hypothesis that the conditioning was decorative is dead.

**Leading hypothesis: `recreate_optimizers` prevents fidelity being learned.**
A fresh Adam step is sign-SGD (§3.3): every parameter moves exactly `lr` in the
gradient's sign direction, **discarding magnitude entirely**. That is enough to
learn a coarse spectral envelope — which is what LSD measures, and LSD improved
monotonically — and not enough for the fine content correlation SI-SDR measures,
which never moved. The split between the two metrics is the signature.

Supporting evidence from the authors' own flagship codebase: it creates the
optimisers **once** and calls `.step()` each update. Recreating them per update
is not their standard practice, which fits the author raising it unprompted and
suggesting that retaining state would be "a reasonable improvement".

## 4b. [A] Diagnosis of the fidelity failure — 2026-10-06

> **The magnitude and phase figures in this section are retracted.** They were
> measured on `local.yaml` while it carried the §3.1 `latent_freq` defect, and
> with two metrics that §4c shows do not measure what they were read as. The
> conditioning and input-use verdicts stand. Read §4c for the corrected picture;
> this section is kept because the two retracted rows are the ones that set the
> investigation's direction for a day.

Four hypotheses, three eliminated by measurement. All numbers from
`outputs/local_16k/checkpoints/step004000.pt` on held-out val excerpts.

| hypothesis | test | verdict |
|---|---|---|
| the divergence is the root problem | gradient penalty stayed 1–3 until step 4000, so the Lipschitz constraint was binding; `W = 61` is legitimate for a 1-Lipschitz critic when ‖real − fake‖ is also ~61, which SNR falling to −33 confirms | **symptom, not cause** |
| the critic ignores its conditioning | mismatching `y` costs **78.6**; swapping real for fake costs **84.4** (ratio 0.93) | **ruled out** |
| the generator ignores its input | changing the input moves **92%** of the output, 4.3x what changing the noise moves | **ruled out** |
| magnitude is fine, only phase is broken | see below | **both are broken** |

**The measurement that reframes the problem:**

| | magnitude correlation with target | phase agreement |
|---|---|---|
| **G output** | **0.394** | **−0.0006** (random) |
| MP3 input | 0.985 | 0.128 |

**G's output correlates with the target less than its own input does.** Phase is
pure noise, and the magnitude is worse than a pass-through. Substituting the
MP3's phase into G's magnitude lifts SI-SDR only from −50 to −37, against the
MP3 baseline's **+13.3**.

So this is not "restoration that has not converged yet". The adversarial
objective is driving G *away* from its input, while D demonstrably checks that
input. The paper loses 4 dB of SNR to imperfect phase and says so explicitly
(§4.1: "the phase information is not faithfully regenerated"); we lose 48 dB and
the magnitude with it.

**Relevant architectural note.** §3.1.1 rejects the U-Net paradigm outright —
"the architecture proposed in this work is non-symmetrical ... Only the top parts
of the stacks differ" — and the skip connections it does specify are **D only**.
So G must learn a near-identity mapping through ~20 layers with no shortcut,
which is exactly the case U-Net skips exist to make easy. That is the paper's
design, not a deviation, but it raises the bar on how much the adversarial signal
has to accomplish unaided.

### 4c. [A] The fidelity failure was mostly our own metric — 2026-10-07

> **Every SI-SDR and SNR figure recorded in this file before 2026-10-07 is
> wrong, by up to 46 dB.** `istft` amplified the segment boundary by up to 1e8,
> and every waveform metric sums squared error over the whole signal, so a
> handful of samples dominated all of them. `lsd` and `spectral_mse` are
> computed in the spectrogram domain and are unaffected — which is why those two
> always looked more reasonable than the rest.

**The bug.** `istft` divides by the overlap-add window sum. With `center=False`
that sum falls to *exactly zero* at the segment boundary, and the code read
`out / norm.clamp(min=1e-8)` — a gain of up to 1e8 on the first and last
`n_fft - hop` samples. The docstring asserted the edges "degrade gracefully"
and were "outside G's output crop anyway". Both were false: the crop happens in
the **spectrogram** domain, so the 212-frame spec that reaches `istft` has its
own fresh, uncovered edges.

`istft` now returns zeros for samples without full window coverage. They are
unrecoverable — the frames that would complete them do not exist — so a best
effort there is worse than an honest gap, and `clean` and `restored` both pass
through the same path so both are zeroed identically.

**Measured, same perturbation of a real target, before and after**
(`check_metric_path.py`; "inner" trims one window from each end):

| | train MSE | complex SNR | SI-SDR **before** | SI-SDR **after** | inner |
|---|---|---|---|---|---|
| MP3 input | 0.0370 | 14.96 | +13.12 | +12.82 | +12.83 |
| target + 0.1·N | 0.0100 | 23.23 | **−10.03** | **+23.51** | +23.53 |
| target + 0.03·N | 0.0009 | 33.75 | **+4.73** | **+34.43** | +34.46 |
| target + 0.01·N | 0.0001 | 43.36 | **+12.63** | **+44.09** | +44.10 |

After the fix every row agrees with the Parseval-derived complex SNR. Before it,
they were out by up to 33 dB.

**Why it hid for two days.** The MP3 baseline is the one signal that was
*unaffected* (13.12 → 12.82): `clean` and `y` carry identical edge artefacts
which cancel in their difference. A model output has independent edge content,
so nothing cancels. The bug therefore left the baseline looking healthy and made
every model look catastrophic — the exact shape of a real fidelity failure.
With the fix, the untrained generator at step 1 scores **−2.97 dB**, not the
**−48.99** we recorded.

The existing round-trip tests in `test_stft.py` already trimmed
`slice(N_FFT, SEGMENT - N_FFT)` before comparing. The limitation was known and
encoded in the tests; it was never encoded in the code or applied to the metrics
path. `test_waveform_error_tracks_spectral_error` now asserts the invariant that
was violated — overlap-add is a least-squares pseudo-inverse, so waveform error
may come out *below* spectral error but can never exceed it.

#### What still stands, and what is retracted

Also note that everything in §4b was measured on `local.yaml` **before** the
§3.1 `latent_freq` defect was found, so two of the ten stack layers were
degenerate in frequency. Two independent defects were therefore in play.

| claim | status |
|---|---|
| critic uses its conditioning (§4b) | **stands** — ratio of score differences, no waveform metric involved |
| generator uses its input (§4b) | **stands** — same |
| "G's magnitude correlates 0.394, worse than its input" | **retracted** — degenerate config *and* a saturating metric |
| "phase is pure noise (−0.0006)" | **retracted** — unweighted cosine reads 0.135 for a +13 dB signal |
| "SNR falls to −33 / SI-SDR −50" | **retracted** — the `istft` boundary |
| the W-slide and divergence in `outputs/local_16k` | **unretracted but unexplained** — `wasserstein` and `gradient_penalty` are computed from critic scores, not waveforms, so the bug does not touch them. But that run also had the `latent_freq` defect, so it needs re-running |

Two further corrections fall out of the capacity runs.

**The 2026-10-06 run never beat the identity.** It reported `spectral_mse`
3.53 → 0.567 as "84% better", but the input it was handed scores **0.0623** —
nine times better than where the model finished. Reading a loss curve against
zero rather than against the baseline is what made a model still far worse than
its own input look like progress. The baselines are now printed first for
exactly this reason.

**Pooled magnitude correlation is not a usable metric, and §4b leaned on it.**
At step 1000 it reads 0.965, level with the pass-through's 0.964, while the
model is still 4.6 dB behind it on SI-SDR. Correlation over ~4·10⁵ bins spanning four
decades of dynamic range is dominated by the shared spectral envelope and
saturates near 0.96 for anything that is roughly audio-shaped. The same applies
to the phase cosine as §4b computed it: unweighted, it read **0.135** for the
MP3 input that scores +13 dB, because near-silent bins — where phase is
meaningless and unlearnable — outnumber everything else. Magnitude-weighted, the
same input reads **0.574**. §4b's "phase is pure noise (−0.0006)" was measuring
the noise floor.

#### The capacity test, on the fixed config and the fixed metric

`overfit_one_batch.py` fits ONE batch under pure MSE, no critic
(`2026-10-07-capacity-fixed-istft.txt`). The pass-through is the bar:

| | `mse_loss` | SI-SDR | `spectral_mse` | mag corr | phase cos |
|---|---|---|---|---|---|
| **pass-through (y vs clean)** | **0.0370** | **+12.82** | **0.0623** | **0.964** | **0.574** |
| G at step 1 | 0.2040 | −2.97 | 3.5311 | 0.095 | ~0 |
| G at step 500 | 0.0528 | +1.09 | 0.6113 | 0.914 | 0.596 |
| G at step 1000 | 0.0212 | +8.19 | 0.1473 | 0.965 | 0.727 |
| G at step 2000 | 0.0104 | **+13.30** | 0.0470 | 0.984 | 0.829 |
| G at step 3000 | 0.0044 | +20.73 | 0.0200 | 0.994 | 0.915 |
| G at step 4000 | 0.0022 | **+23.66** | 0.0189 | 0.997 | 0.956 |

**The generator can represent the mapping, and comfortably.** It passes the MP3
input's SI-SDR around step 2000 and reaches **+23.66 dB, 10.8 dB above the
pass-through**, by step 4000, with every other metric ahead too — magnitude
correlation 0.095 → 0.997, weighted phase cosine → 0.956. Two hypotheses die
here: "the architecture cannot represent it", and "the objective's domain is a
poor proxy for the waveform". In the fixed measurement the two domains track,
exactly as Parseval requires.

What this does **not** show: that the *adversarial* objective finds that
solution. This run has no critic. §4a's open question — a critic at 1.00
accuracy from step 1, and a Wasserstein distance sliding 0.23 → 48 over 3,000
steps — is computed from critic scores, so neither of today's fixes touches it.
It is the live issue.

**How the measurement bug was found.** Not by suspecting it — by an arithmetic
contradiction that would not resolve. At step 5400 the generator's training MSE
was **0.00094 against the MP3's 0.0370**, thirty-nine times lower, magnitude
correlation 0.999, weighted phase cosine 0.979 against 0.574. Every quantity the
loss can see said G was comfortably better than its own input; SI-SDR said 23 dB
worse. Parseval bounds how far apart those can be, so one of them had to be
wrong, and the cheap way to find out which was to **stop measuring the model and
measure the measurement**: perturb a real target by known amounts and check that
the domains track. They did not, by 33 dB. No model, no GPU, two minutes
(`check_metric_path.py`).

> Two earlier attempts to explain the same contradiction were both wrong, and
> both were plausible stories about the *model*. The first blamed eq. 4's
> squaring — inversion squares, so error at loud coefficients is amplified
> quadratically — and the error-by-decile table seemed to confirm it (98.6% of
> G's waveform error in the loudest decile). But the MP3 baseline's profile is
> nearly identical (96.7%), so the mechanism is real and explains nothing: it
> applies equally to a signal scoring +13 dB. The second cited a probe row
> reading "target's own STFT through `istft`: +144.89 dB" as proving the
> transform exact — that helper compared `istft(clean)` with `istft(clean)`,
> measuring float noise. Both are kept here because the pattern is the point:
> when a number is impossible, the first hypotheses were about the model, and
> the answer was in the ruler.

**Status of the magnitude/phase transplant table.** The figures below were taken
with the broken `istft` and are **retracted**; the comparison is being re-made.
What they appeared to show — that neither perfect phase nor perfect magnitude
rescues the output — was an artefact: both hybrids were reassembled into fresh
spectrograms and re-inverted, so both acquired their own exploded boundary.

| | SI-SDR (dB), **invalid** |
|---|---|
| MP3 input | +13.12 |
| G as-is (step 1000) | −25.10 |
| G magnitude + **target** phase | −15.71 |
| **target** magnitude + G phase | −23.15 |

**Open, and now a different question.** With the boundary fixed, the untrained
generator scores −2.97 dB and the capacity run is being repeated
(`2026-10-07-capacity-fixed-istft.txt`). The question is no longer "why is the
generator 30 dB worse than a pass-through" — that gap was mostly the ruler. It
is whether G under a pure reconstruction loss clears the MP3's **+12.82 dB**,
and then whether the adversarial objective still drives it away from its input.
The 15k run's SI-SDR column is void; its spectral columns stand, and they had
already converged (MSE 0.001, magnitude correlation 0.999, weighted phase cosine
0.993 at step 10,000), which is why it was stopped early rather than finished.

### 4d. [A] Cross-check against Apollo — 2026-10-13

[Apollo](https://github.com/JusperLee/Apollo) (Li et al., *Apollo: Band-sequence
Modeling for High-Quality Audio Restoration*) restores MP3-compressed music from
**musdb18hq stems** — our corpus, our degradation, a different architecture. It
was read specifically to check our lag alignment against a working
implementation.

**On alignment, we are ahead of them, which was not the expected answer.** Their
`match2` (`look2hear/datas/musdb_moisesdb_datamodule.py`) is GCC-PHAT: whiten
the cross-power spectrum, `argmax` the inverse transform, then
`torch.roll(wav_encdec, -tau)`. Compared with `mp3ml.align.estimate_lag`:

| | Apollo | ours |
|---|---|---|
| estimator | GCC-PHAT | plain cross-correlation, **explicitly not** GCC-PHAT — whitening amplifies phase in the empty bands a bass stem mostly consists of |
| window | the whole random crop | the **loudest** window, so a silent lead-in cannot produce a spurious lag |
| periodic material | unguarded | constrained search around `expected_lag`; a 49 Hz bass note has autocorrelation 0.984 at its period, so an unconstrained `argmax` picks between cycles arbitrarily |
| silence | returns garbage silently | raises `SilentAudioError` |
| boundary artefact | unguarded | `max_lag` capped at `size // 4`, or `argmax` returns `max_lag` and that looks like an answer |
| consistency check | none | `apollo.verify_constant_delay` requires a zero residual, and `ApolloEvalPairs` corrects the per-clip lag the manifest records |
| applying the shift | `torch.roll` — **circular**, wrapping the tail into the head | trim |

So Apollo offers no fix here; it confirms the problem is real and universal, and
their handling is the weaker of the two. Adopting it would be a regression.

**Three things it does tell us, all touching the §4a divergence:**

1. **Their discriminator learns 10x slower than their generator** —
   `lr_g: 0.001`, `lr_d: 0.0001` (`configs/apollo.yaml`). Ours are symmetric at
   1e-3, and that is the faithful reading: the paper says "the ADAM optimizer
   with a learning rate of 1e-3", singular, with no G/D distinction. Our
   diverged run had critic accuracy **1.00 from step 1**, which is exactly what
   an asymmetric rate exists to prevent — but handicapping D would be a **[D]**
   deviation, not a bug fix, so it stays on the shelf unless the faithful
   configuration fails at full size.
2. **Every loss term they reduce is a `.mean()`** — no sum/mean mixing anywhere.
   Independent corroboration that our summed penalties against a mean-reduced
   Wasserstein term (§3.1) are at minimum a smell, even though `sum` is the
   authors' own convention.
3. **They do not use WGAN-GP at all.** `MultiFrequencyDisLoss` is LSGAN
   (`(D_real - 1)².mean()` + `D_fake².mean()`) plus feature matching. A team
   solving this exact task chose to avoid the objective we are reproducing.
   That does not change what we must implement; it calibrates how much
   instability to expect from it.

Also noted, not adopted: Apollo peak-normalises each segment by a scale
**shared** between clean and codec, bounding inputs to [-1, 1] while preserving
their relative level. We only scale int16 by 32768, so segment loudness varies
freely. The paper specifies no normalisation, so this stays a **[A]** difference
rather than a change.

### 4e. [P] Verification pass — 2026-10-14

Prompted by two measurement bugs in two days (§4c), every equation the paper
states was re-derived from the paper text and checked against the code, rather
than against its own docstring. This table records what has been **audited**, as
distinct from merely written — the distinction matters, because every bug in §5
lived in code that read correctly.

| element | paper | verdict | asserted by |
|---|---|---|---|
| **eq. 5** LSD | §3.4.2 | **correct** — RMS over frequency *then* mean over frames, the asymmetric reduction implementations usually get wrong | `test_lsd_reduces_rms_over_frequency_then_mean_over_frames` |
| **eq. 6** spectral MSE | §3.4.3 | **correct** — on `sqrt(P)`, not `P`; the paper's double mean is one `mean` over both axes | `test_spectral_mse_uses_sqrt_of_power_not_power` |
| **eq. 7** SNR | §3.4.4 | **correct** — `10log10(‖s‖²/‖s−ŝ‖²)` exactly | `test_snr_matches_equation_7_by_hand` |
| **eq. 3** `L_profile` | §3.2.1 | **correct** — the "Hadamard root" is `|X| = sqrt(P)`, i.e. `magnitude`, *not* `sqrt_magnitude` = `P^(1/4)`; axis sums match the paper's own wording ("frequency profiles summing over the time axis"); p = 1.3 / 1.6 | `tests/test_losses.py` |
| `target_crop` **alignment** | §3.1.1 | **correct** — offset 62, measured against the model's real behaviour, not inferred | `test_target_crop_matches_the_generators_real_receptive_field` *(new)* |
| **best-of-20** | §4.1 | **correct** — 20 draws, and the best taken **per metric independently**, which is what "we do not pick the sample with the best quality" requires | `test_best_of_n_respects_each_metric_direction` |

**The crop test is new and the gap it closed is instructive.** The pre-existing
`test_target_crop_is_centred` only checked that the offset was *symmetric* —
it verified the config's arithmetic against itself. Nothing checked that 62 is
what the generator actually removes. Had G's reduction been asymmetric, every
loss would have been computed against a misaligned target, the loss would simply
have been higher, and the model would have taken the blame. The new test
measures by gradient which input frames influence each output frame.

It also produced a number nobody had measured: **the generator's receptive field
is 153 frames**, not the dilated stack's 125 — the early and late layers add
time extent. That is worth knowing for §4b's observation that G must learn a
near-identity through ~20 layers with no skip connection.

**Lag estimation was the expected weak point and is not one.** It is the
best-covered module here: 12 tests, including known injected shifts (0, 1, 137,
2210 and a *negative* lag), both documented traps — a silent lead-in
(`test_lag_is_measured_on_the_loudest_region_not_the_start`) and periodic bass
(`test_periodic_content_needs_a_constrained_search`) — the boundary artefact
where `argmax` returns `max_lag` and looks like an answer, silence raising
rather than guessing, and real MP3 round-trips at each bitrate.

**What this pass changes about where to look for bugs.** Six of the paper's
equations and protocols were audited and all six were already right. The bugs
that have actually cost time were *not* in the paper's maths — they were in the
STFT inversion (§4c), in a reduced config's dimensions (§3.1), and in the
batch-size coupling of two coefficients (§3.1). The pattern is that
**faithfulness to the paper has held up; the surrounding numerical plumbing has
not.** Future effort is better spent on the plumbing and on measurement
instruments than on re-reading Table 1.

### 4f. [D] The Apollo comparison run — 2026-10-21

The next run trains the paper's model on **Apollo's corpus** rather than ours, so
that this reproduction and the Apollo replication can be scored on identical
audio. `configs/apollo64.yaml`; the reader is `mp3ml.apollo`.

**Only 64 kbit/s is usable, and that was a measurement, not a choice.**

| kbit/s (mono) | MP3 output rate | in Apollo | in the paper |
|---|---|---|---|
| 24 | 16 000 Hz — resampled | yes | no |
| 32 | 22 050 Hz — resampled | yes | **yes** |
| 48 | 32 000 Hz — resampled | yes | no |
| **64** | **44 100 Hz — clean** | **yes** | **yes** |
| 96, 128 | 44 100 Hz — clean | yes | no |

Measured with `ffprobe` on real Apollo audio. Below 64 kbit/s **mono** LAME
silently resamples — the CBR floor CLAUDE.md records — which makes the pair a
different degradation from the one the paper measures, and changes MP3's priming
delay. `apollo.verify_constant_delay` caught exactly that: after trimming the
usual 1,105 samples, 32 kbit/s still had a **+1,105 residual lag**, because its
true delay is 2,210. At 64 kbit/s the residual is 0.

So 32 kbit/s is in both papers' bitrate sets and is still unusable, and
**64 kbit/s is the only rate that is in Apollo's set, in the paper's set, and
resample-free.** `degrade` raises on the rest rather than quietly producing
misaligned pairs, and `test_comparable_bitrates_are_the_only_defensible_choice`
fails if that reasoning ever stops holding.

**Why the data cannot be used as it stands.** Apollo stores originals only —
`hdf5_train/<stem>/*.hdf5` holds `"data": (n_windows, 2, 264600)` and its loader
applies MP3 inside `__getitem__`. Our pipeline reads pre-aligned pairs, so the
degraded half simply does not exist on disk. `ApolloPairs` therefore degrades at
training time, as Apollo does. The **architecture is unchanged**; only the data
layer is new.

**Deviations from Apollo, taken deliberately:**

| | Apollo | here | why |
|---|---|---|---|
| channels | stereo | left and right as separate items | the generator is single-channel — in a `(2, bins, frames)` spectrogram the 2 is real/imag. Splitting keeps Table 1 untouched and doubles the items |
| normalisation | both signals / larger peak | none | the paper specifies none |
| alignment | `match2` + `torch.roll`, leaving **7–15%** of pairs misaligned (Apollo's own README) | constant priming-delay trim, residual verified 0 | misalignment is invisible to LSD; importing a known 7–15% corruption would recreate the failure mode of §4c |
| whole window | crops then degrades | degrades then crops | cropping first gives every segment an encoder start transient no real MP3 of a song has |

**Measured cost of on-the-fly degradation** (this machine, 64 kbit/s): about
**0.8 s per item** — HDF5 read 7 ms warm (379 ms cold from USB), 6 s window
degrade 719 ms, two STFTs. At batch 12 that is ~9.5 CPU-seconds per step, so the
**data pipeline, not the GPU, can set the floor**:

| workers | s/step from data | 40 000 steps |
|---|---|---|
| 8 | 1.5 | 16.7 h |
| 16 | 0.75 | 8.3 h |
| 32 | 0.38 | 4.2 h |

`scripts/bench_cluster.py` measures this alongside step time and peak memory,
and says which side is binding. **Run it before committing to 40 000
iterations** — and note it took two attempts to measure correctly: `DataLoader`
prefetches `prefetch_factor * workers` batches and returns them in index order,
so a short median reads the queue draining, not the pipeline producing, and
reported 3,000 items/s against a known ~1.3.

**Pre-building would remove that floor.** Degrading all 77,629 windows x 2
channels once costs ~31 CPU-hours (about 2.6 h on 12 cores) and 82 GB per
bitrate, against 1.1 TB free on the external drive. Over six models x 40,000
steps that is overwhelmingly cheaper than paying 0.8 s per item per epoch.

### 4g. [D] Scope narrowed to SR-GAN on Apollo's corpus — 2026-10-21

The repository was refocused to reproduce Lattner & Nistal alone, training
exclusively from the Apollo corpus. Two earlier phases moved out to
`/home/chris/thesis/mdct-granule-pairs` (its own README and git history; the
last commit here containing them is tagged `mdct-granule-phase`):

* **Granule pairs** — LAME's pre-quantization `xr[576]` against the decoder's
  dequantized `xr_hat[576]`, read from the encoder's analysis path by a C
  dumper. The original thesis premise and a strictly tighter pairing than any
  waveform comparison, but this model operates on STFT spectrograms, so the
  granules were never its input.
* **Waveform pairs** — our own pre-aligned (clean, degraded) corpus, which fed
  this reproduction until the Apollo comparison required a shared corpus. The
  built 58 GB dataset still exists at `/home/chris/thesis/data/mp3ml-waveform`
  and is **the only data here containing 16 kbit/s**, which Apollo's corpus
  lacks — so it is the only route to the paper's headline compression rate if
  that becomes necessary again.

Neither was defective. What changed is the question being asked.

**What survived, and why.** `align.py` keeps `estimate_lag` out of the archived
`waveform.py`, because `apollo.verify_constant_delay` needs it to *prove* the
constant priming-delay trim aligns a pair rather than assuming it. The two traps
it encodes are tested here (`test_align.py`): a silent lead-in defeats
cross-correlation, and periodic bass defeats it differently — a 49 Hz note has
autocorrelation 0.984 at its period, and **GCC-PHAT makes that case worse**
because whitening amplifies phase in the empty bands a bass stem mostly consists
of. Apollo's own `match2` is GCC-PHAT, which is part of why it misaligns 7–15%
of its pairs.

**Consequences for Table 2.** Reaching it now needs six models at rates this
corpus cannot supply: 16 kbit/s is absent, and 32 kbit/s resamples in mono
(§4f). At 64 kbit/s a result is comparable to Apollo *and* to the paper's
64 kbit/s column, which is the one column this setup can honestly claim.
`COVERAGE.md` row 66 is marked **PARTIAL** accordingly.

## 5. [A] Errors worth remembering

Kept deliberately: knowing which parts were hard to get right is information
about how much to trust the result. Each was found by re-reading or measuring,
**never by a test failing**.

* **Convention beat none of five coefficients.** We assumed betas, `n_critic`,
  `gp_weight`, `drift_weight` and θ from WGAN-GP and PGGAN defaults. The authors'
  actual values differed in **all five** cases. Measurement did better: our
  measured estimate of θ's balanced range (1e-5…1e-4) was exactly right.
* **A short measurement window cannot show stability**, only the absence of fast
  divergence. Five "stable" conclusions were overturned by longer runs. Report
  survival time over a long horizon.
* **Four mechanisms for the divergence were proposed from first principles and
  all four were wrong** (`P^(1/4)` derivative explosion; the fresh optimiser
  stabilising rather than destabilising; small-batch gradient noise; lowering the
  learning rate). The real cause came from reading someone else's code.
* **The answer was in a source we had a pointer to from the start.** Reference
  [67] of the paper is the authors' own GAN codebase; it went unread until the
  systematic review, and it contained both remaining bugs.
* **A "full cross-reference" that walks the code cannot find omissions.** It
  missed two explicit sentences in §3.1.1. `COVERAGE.md` exists because of this —
  it runs from the paper to the code, not the other way.
* **`clamp_min(1e-12)` destroyed the signal.** At initialisation 100% of power
  values fell under that floor, so `L_profile`'s denominator was identically zero
  and the critic's magnitude view was a constant image. Neither raised.
* Evaluation used **random crops** until 2026-10-04, so two runs of the same
  checkpoint scored different audio.
* **A metric was read for a day before anyone checked what it says about the
  input.** Pooled magnitude correlation saturates near 0.96 for anything
  audio-shaped, and an unweighted phase cosine is dominated by near-silent bins:
  they scored 0.964 and 0.135 for the MP3 input that gets +13 dB SI-SDR, so
  §4b's "magnitude 0.394, phase random" was partly measuring the metric. **Score
  the trivial baseline with every new metric before trusting a model's number
  on it** — the pass-through is free to compute and would have caught this
  immediately.
* **Two days were spent diagnosing a model that was working, because `istft`
  amplified the segment boundary by 1e8** (§4c). Every SI-SDR and SNR in this
  file was wrong by up to 46 dB. Three things made it survive: the MP3 baseline
  was immune (identical edge artefacts cancel in the difference), the spectral
  metrics could not see it, and `test_stft.py` already trimmed the edges before
  comparing — so the limitation was *documented in the tests* while the
  production path ignored it. **When a test works around a limitation, the code
  must enforce it**, or the workaround becomes the only place the knowledge
  lives.
* **When two metrics contradict each other, measure the metrics, not the
  model.** The contradiction here (training MSE 39x better than the input while
  SI-SDR said 23 dB worse) was visible for two days and produced two confident,
  wrong, model-shaped explanations before anyone perturbed a known signal and
  checked that the two domains track. That check needs no model and no GPU and
  took two minutes.
* **A loss that falls is not a model that works.** The 2026-10-06 capacity run
  was reported as "84% better" against its own starting point while remaining 9x
  worse than the unprocessed input. Improvements are reported against the
  baseline, never against step 1.
* **A reduced config can break the architecture while every test passes.**
  `local.yaml` shrank `latent_freq` to 16 and `smoke.yaml` to 8, against a
  dilation schedule reaching 16 — silently degenerating stack layers to a single
  tap in frequency (§3.1). Shapes, parameter counts and all 200+ tests were
  green, because the tests check the config against *itself*, not against the
  faithful config's structure. Found only by asking why the small model
  underperformed. The lesson is narrower than "test more": a reduction is only
  safe along axes the paper's own hyperparameters do not reference, and
  `latent_freq` is referenced by the dilation schedule.

---

## 6. [A] Correspondence

The reply that closed §1 is in §1.1. A follow-up is drafted below — both
questions are answerable without him opening any code.

**Send to** `stefan.lattner@sony.com`, **CC** `javier.nistalhurle@sony.com`.

> Subject: Re: Questions about reproducing your compressed audio restoration paper
>
> Dear Dr Lattner,
>
> Thank you, that was extremely helpful — the eight-channel concatenation and
> the grouped convolutions cleared up the part I was most stuck on, and the two
> profile weights explained why my runs were behaving so oddly.
>
> Two small follow-ups, if you have a moment.
>
> 1. Did the model use the equalized learning rate from your
>    Comparing-Representations codebase, where weights are initialised to N(0, 1)
>    and scaled by He's constant at runtime? I ask because without it the weight
>    scales in my model span about 35x across layers, and the frequency
>    aggregation layer ends up moving around 10% of its own weight scale in a
>    single step at lr 1e-3. Switching to the runtime scaling makes that uniform.
>
> 2. Was the drift penalty summed over the batch like the gradient penalty, or
>    averaged? Your Comparing-Representations code sums it, and at batch 12 the
>    two differ by a factor of twelve.
>
> Happy to share what I end up with either way.
>
> Thanks again,
> Chris Warren
