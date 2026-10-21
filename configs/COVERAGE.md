# Specification coverage matrix

Every normative statement in §3 and §4.1 of the paper, mapped to exactly one of:

| status | meaning |
|---|---|
| **DONE** | implemented, with the test that holds it in place |
| **[A]** | recorded assumption — the paper does not say |
| **[D]** | recorded deliberate deviation |
| **N/A** | not a requirement on the implementation (motivation, hardware, prose) |
| **GAP** | **specified and not implemented** |

## Why this file exists, and how it differs from REPRODUCTION.md §1

§1.2 walks *our code* and asks "does the paper support this?". That direction
can only find errors in things we already built — **it cannot find omissions**,
and every serious bug so far has been an omission. It claimed to be a "full
cross-reference" and still missed two explicit sentences in §3.1.1 (the noise
being a repeated vector, and the critic's skip connections), because neither
corresponded to any line of code to walk.

This file runs the other direction: **driven from the paper, sentence by
sentence.** Anything the paper says that has no row here is missing by
construction. That is the only way omissions become visible.

Source text: `pdftotext -layout` of the published PDF. Regenerate with
`pdftotext -layout <paper>.pdf paper.txt`; line numbers below refer to that
extract. The paper is CC BY (MDPI open access).

**Audit date: 2026-09-25.** Re-run this whenever the paper is re-read.

---

## §3.1 Model Architecture

| # | Paper says | Status | Where |
|---|---|---|---|
| 1 | GAN with generator G and critic D | DONE | `models.Generator`, `models.Critic` |
| 2 | G input: MP3 spectrogram `y`, non-linearly scaled complex STFT | DONE | `apollo.ApolloPairs` · `test_data` |
| 3 | G outputs restored `x̂` approximating original `x` | DONE | `test_config::faithful_config_reproduces_table_1_exactly` |
| 4 | D distinguishes `x̂` from `x` | DONE | `test_losses` |
| 5 | "In addition to the true/fake data, **D also receives the MP3 versions**" | DONE | **[P] authors §6.2**: channel-wise concat, 8 channels · `test_models::critic_takes_eight_channels_in_two_groups` |
| 6 | "We test **stochastic and deterministic** generators" | DONE | `config.deterministic` · `test_train::deterministic_generator_runs_one_pass_and_skips_l_profile` |
| 7 | Stochastic models get noise `z ~ N(0, I)` | DONE | `train.sample_noise` |
| 8 | Eq. 1: `Γ = 1/N Σ D(y,x) − D(y,G(y,z))` | DONE | `losses.wasserstein_*` · `test_losses::critic_loss_sign_follows_equation_1` |
| 9 | `min_G max_D Γ` — D maximises, G minimises | DONE | `test_losses::critic_loss_sign_follows_equation_1` |
| 10 | "optimization of G only affects the second term" | DONE | `losses.wasserstein_generator_loss` |

## §3.1.1 Architecture Details

| # | Paper says | Status | Where |
|---|---|---|---|
| 11 | Table 1 is the architecture | DONE | `test_config::faithful_config_reproduces_table_1_exactly` (all 33 rows) |
| 12 | G and D both convolutional in time | DONE | #14, #16 |
| 13 | "allows us to use less overlap (i.e., **50%**)" when chopping training data | **[D]** | random offsets instead — §4.4 |
| 14 | "At test time, G is applied to **variable-length** ... input sequences (e.g., full songs)" | DONE | `test_models::generator_accepts_variable_length_input` |
| 15 | "we do **not** use zero-padding for G during training" | DONE | `test_config::faithful_config_reproduces_table_1_exactly` (336→212) |
| 16 | D emits "as many loss outputs as there are spectrogram frames" | DONE | `test_models::critic_emits_one_score_per_frame` |
| 17 | "the individual costs are **simply averaged**" | DONE | `test_losses::scores_are_averaged_over_frames` |
| 18 | "**two convolutional groups** throughout the critic stack" | DONE | **[P] authors §6.2**: PyTorch `groups=2` · `test_models::every_critic_conv_is_grouped_except_the_output_layer` |
| 19 | "Only in the output layer, those two groups are **joined again**" | DONE | **[P] authors §6.2**: Conv16 is `groups=1` · `test_models::critic_takes_eight_channels_in_two_groups` |
| 20 | Two views: signed-sqrt complex STFT, and sqrt magnitude | DONE | `Critic.complex_*` / `magnitude_*` · `test_stft::magnitude_and_sqrt_magnitude_are_different_transforms` |
| 21 | sqrt-magnitude beat log-magnitude empirically | N/A | justification |
| 22 | Non-symmetrical, not U-Net, no bottleneck | DONE | `models` has neither |
| 23 | "the main part of the architectures of G and D are **identical**" | DONE | both use `FrequencyAggregation` + `DilatedStack` |
| 24 | "Only the **top parts** of the stacks differ" | DONE | G: DeConv4–1; D: Conv15–16 |
| 25 | "PReLUs for **all** layers" | DONE | `test_config::faithful_config_reproduces_table_1_exactly` non-linearity column |
| 26 | "**skip connections for D** in the convolutional layers Conv6 - Conv14" | DONE | `DilatedStack(skip=True)` · `test_train::critic_has_skip_connections_and_generator_does_not` — **was a GAP until 2026-09-25** |
| 27 | Noise "**simply repeated in the two convolutional dimensions**" and concatenated to Conv5 | DONE | `models._broadcast` · `test_train::noise_is_a_per_excerpt_vector_not_a_spatial_map` — **was a GAP until 2026-09-25** |

## §3.1.2 Gated Convolutions

| # | Paper says | Status | Where |
|---|---|---|---|
| 28 | Self-gating from [63] Dauphin, but **PReLU not linear** ("linear units resulted in unstable training") | DONE | `SelfGatedConv2d` · §1.1 · `test_models::self_gating_is_prelu_times_sigmoid_not_glu` |
| 29 | "half the output maps ... element-wise gate the other half" | DONE | `.chunk(2, dim=1)` |
| 30 | "sigmoid on the gating units and PReLUs on the gated units" | DONE | `test_models` |
| 31 | "one layer with **twice as many output maps** as input maps" | DONE | `Conv2d(in, 2*out)` |
| 32 | Eq. 2 | DONE | `SelfGatedConv2d.forward` |
| 33 | "where **S = R** in our architecture" | **note** | holds for Conv7–14 (128→128). Not for Conv6, which Table 1 gives as `(320)[256] → 256 → 128`. Table 1 is the authority; the prose is a simplification |
| 34 | "applied to all convolutional layers Conv6 - Conv14" | DONE | `DilatedStack.gated` covers exactly those |

## §3.1.3 Frequency Aggregation Filters

| # | Paper says | Status | Where |
|---|---|---|---|
| 35 | Motivation: frequency correlations are non-local | N/A | motivation |
| 36 | Conv4: "4096 filter kernels that span the whole frequency dimension and **only convolve in time**" | DONE | `Conv2d(e2, 4096, (n_bins, 1))` |
| 37 | "i.e., **no padding in the frequency dimension**" | DONE | no padding arg · Table 1 `0, 0` |
| 38 | Reshape1: "separated into **32 groups of size 128** each" | **note** | we reshape `(4096,1) → (128,32)`, i.e. channel-major. The alternative grouping is a fixed permutation of Conv4's output filters, which are free learned parameters — so the two are equivalent and neither is a choice with consequences |
| 39 | Recombined "through a stack of dilated convolutions (Conv5 - Conv14)" | DONE | `DilatedStack` |
| 40 | "In G, Reshape2 reshapes back to 4096 ... and DeConv4 inverts the frequency aggregation" | DONE | `Generator.forward` |

## §3.2 Training Procedure

| # | Paper says | Status | Where |
|---|---|---|---|
| 41 | "**40k iterations**" | DONE | `test_train::paper_training_hyperparameters` |
| 42 | "batch size of **12**" | DONE | same |
| 43 | "~2 days on two NVIDIA Titan RTX 24GB" | N/A | hardware |
| 44 | "**ADAM** optimizer [65] with a learning rate of **1e-3**" | DONE | same |
| 45 | "**gradient penalty** loss, to restrict the gradients of D to 1 Lipschitz [66]" | DONE | `losses.gradient_penalty` · `test_losses` |
| 46 | "loss term that **penalizes the magnitudes of the output of D for real input data**" | DONE | `losses.drift_penalty` |
| 47 | "**He's initialization** is performed for all the layers" | DONE | `models.he_init` |
| 48 | gradient-penalty **weight** | DONE | **[P] authors §6.2**: **5** (we had assumed 10) · `test_train::author_stated_training_coefficients` |
| 49 | drift-penalty **weight** | DONE | **[P] authors §6.2**: **0.5** (we had assumed 1e-3) · same test |
| 50 | Adam **β** values | DONE | **[P] authors §6.2**: **(0, 0.99)** (we had assumed (0.5, 0.9)) · same test |
| 51 | **critic steps per generator step** | DONE | **[P] authors §6.2**: **1:1** (we had assumed 5) · same test |

## §3.2.1 Preventing Mode Collapse

| # | Paper says | Status | Where |
|---|---|---|---|
| 51a | *optimiser state across updates* | **[P/D]** | not in the paper; authors §6.2: **recreated each update** (= sign-SGD). `recreate_optimizers` — §4.8 |
| 51b | *gradient-penalty reduction* | **[P/D]** | not in the paper; authors §6.2: **summed over the batch**, not averaged. `gp_reduction` — §4.8 |
| 52 | "G tends to ignore the input noise z" | N/A | motivation |
| 53 | Eq. 3 `L_profile` | DONE | `losses.profile_loss` · `test_losses` |
| 54 | Ratio uses "Euclidean distance of two arbitrary **input noise vectors**" | DONE | `sample_noise` returns a vector (see #27) |
| 55 | Frequency profiles = "summing over the **time** axis" | DONE | `losses._profile(over_time=True)` |
| 56 | Rhythm profiles = "summing over the **frequency** axis" | DONE | `losses._profile(over_time=False)` |
| 57 | "**Hadamard root of the power spectrum**" = `sqrt(P)` | DONE | `stft.magnitude` — §2.7 |
| 58 | `d` = column vector of 1s (freq) / row vector of 1s (rhythm) | DONE | implemented as `.sum()` over the axis — identical |
| 59 | "The scalar ϑ controls the strength of the regularization" | DONE | **[P] authors §6.2**: **two** weights, 1e-4 freq / 1e-5 rhythm · `test_losses::profile_loss_weights_the_two_profiles_independently`. No longer blocking |
| 60 | "p = 1.3 for frequency profiles and p = 1.6 for rhythm profiles" | DONE | `test_train::profile_exponents_match_equation_3` |
| 61 | "we compute **two outputs** G(x_k, {z_i, z_j}_k)" per conditional input | DONE | `test_train::generator_runs_twice_per_step` |
| 62 | "use those outputs to compute Lprofile, **as well as the common gradient update of G**" | DONE | both draws feed the adversarial term |

## §3.3 Data

| # | Paper says | Status | Where |
|---|---|---|---|
| 63 | Pairs of MP3 and high-quality 44.1 kHz | DONE | `apollo.py` · REPRODUCTION.md §4f |
| 64 | "~64 hours of Nr 1 songs of the US charts 1950–2020" | **[D]** | musdb18hq, 47 h — §4.2 |
| 65 | "**LAME** MP3 codec, version **3.100**" | DONE | `apollo.degrade` (CBR via the `lame` CLI) |
| 66 | "16kbit/s, 32kbit/s and 64kbit/s **mono**" | **PARTIAL** | mono confirmed, but Apollo's corpus has no 16 kbit/s and only 64 is resample-free in mono — REPRODUCTION.md §4f |
| 67 | Split "**80%, 10%, 10%**" train/eval/test | **[D]** | 59/6/35, musdb's canonical test split — §4.4 |
| 68 | "4-second-long segments with **50% overlap** for training and validation" | **[D]** | random offsets — §4.4 |
| 69 | "For the **subjective** evaluation, we split the songs into segments of **8 seconds**" | **[D]** | tied to #81 — MOS not run |

## §3.3.1 Data Representation

| # | Paper says | Status | Where |
|---|---|---|---|
| 70 | Complex STFT components as the main representation | DONE | `stft.stft` |
| 71 | "window size **2048**" | DONE | `StftConfig.n_fft` |
| 72 | "hop size of **512**" | DONE | `StftConfig.hop` |
| 73 | Signed square-root applied to **each component separately** | DONE | `test_stft::signed_sqrt_applies_per_component_not_to_magnitude` |
| 74 | Eq. 4: `σ(r) = sign(r)·sqrt(|r|)` | DONE | `stft.signed_sqrt` · `test_stft` |
| 75 | *window function* | **[A]** | never named; we use Hann — §3. Found unrecorded in this audit |
| 76 | *1024 vs 1025 bins* | **[A]** | drop Nyquist — §2.3 |

## §3.4 Evaluation

| # | Paper says | Status | Where |
|---|---|---|---|
| 77 | Metrics: LSD, MSE, SNR, ODG, DI, MOS | DONE except MOS | `metrics.py` |
| 78 | ODG/DI from **PEAQ basic**, `peaqb-fast` | DONE (wrapper) | `peaq.py` — **binary not installed**, so ODG/DI cannot be produced yet |
| 79 | ODG ranges 0 to −4, lower is worse | DONE | `metrics.HIGHER_IS_BETTER` |
| 80 | Eq. 5 LSD, rms over frequency then mean over frames | DONE | `test_metrics::lsd_reduces_rms_over_frequency_then_mean_over_frames` |
| 81 | Eq. 6 MSE between `sqrt(P)` | DONE | `test_metrics::spectral_mse_uses_sqrt_of_power_not_power` |
| 82 | Eq. 7 SNR | DONE | `test_metrics::snr_matches_equation_7_by_hand` |
| 83 | LSD floor for silent bins | **[A]** | not stated; `LOG_FLOOR = 1e-10` — §2.9 |
| 84 | **MOS**: 15 expert listeners, 5-point Likert, 70 ratings each | **[D]** | human study, not run — see below |

## §4.1 Objective Evaluation Protocol

| # | Paper says | Status | Where |
|---|---|---|---|
| 85 | "**for each compression rate**, we train a stochastic and a deterministic generator" | DONE | `--bitrate`, `--deterministic` — §2.10, §4.4 |
| 86 | "we **sample 20 times** ... and for each objective metric, we take the **best** value of that set" | DONE | `evaluate._best` · `test_metrics::best_of_n_respects_each_metric_direction` |
| 87 | "all objective metrics are computed by comparing the **restored data with the original** versions" | DONE | `evaluate.evaluate` |
| 88 | Higher better for ODG/DI/SNR; lower for LSD/MSE | DONE | `metrics.HIGHER_IS_BETTER` |
| 89 | §4.1.1: a fixed `z` gives a consistent frequency profile across examples | **not tested** | independently corroborates #27 (z is a global style vector). Worth reproducing as a sanity check once a model trains |

---

## Gaps and open items

**Updated 2026-09-29, after the authors replied (§6.2).**

No GAP rows remain, and **no row is blocked on missing information any more** —
the four [A] rows that were blocking (#5, #18/#19, #48-51, #59) are now [P].

What is left:

1. **Our training diverges under the authors' own settings** (§4.6). The only
   thing between us and a real run, and an implementation problem rather than a
   specification gap.
2. **[D] The two fidelity switches** (#51a, #51b, §4.8) — a deliberate choice
   the authors offered, not an unknown.
3. **PEAQ binary not installed** (#78) — ODG and DI are two of Table 2's five
   columns and cannot be produced at all. Cheapest remaining fix.
4. **MOS not run** (#84, #69) — a 15-participant listening test. The paper's
   headline claim (stochastic ≈ deterministic in perceived quality) cannot be
   checked without it.
5. **Four data-side deviations** (#64, #67, #68, #69).
6. **#89 not tested** — a fixed `z` giving a consistent frequency profile.
   Cheap, and it independently validates the noise design once a model trains.

One caveat on the new [P] rows: the authors described the recovered settings as
those of the **stochastic 32 kbit/s** run, so they may differ for 16 and
64 kbit/s and for the deterministic models.

## What this audit changed

* Found and fixed a real gap: the **deterministic generator** (#6) was
  supported only accidentally — `L_profile` evaluated to 0 by luck, and G ran
  twice producing identical outputs. Now explicit, with `config.deterministic`
  and a `--deterministic` flag.
* Found and verified two previously untested functional requirements:
  **variable-length inference** (#14) and **one critic score per frame** (#16).
* Found one unrecorded assumption: the **STFT window function** (#75).
* Resolved two items that had looked like choices but are not: **eq. 2's
  `S = R`** (#33) and **Reshape1's grouping order** (#38).
