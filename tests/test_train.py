"""Checks on the training loop (§3.2, §3.2.1).

Split in two: the paper-stated hyperparameters must appear verbatim in
`faithful.yaml`, and the loop mechanics run on `smoke.yaml` with synthetic
batches, so nothing here needs the corpus or a GPU.
"""

import math
from pathlib import Path

import pytest
import torch

from mp3ml.config import load_config, output_frames, target_crop
from mp3ml.train import Trainer, latest_checkpoint, noise_frames, non_finite, sample_noise

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
FAITHFUL = load_config(CONFIGS / "faithful.yaml")
SMOKE = load_config(CONFIGS / "smoke.yaml")


# ------------------------------------------------------------ paper-stated


def test_paper_training_hyperparameters() -> None:
    """§3.2: "the networks are trained for 40k iterations and a batch size of
    12" using "the ADAM optimizer with a learning rate of 1e-3", with "He's
    initialization ... for all the layers".

    These four are stated outright. An earlier truncated search of the PDF
    missed §3.2 entirely and led to inferring them from a different paper — see
    REPRODUCTION.md's changelog.
    """
    t = FAITHFUL.train
    assert t.iterations == 40_000
    assert t.batch_size == 12
    assert t.lr_g == 1e-3
    assert t.lr_d == 1e-3
    assert t.optimizer == "adam"
    assert t.init == "he"


def test_generator_uses_two_noise_draws() -> None:
    """§3.2.1: "we compute two outputs G(x_k, {z_i, z_j}_k)"."""
    assert FAITHFUL.train.noise_draws == 2


def test_profile_exponents_match_equation_3() -> None:
    assert (FAITHFUL.train.profile_p_freq, FAITHFUL.train.profile_p_rhyt) == (1.3, 1.6)


# ------------------------------------------------------------------ noise


def test_noise_is_a_per_excerpt_vector_not_a_spatial_map() -> None:
    """§3.1.1: "The noise input z (to the generator G) is **simply repeated in
    the two convolutional dimensions** and concatenated to layer Conv5", and
    §3.2.1 calls them "input noise vectors".

    We first sampled an independent value per (channel, freq, time) cell. That
    trains a different model — per-cell dither rather than one global style
    control — and inflates `||z_i - z_j||` in eq. 3 by ~100x. See §2.8.
    """
    cfg = SMOKE
    z = sample_noise(cfg, 2, torch.device("cpu"))
    assert z.shape == (2, cfg.model.noise_ch), "z must be a vector per excerpt"

    from mp3ml.models import Generator

    out = Generator(cfg)(torch.randn(2, 2, cfg.stft.n_bins, cfg.data.segment_frames), z)
    assert out.shape == (2, 2, cfg.stft.n_bins, output_frames(cfg))
    # The width it is repeated across is still the post-Conv5 time extent.
    assert noise_frames(cfg) == cfg.data.segment_frames - 2 * cfg.model.dilations[0]


def test_noise_is_constant_across_frequency_and_time_after_broadcast() -> None:
    """The defining property of "simply repeated": every cell of a given noise
    channel holds the same value. A per-cell sample would pass every shape
    check and still be the wrong model."""
    from mp3ml.models import _broadcast

    z = sample_noise(SMOKE, 2, torch.device("cpu"))
    grid = _broadcast(z, torch.Size([2, SMOKE.model.noise_ch, 8, 5]))
    assert grid.shape == (2, SMOKE.model.noise_ch, 8, 5)
    assert torch.equal(grid[..., 0, 0], z)
    assert (grid.std(dim=(-2, -1)) == 0).all(), "must be constant over freq and time"


def test_noise_actually_changes_the_output() -> None:
    """If it did not, L_profile would be dividing by zero and the stochastic
    part of "Stochastic Restoration" would be absent."""
    from mp3ml.models import Generator

    torch.manual_seed(0)
    g = Generator(SMOKE)
    y = torch.randn(1, 2, SMOKE.stft.n_bins, SMOKE.data.segment_frames)
    a = g(y, sample_noise(SMOKE, 1, torch.device("cpu")))
    b = g(y, sample_noise(SMOKE, 1, torch.device("cpu")))
    assert not torch.allclose(a, b)


def test_critic_has_skip_connections_and_generator_does_not() -> None:
    """§3.1.1: "We use PReLUs for all layers, and **skip connections for D** in
    the convolutional layers Conv6 - Conv14."

    D only, and that is self-consistent: G is unpadded in time, so its
    activations shrink at every dilated layer and there is nothing shape-
    compatible to add. D holds 212 frames throughout, so skips work.
    """
    from mp3ml.models import Critic, Generator

    g, d = Generator(SMOKE), Critic(SMOKE)
    assert g.stack.skip is False
    assert d.stack.skip is True


def test_skip_connections_change_the_critic_output() -> None:
    """Guards against `skip` being wired but never firing — the shape guard in
    DilatedStack silently disables it if channel counts do not line up."""
    import copy

    from mp3ml.models import Critic

    torch.manual_seed(0)
    with_skip = Critic(SMOKE)
    without = copy.deepcopy(with_skip)
    without.stack.skip = False

    cand = torch.randn(1, 2, SMOKE.stft.n_bins, output_frames(SMOKE)) * 0.1
    cond = torch.randn_like(cand)
    assert not torch.allclose(with_skip(cand, cond), without(cand, cond))


# ------------------------------------------------------------------- guard


def test_non_finite_names_the_offending_metrics() -> None:
    assert non_finite({"a": 1.0, "b": 2.0}) == []
    assert non_finite({"a": float("nan"), "b": 1.0, "c": float("inf")}) == ["a", "c"]


# ------------------------------------------------------------------- loop


@pytest.fixture
def trainer(tmp_path: Path) -> Trainer:
    return Trainer(SMOKE, tmp_path / "run", device="cpu")


def _batch(cfg, n: int = 2) -> dict:
    bins = cfg.stft.n_bins
    return {
        "degraded": torch.randn(n, 2, bins, cfg.data.segment_frames) * 0.1,
        "clean": torch.randn(n, 2, bins, output_frames(cfg)) * 0.1,
    }


def _batches(cfg):
    while True:
        yield _batch(cfg)


def test_train_step_reports_every_component(trainer: Trainer) -> None:
    metrics = trainer.train_step(_batches(SMOKE))
    assert set(metrics) >= {
        "d_loss",
        "wasserstein",
        "gradient_penalty",
        "drift",
        "g_loss",
        "adversarial",
        "profile",
    }
    assert non_finite(metrics) == []


def test_generator_runs_twice_per_step(trainer: Trainer) -> None:
    """§3.2.1 again, this time as a property of the loop rather than the config:
    one generator update must involve two forward passes with different noise,
    plus `n_critic` no-grad passes for the critic steps."""
    calls: list[int] = []
    trainer.g.register_forward_hook(lambda *_: calls.append(1))
    trainer.train_step(_batches(SMOKE))
    assert len(calls) == SMOKE.train.n_critic + 2


def test_critic_and_generator_update_in_the_configured_ratio(trainer: Trainer) -> None:
    """[P] The authors give a **1:1** critic-to-generator ratio, not the 5:1 we
    had assumed by WGAN convention (REPRODUCTION.md 1.1). The loop must honour
    whatever the config says, so assert against `n_critic` rather than a
    hard-coded number."""
    before_d = trainer.d.join.weight.detach().clone()
    before_g = next(trainer.g.late.parameters()).detach().clone()

    trainer.train_step(_batches(SMOKE))

    assert not torch.equal(before_d, trainer.d.join.weight)
    assert not torch.equal(before_g, next(trainer.g.late.parameters()))


def test_author_stated_training_coefficients() -> None:
    """[P] Recovered by the authors from the original implementation for the
    stochastic runs (REPRODUCTION.md 1.1). None of these are in the paper, and
    every one of them differs from what we had assumed."""
    t = FAITHFUL.train
    assert t.gp_weight == 5.0  # we assumed 10
    assert t.drift_weight == 0.5  # we assumed 1e-3
    assert t.betas == (0.0, 0.99)  # we assumed (0.5, 0.9)
    assert t.n_critic == 1  # we assumed 5
    assert t.profile_theta_freq == 1e-4  # we assumed a single theta of 1.0
    assert t.profile_theta_rhyt == 1e-5


def test_recreating_optimizers_makes_adam_behave_as_sign_sgd(tmp_path) -> None:
    """[P] The authors: "the optimizer objects were recreated for each update ...
    their Adam moment estimates were not retained across batches."

    That is the whole explanation for our divergence. A fresh Adam step from
    zeroed moments has `m_hat = g`, `v_hat = g^2`, so the update is
    `lr * g/(|g| + eps)` — sign-SGD, with the step bounded by `lr` however large
    the gradient is. Verified directly on a scalar parameter.
    """
    for grad, label in ((1e-3, "tiny"), (1e6, "huge")):
        w = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.Adam([w], lr=1e-3, betas=(0.0, 0.99))
        w.grad = torch.full_like(w, grad)
        opt.step()
        moved = -w.item()
        assert moved == pytest.approx(1e-3, rel=1e-3), (
            f"{label} gradient should still move exactly lr, moved {moved}"
        )


def test_retained_state_adam_differs_only_modestly(tmp_path) -> None:
    """The honest contrast, measured rather than assumed.

    Retaining state does **not** unbound the step: Adam normalises by a running
    RMS, so a gradient 1e9 times larger moves the weight only ~1.4x further, and
    that decays over subsequent steps. An earlier note in this repo claimed the
    recreate-optimizers detail explained our divergence — it does not. It is a
    real difference worth reproducing, but a small one. See REPRODUCTION.md 3.3.
    """
    w = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([w], lr=1e-3, betas=(0.0, 0.99))
    steps = []
    for grad in (1e-3, 1e6):
        w.grad = torch.full_like(w, grad)
        before = w.item()
        opt.step()
        steps.append(abs(w.item() - before))
    ratio = steps[1] / steps[0]
    assert 1.2 < ratio < 2.0, f"measured 1.41x; got {ratio:.2f}"


def test_critic_condition_is_the_centre_crop(trainer: Trainer) -> None:
    """The critic's candidate has `output_frames` frames while y has
    `segment_frames`; they are concatenated, so y must be cropped to the same
    window the target uses (§2.6). A front crop would misalign the condition
    from the candidate by 62 frames without any shape error."""
    y = torch.randn(2, 2, SMOKE.stft.n_bins, SMOKE.data.segment_frames)
    lo, hi = target_crop(SMOKE)
    cond = trainer._condition(y)
    assert cond.shape[-1] == output_frames(SMOKE)
    assert torch.equal(cond, y[..., lo:hi])


def test_rejects_a_config_with_no_critic_steps(tmp_path: Path) -> None:
    import copy

    cfg = copy.deepcopy(SMOKE)
    cfg.train.n_critic = 0
    with pytest.raises(ValueError, match="n_critic"):
        Trainer(cfg, tmp_path / "run")


# ------------------------------------------------------------ checkpoints


def test_checkpoint_round_trip_restores_step_and_weights(trainer: Trainer, tmp_path: Path) -> None:
    trainer.train_step(_batches(SMOKE))
    saved = trainer.save()
    weights = {k: v.detach().clone() for k, v in trainer.g.state_dict().items()}
    step = trainer.step

    trainer.train_step(_batches(SMOKE))
    assert not all(torch.equal(weights[k], v) for k, v in trainer.g.state_dict().items())

    trainer.load(saved)
    assert trainer.step == step
    for k, v in trainer.g.state_dict().items():
        assert torch.equal(weights[k], v), k


def test_latest_checkpoint_picks_the_newest_numbered_one(trainer: Trainer) -> None:
    assert latest_checkpoint(trainer.run_dir) is None
    trainer.step = 5
    trainer.save()
    trainer.step = 120
    newest = trainer.save()
    trainer.save(tag="final")  # tagged checkpoints must not win the sort
    assert latest_checkpoint(trainer.run_dir) == newest


def test_config_is_written_beside_the_checkpoints(trainer: Trainer) -> None:
    import json

    written = json.loads((trainer.run_dir / "config.json").read_text())
    assert written["name"] == "smoke"
    assert written["train"]["n_critic"] == SMOKE.train.n_critic


def test_notes_record_the_caveats_a_reader_of_results_needs(trainer: Trainer) -> None:
    """`.claude/rules/experiments.md` wants notes.md per run. Its job is to keep
    a checkpoint from outliving the reasons its numbers are not a reproduction —
    above all the unswept theta (§2.8)."""
    notes = (trainer.run_dir / "notes.md").read_text()
    assert "Fidelity mode" in notes
    assert "not `faithful.yaml`" in notes
    assert f"n_critic {SMOKE.train.n_critic}" in notes
    if SMOKE.train.gp_reduction == "sum":
        assert "SUMMED over the batch" in notes


# --------------------------------------------------- gradient checkpointing


def test_gradient_checkpointing_does_not_change_gradients() -> None:
    """It is a memory/compute trade, not a modelling change — so it must be
    numerically an identity, or it would silently become a deviation."""
    import copy

    from mp3ml.models import Generator

    torch.manual_seed(0)
    plain = Generator(SMOKE)
    checkpointed = copy.deepcopy(plain)
    checkpointed.stack.checkpoint = True

    y = torch.randn(1, 2, SMOKE.stft.n_bins, SMOKE.data.segment_frames)
    z = sample_noise(SMOKE, 1, torch.device("cpu"))

    grads = []
    for model in (plain, checkpointed):
        model.zero_grad()
        model(y, z).square().mean().backward()
        grads.append([p.grad.detach().clone() for p in model.parameters()])

    for a, b in zip(*grads, strict=True):
        assert torch.allclose(a, b, atol=1e-6)


# --------------------------------------------------- deterministic variant


def test_deterministic_config_removes_the_noise_input() -> None:
    """§4.1: "for each compression rate, we train a stochastic generator (with
    noise input z) and a deterministic generator (**without noise input**)"."""
    from mp3ml.config import deterministic

    det = deterministic(SMOKE)
    assert det.model.noise_ch == 0
    assert det.name == "smoke_det"
    assert SMOKE.model.noise_ch != 0, "must not mutate the original"


def test_deterministic_generator_runs_one_pass_and_skips_l_profile(tmp_path) -> None:
    """Without noise the two draws of §3.2.1 would be identical, so L_profile
    is meaningless — its numerator is ||z_i - z_j|| = 0. It must be skipped
    explicitly, and G must run once, not twice."""
    from mp3ml.config import deterministic

    trainer = Trainer(deterministic(SMOKE), tmp_path / "det", device="cpu")
    assert trainer.deterministic is True

    calls: list[int] = []
    trainer.g.register_forward_hook(lambda *_: calls.append(1))
    metrics = trainer.train_step(_batches(SMOKE))

    assert metrics["profile"] == 0.0
    assert len(calls) == SMOKE.train.n_critic + 1, "one generator pass, not two"
    assert non_finite(metrics) == []


def test_stochastic_generator_still_runs_twice(tmp_path) -> None:
    """The deterministic branch must not change the stochastic path (§3.2.1)."""
    trainer = Trainer(SMOKE, tmp_path / "sto", device="cpu")
    assert trainer.deterministic is False
    calls: list[int] = []
    trainer.g.register_forward_hook(lambda *_: calls.append(1))
    trainer.train_step(_batches(SMOKE))
    assert len(calls) == SMOKE.train.n_critic + 2


# ------------------------------------------------------- divergence guard


def test_divergence_guard_fires_on_finite_but_absurd_gradient_penalty(tmp_path) -> None:
    """The NaN guard cannot see a diverged WGAN. A 60-step dry run reached a
    gradient penalty of 9.2e10 — enormous, but finite, so nothing tripped and
    the run kept burning GPU time (§4.6)."""
    import copy

    cfg = copy.deepcopy(SMOKE)
    cfg.train.gp_divergence_limit = 1e4
    cfg.train.gp_divergence_patience = 3
    trainer = Trainer(cfg, tmp_path / "run", device="cpu")
    diverged = {"gradient_penalty": 9.2e10}

    assert non_finite(diverged) == [], "the value is finite — that is the point"
    trainer.check_divergence(diverged)  # 1
    trainer.check_divergence(diverged)  # 2
    with pytest.raises(RuntimeError, match="diverged"):
        trainer.check_divergence(diverged)  # 3 -> patience reached
    assert (tmp_path / "run" / "checkpoints" / "diverged.pt").exists()


def test_a_single_spike_does_not_abort(tmp_path) -> None:
    """Patience matters: a dry run spiked to 1643 at step 14 and recovered to
    0.004 by step 25. Aborting on one bad step would kill healthy runs."""
    import copy

    cfg = copy.deepcopy(SMOKE)
    cfg.train.gp_divergence_limit = 100.0
    cfg.train.gp_divergence_patience = 5
    trainer = Trainer(cfg, tmp_path / "run", device="cpu")

    for _ in range(4):
        trainer.check_divergence({"gradient_penalty": 500.0})
    trainer.check_divergence({"gradient_penalty": 0.01})  # recovery resets it
    assert trainer._gp_over == 0
    for _ in range(4):
        trainer.check_divergence({"gradient_penalty": 500.0})  # no raise


def test_healthy_metrics_never_trip_the_guard(tmp_path) -> None:
    """Real values from the healthy part of the dry run (steps 1-12)."""
    trainer = Trainer(SMOKE, tmp_path / "run", device="cpu")
    for gp in (70.57, 3.114, 0.204, 0.0371, 0.0825):
        trainer.check_divergence({"gradient_penalty": gp})
    assert trainer._gp_over == 0


def test_divergence_limit_is_well_above_healthy_values() -> None:
    """A healthy gradient penalty is (||grad||-1)^2, so O(1). The default limit
    must leave room for early-training transients (step 1 measured 70.6) while
    still catching the 1e7+ values a diverged run produces."""
    assert FAITHFUL.train.gp_divergence_limit >= 1e3
    assert FAITHFUL.train.gp_divergence_limit <= 1e6


# ------------------------------------------------- resume safety


def test_resuming_with_a_changed_config_warns(tmp_path, caplog) -> None:
    """Architecture changes fail loudly on `state_dict` load, but a changed
    learning rate or `gp_weight` is applied silently to old weights. On
    2026-09-29 every training coefficient changed at once (the authors' reply),
    which is exactly when a half-resumed run is easy to create and hard to spot.
    """
    import copy
    import logging

    trainer = Trainer(SMOKE, tmp_path / "run", device="cpu")
    saved = trainer.save()

    changed = copy.deepcopy(SMOKE)
    changed.train.lr_g = 7e-5
    changed.train.gp_weight = 99.0
    resumed = Trainer(changed, tmp_path / "run2", device="cpu")

    with caplog.at_level(logging.WARNING):
        resumed.load(saved)
    message = caplog.text
    assert "CONFIG CHANGED" in message
    assert "lr_g" in message and "gp_weight" in message


def test_resuming_with_an_identical_config_is_quiet(tmp_path, caplog) -> None:
    import logging

    trainer = Trainer(SMOKE, tmp_path / "run", device="cpu")
    saved = trainer.save()
    again = Trainer(SMOKE, tmp_path / "run2", device="cpu")
    with caplog.at_level(logging.WARNING):
        again.load(saved)
    assert "CONFIG CHANGED" not in caplog.text


# ------------------------------------------------------ periodic validation


def test_validate_uses_the_right_split_and_skips_the_slow_metrics() -> None:
    """Training losses say whether G and D are balanced, not whether the audio
    is improving. `validate` supplies the metrics that answer that — but it runs
    inside the training loop, so it must use the **val** split and must not
    invoke PEAQ (an external binary, per excerpt) or FAD (set-level, needs a
    VGGish download).
    """
    import inspect

    source = inspect.getsource(Trainer.validate)
    assert 'split="val"' in source, "must not validate on train"
    assert "with_peaq=False" in source, "PEAQ is far too slow for an inner-loop check"
    assert "with_fad=False" in source, "FAD is set-level and too slow here"


def test_validation_restores_training_mode(tmp_path) -> None:
    """`evaluate` calls `.eval()`. If that leaked, the rest of the run would
    silently continue in eval mode — inert for this model today, but a trap the
    moment anyone adds dropout or batch norm."""
    import inspect

    source = inspect.getsource(Trainer.validate)
    assert "was_training" in source and "self.g.train(was_training)" in source


def test_eval_every_zero_means_never(tmp_path) -> None:
    """The default must not slow down a run that did not ask for validation."""
    import inspect

    source = inspect.getsource(Trainer.fit)
    assert "if eval_every and" in source, "eval_every=0 must short-circuit"


# ---------------------------------------------------------- generator EMA


def test_generator_ema_tracks_the_live_weights(tmp_path) -> None:
    """[I] The authors keep an exponential moving average of G
    (`avg_p.mul_(0.999).add_(0.001, p.data)`) and **evaluate with it**, not the
    live weights. EMA generators usually produce smoother output, so scoring the
    live weights may understate results.
    """
    import copy

    cfg = copy.deepcopy(SMOKE)
    cfg.train.generator_ema = 0.5  # exaggerated so one step is visible
    trainer = Trainer(cfg, tmp_path / "run", device="cpu")
    assert trainer.g_ema is not None
    assert trainer.generator_for_eval is trainer.g_ema

    before = next(trainer.g_ema.parameters()).detach().clone()
    live_before = next(trainer.g.parameters()).detach().clone()
    trainer.train_step(_batches(SMOKE))
    after = next(trainer.g_ema.parameters()).detach()
    live_after = next(trainer.g.parameters()).detach()

    assert not torch.equal(live_before, live_after), "the live generator must move"
    assert not torch.equal(before, after), "the EMA must follow it"
    # EMA lags: it sits between its old value and the new live weights.
    expected = before * 0.5 + live_after * 0.5
    assert torch.allclose(after, expected, atol=1e-6)


def test_generator_ema_can_be_disabled(tmp_path) -> None:
    import copy

    cfg = copy.deepcopy(SMOKE)
    cfg.train.generator_ema = 0.0
    trainer = Trainer(cfg, tmp_path / "run", device="cpu")
    assert trainer.g_ema is None
    assert trainer.generator_for_eval is trainer.g


def test_ema_weights_survive_a_checkpoint_round_trip(tmp_path) -> None:
    """Losing the EMA on resume would silently change what gets evaluated."""
    trainer = Trainer(SMOKE, tmp_path / "run", device="cpu")
    trainer.train_step(_batches(SMOKE))
    saved = trainer.save()
    want = {k: v.detach().clone() for k, v in trainer.g_ema.state_dict().items()}

    trainer.train_step(_batches(SMOKE))
    trainer.load(saved)
    for k, v in trainer.g_ema.state_dict().items():
        assert torch.equal(want[k], v), k


def test_ema_requires_no_gradient(tmp_path) -> None:
    """It is a copy updated in-place, never optimised."""
    trainer = Trainer(SMOKE, tmp_path / "run", device="cpu")
    assert all(not p.requires_grad for p in trainer.g_ema.parameters())


# ----------------------------------------------------- training-batch quality


def test_quality_metrics_are_computed_on_the_training_batch(trainer: Trainer) -> None:
    """The GAN losses say whether G and D are balanced; these say whether the
    audio is getting closer to its target. Free of extra forward passes, so they
    ride the logging cadence rather than every step."""
    plain = trainer.train_step(_batches(SMOKE))
    assert "q_lsd" not in plain, "quality metrics should be opt-in"

    rich = trainer.train_step(_batches(SMOKE), with_quality=True)
    for key in ("q_lsd", "q_spectral_mse", "q_snr", "q_si_sdr"):
        assert key in rich, key
        assert math.isfinite(rich[key]), key


def test_critic_accuracy_is_a_fraction(trainer: Trainer) -> None:
    """A WGAN critic is unbounded, so its scores are hard to read. How often it
    ranks real above fake is not: ~0.5 means it cannot separate them, a
    sustained 1.0 means it has won and the generator gets no useful gradient."""
    metrics = trainer.train_step(_batches(SMOKE))
    assert "critic_accuracy" in metrics
    assert 0.0 <= metrics["critic_accuracy"] <= 1.0


def test_quality_metrics_improve_as_output_approaches_target(trainer: Trainer) -> None:
    """Sanity: the metrics must move the right way, or they are decoration."""
    clean = torch.randn(2, 2, SMOKE.stft.n_bins, output_frames(SMOKE)) * 0.1
    close = clean + torch.randn_like(clean) * 0.005
    far = clean + torch.randn_like(clean) * 0.20

    a = trainer._batch_quality(close, clean)
    b = trainer._batch_quality(far, clean)
    assert a["q_lsd"] < b["q_lsd"]
    assert a["q_spectral_mse"] < b["q_spectral_mse"]
    assert a["q_snr"] > b["q_snr"]
    assert a["q_si_sdr"] > b["q_si_sdr"]
