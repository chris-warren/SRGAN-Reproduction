"""Checks on the training objectives (eq. 1, eq. 3, §3.2)."""

from pathlib import Path

import pytest
import torch

from mp3ml.config import load_config, output_frames
from mp3ml.losses import (
    critic_step_loss,
    drift_penalty,
    gradient_penalty,
    profile_loss,
    wasserstein_critic_loss,
    wasserstein_generator_loss,
)
from mp3ml.models import Critic

CFG = load_config(Path(__file__).resolve().parents[1] / "configs" / "local.yaml")
BINS, FRAMES = CFG.stft.n_bins, output_frames(CFG)


def test_critic_loss_sign_follows_equation_1() -> None:
    """Eq. 1 is maximised by D, so the minimised quantity is its negation:
    scoring real higher than fake must give a lower loss."""
    real_high = wasserstein_critic_loss(torch.full((2, 8), 5.0), torch.full((2, 8), -5.0))
    real_low = wasserstein_critic_loss(torch.full((2, 8), -5.0), torch.full((2, 8), 5.0))
    assert real_high < real_low
    assert real_high == pytest.approx(-10.0)


def test_generator_loss_rewards_fooling_the_critic() -> None:
    assert wasserstein_generator_loss(torch.full((2, 8), 5.0)) == pytest.approx(-5.0)
    assert wasserstein_generator_loss(torch.full((2, 8), 5.0)) < wasserstein_generator_loss(
        torch.full((2, 8), 1.0)
    )


def test_scores_are_averaged_over_frames() -> None:
    """§3.1.1: D emits one cost per frame, "simply averaged"."""
    per_frame = torch.arange(10, dtype=torch.float32).reshape(1, 10)
    assert wasserstein_generator_loss(per_frame) == pytest.approx(-per_frame.mean().item())


def test_drift_penalty_grows_with_critic_magnitude() -> None:
    """§3.2: penalises the magnitude of D's output on real data."""
    assert drift_penalty(torch.zeros(2, 8)) == pytest.approx(0.0)
    assert drift_penalty(torch.full((2, 8), 3.0), reduction="mean") == pytest.approx(9.0)


def test_drift_penalty_sum_scales_with_batch_size() -> None:
    """[I] The authors' code reduces the drift term by **sum** over the batch
    (`(predRealD[:,-1]**2).sum() * epsilonD`), as it does the gradient penalty.
    We averaged until 2026-10-04, making it batch_size times weaker — and the
    first real run showed `drift` climbing 0.99 -> 809, which is exactly the
    drift the term exists to suppress.
    """
    scores = torch.full((4, 8), 3.0)  # 4 samples, 8 frames each
    assert drift_penalty(scores, reduction="mean") == pytest.approx(9.0)
    # per-sample mean over frames (9.0), then summed over the 4 samples
    assert drift_penalty(scores, reduction="sum") == pytest.approx(36.0)

    with pytest.raises(ValueError, match="reduction"):
        drift_penalty(scores, reduction="median")


def test_drift_default_matches_the_configs() -> None:
    """A mismatch between the function default and the configs would make a
    direct caller silently measure something else — the same trap the gradient
    penalty had."""
    from pathlib import Path as _P

    from mp3ml.config import load_config as _load

    cfg = _load(_P(__file__).resolve().parents[1] / "configs" / "faithful.yaml")
    scores = torch.full((4, 8), 3.0)
    assert drift_penalty(scores) == pytest.approx(
        drift_penalty(scores, reduction=cfg.train.drift_reduction)
    )


def test_gradient_penalty_is_zero_for_a_unit_norm_critic() -> None:
    """Gulrajani's penalty is `(||grad|| - 1)^2`, so a critic whose gradient has
    norm exactly 1 must incur no penalty."""

    class UnitNorm(torch.nn.Module):
        def forward(self, candidate, condition):
            flat = candidate.reshape(candidate.shape[0], -1)
            # sum of one element -> gradient is a unit basis vector, norm 1
            return flat[:, :1]

    real = torch.randn(3, 2, 4, 4)
    fake = torch.randn(3, 2, 4, 4)
    gp = gradient_penalty(UnitNorm(), real, fake, torch.randn(3, 2, 4, 4))
    assert gp == pytest.approx(0.0, abs=1e-5)


def test_gradient_penalty_punishes_steep_critics() -> None:
    class Steep(torch.nn.Module):
        def forward(self, candidate, condition):
            return (candidate * 10).reshape(candidate.shape[0], -1).sum(1, keepdim=True)

    gp = gradient_penalty(
        Steep(), torch.randn(3, 2, 4, 4), torch.randn(3, 2, 4, 4), torch.randn(3, 2, 4, 4)
    )
    assert gp > 1.0


def _outputs(scale_i: float, scale_j: float) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    base = torch.randn(2, 2, BINS, FRAMES)
    return base * scale_i, base * scale_j


def test_profile_loss_is_large_when_noise_is_ignored() -> None:
    """Eq. 3 is a ratio, "maximal when the noise input to G does not influence
    the output" (§3.2.1) — the property that makes minimising it prevent collapse.
    """
    z_i, z_j = torch.randn(2, 16), torch.randn(2, 16)
    identical = torch.randn(2, 2, BINS, FRAMES)
    ignored = profile_loss(identical, identical.clone(), z_i, z_j)
    responsive = profile_loss(*_outputs(1.0, 2.0), z_i, z_j)
    assert ignored > responsive
    assert torch.isfinite(ignored), "must stay finite when the denominator vanishes"


def test_profile_loss_weights_the_two_profiles_independently() -> None:
    """[P] The authors (REPRODUCTION.md 1.1): eq. 3 writes one scalar theta, but
    "the final stochastic 32 kbit/s setup used separate weights ... 1e-5 for the
    rhythm/time profile and 1e-4 for the frequency profile"."""
    z_i, z_j = torch.randn(2, 16), torch.randn(2, 16)
    a, b = _outputs(1.0, 2.0)
    freq_only = profile_loss(a, b, z_i, z_j, theta_freq=1.0, theta_rhyt=0.0)
    rhyt_only = profile_loss(a, b, z_i, z_j, theta_freq=0.0, theta_rhyt=1.0)
    both = profile_loss(a, b, z_i, z_j, theta_freq=1.0, theta_rhyt=1.0)

    assert freq_only > 0 and rhyt_only > 0
    assert both == pytest.approx((freq_only + rhyt_only).item(), rel=1e-5)
    # Doubling one weight must move only that half.
    doubled = profile_loss(a, b, z_i, z_j, theta_freq=2.0, theta_rhyt=1.0)
    assert doubled == pytest.approx((2 * freq_only + rhyt_only).item(), rel=1e-5)


def test_gradient_penalty_sum_is_batch_times_mean() -> None:
    """[P] The authors: the original code summed the penalty over the batch
    rather than averaging, so the effective weight scales with batch size."""

    class Steep(torch.nn.Module):
        def forward(self, candidate, condition):
            return (candidate * 10).reshape(candidate.shape[0], -1).sum(1, keepdim=True)

    torch.manual_seed(0)
    real, fake, cond = (torch.randn(6, 2, 4, 4) for _ in range(3))
    torch.manual_seed(1)
    mean = gradient_penalty(Steep(), real, fake, cond, reduction="mean")
    torch.manual_seed(1)
    total = gradient_penalty(Steep(), real, fake, cond, reduction="sum")
    assert total == pytest.approx(6 * mean.item(), rel=1e-5)

    with pytest.raises(ValueError, match="reduction"):
        gradient_penalty(Steep(), real, fake, cond, reduction="median")


def test_profile_loss_uses_both_exponents() -> None:
    """Eq. 3: p = 1.3 for frequency profiles, p = 1.6 for rhythm profiles."""
    z_i, z_j = torch.randn(2, 16), torch.randn(2, 16)
    a, b = _outputs(1.0, 2.0)
    base = profile_loss(a, b, z_i, z_j)
    assert base != profile_loss(a, b, z_i, z_j, p_freq=2.0)
    assert base != profile_loss(a, b, z_i, z_j, p_rhyt=2.0)


def test_critic_step_loss_runs_and_reports_components() -> None:
    torch.manual_seed(0)
    d = Critic(CFG)
    real = torch.randn(2, 2, BINS, FRAMES) * 0.1
    fake = torch.randn(2, 2, BINS, FRAMES) * 0.1
    cond = torch.randn(2, 2, BINS, FRAMES) * 0.1
    loss, parts = critic_step_loss(d, real, fake, cond)
    assert torch.isfinite(loss)
    assert set(parts) == {"d_loss", "wasserstein", "gradient_penalty", "drift"}
    assert parts["gradient_penalty"] >= 0.0 and parts["drift"] >= 0.0


def test_no_reconstruction_term_is_present() -> None:
    """The paper has no L1/L2 reconstruction loss — conditioning the critic is
    what enforces fidelity. Guards against one creeping in."""
    import inspect

    import mp3ml.losses as losses

    source = inspect.getsource(losses)
    for banned in ("l1_loss", "mse_loss", "smooth_l1"):
        assert banned not in source, f"{banned} would be a deviation from the paper"
