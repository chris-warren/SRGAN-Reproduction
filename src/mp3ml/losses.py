"""Training objectives, from §3.2 and eq. 1 / eq. 3.

The paper's total objective has no reconstruction term. Fidelity to the input is
enforced by conditioning the critic on the MP3 (`D(y, x)` in eq. 1), not by an
L1/L2 penalty — adding one would be a deviation, and an easy mistake since most
restoration work has one.

Critic loss  = -(mean D(y, real) - mean D(y, fake))        eq. 1
             + gp_weight   * gradient penalty              §3.2, Gulrajani [66]
             + drift_weight * mean D(y, real)^2            §3.2

Generator loss = -mean D(y, fake)                          eq. 1
               + L_profile                                 eq. 3
"""

from __future__ import annotations

import torch
from torch import nn

from .stft import magnitude


def wasserstein_critic_loss(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
    """Eq. 1, negated so it is minimised.

    Eq. 1 defines `Γ = mean D(y, x) - mean D(y, G(y, z))` with D *maximising* it,
    so the quantity to minimise is `-Γ`. §3.1.1: D emits one score per frame and
    "the individual costs are simply averaged".
    """
    return fake_scores.mean() - real_scores.mean()


def wasserstein_generator_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    """Eq. 1 for G: "the optimization of G only affects the second term ...
    resulting in a maximization of D(y_i, G(y_i, z_i))"."""
    return -fake_scores.mean()


def gradient_penalty(
    critic: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    condition: torch.Tensor,
    reduction: str = "sum",
) -> torch.Tensor:
    """Gulrajani [66]: `(||grad_x D(x_hat)||_2 - 1)^2` at random interpolates.

    §3.2: "gradient penalty loss, to restrict the gradients of D to 1 Lipschitz".
    Only the candidate is interpolated — the condition `y` is the same for the
    real and fake pair, so there is nothing to interpolate between.
    """
    batch = real.shape[0]
    eps = torch.rand(batch, 1, 1, 1, device=real.device, dtype=real.dtype)
    mixed = (eps * real + (1 - eps) * fake).requires_grad_(True)

    # **Differentiate the critic's effective scalar score, which is the MEAN over
    # frames** (§3.1.1: "the individual costs are simply averaged for computing
    # the final Wasserstein loss"). D emits one score per time frame, so passing
    # `grad_outputs=ones` would differentiate their *sum* and inflate the
    # gradient norm by exactly `n_frames` — 212x at the faithful config, making
    # `(||grad|| - 1)^2` up to ~45,000x too large. The Lipschitz constraint has
    # to apply to the same function the Wasserstein loss uses, or the two halves
    # of the critic objective are constraining different things.
    scores = critic(mixed, condition)
    if scores.ndim > 1:
        scores = scores.mean(dim=tuple(range(1, scores.ndim)))

    grad = torch.autograd.grad(
        outputs=scores,
        inputs=mixed,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    norm = grad.reshape(batch, -1).norm(2, dim=1)
    penalty = (norm - 1) ** 2
    # [P] The authors report that the original code **summed** the penalty over
    # the batch rather than averaging, making the effective weight ~batch_size
    # times larger. "mean" is their suggested modern improvement (§4.8).
    if reduction == "sum":
        return penalty.sum()
    if reduction == "mean":
        return penalty.mean()
    raise ValueError(f"reduction must be 'sum' or 'mean', got {reduction!r}")


def drift_penalty(real_scores: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
    """§3.2: "a loss term that penalizes the magnitudes of the output of D for
    real input data, preventing the loss from drifting" — the PGGAN drift term.

    The paper gives the term but not its coefficient; **[P] the authors supply
    0.5** (§6.2), far above the 1e-3 PGGAN default we had assumed.

    **Reduced by `mean`, and that is an [A].** The authors told us the *gradient*
    penalty was summed over the batch in the original code (§4.8) and said
    nothing about this term. Summing it too would scale it by batch size, so the
    two readings differ by 12x at the paper's batch. Recorded rather than
    guessed; see REPRODUCTION.md §2.
    """
    squared = real_scores**2
    if reduction == "sum":
        # Per-sample scores first, then summed over the batch, matching
        # `(predRealD[:,-1]**2).sum()` in the authors' code.
        if squared.ndim > 1:
            squared = squared.mean(dim=tuple(range(1, squared.ndim)))
        return squared.sum()
    if reduction == "mean":
        return squared.mean()
    raise ValueError(f"reduction must be 'sum' or 'mean', got {reduction!r}")


def _profile(spec: torch.Tensor, over_time: bool) -> torch.Tensor:
    """Frequency or rhythm profile of `spec`, per eq. 3.

    Eq. 3 uses the "Hadamard root of the power spectrum", i.e. `|X| = sqrt(P)` —
    `magnitude`, not `sqrt_magnitude`. §3.2.1 defines the two profiles as the
    "frequency profiles (summing over the time axis) and the rhythm profiles
    (summing over the frequency axis)".
    """
    mag = magnitude(spec).squeeze(-3)  # (B, freq, time)
    return mag.sum(dim=-1) if over_time else mag.sum(dim=-2)


def profile_loss(
    out_i: torch.Tensor,
    out_j: torch.Tensor,
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    theta_freq: float = 1e-4,
    theta_rhyt: float = 1e-5,
    p_freq: float = 1.3,
    p_rhyt: float = 1.6,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Eq. 3, summed over its frequency and rhythm variants.

        L_profile = theta * ||z_i - z_j|| / || profile(G_zi) - profile(G_zj) ||_p

    A *ratio*, and the sole reason it works: it is maximal when the noise does
    not influence the output (denominator -> 0), so minimising it forces G to
    respond to `z`. §3.2.1: "we define an additional cost term which is maximal
    when the noise input to G does not influence the output of G".

    `p = 1.3` for frequency profiles and `p = 1.6` for rhythm profiles (eq. 3).

    **Eq. 3 writes one scalar `theta`, but the authors used two** (§6.2):
    "the final stochastic 32 kbit/s setup used separate weights rather than a
    single shared theta: 1e-5 for the rhythm/time profile and 1e-4 for the
    frequency profile". So the two variants are weighted independently, and the
    frequency profile is weighted 10x more heavily than the rhythm profile.
    """
    batch = z_i.shape[0]
    z_dist = (z_i - z_j).reshape(batch, -1).norm(2, dim=1)

    total = out_i.new_zeros(())
    for over_time, p, theta in ((True, p_freq, theta_freq), (False, p_rhyt, theta_rhyt)):
        diff = _profile(out_i, over_time) - _profile(out_j, over_time)
        total = total + theta * (z_dist / (diff.norm(p=p, dim=-1) + eps)).mean()
    return total


def critic_step_loss(
    critic: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    condition: torch.Tensor,
    gp_weight: float = 5.0,
    drift_weight: float = 0.5,
    gp_reduction: str = "sum",
    drift_reduction: str = "sum",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Full critic objective. Returns the loss and its components for logging."""
    real_scores = critic(real, condition)
    fake_scores = critic(fake, condition)
    wasserstein = wasserstein_critic_loss(real_scores, fake_scores)
    gp = gradient_penalty(critic, real, fake, condition, reduction=gp_reduction)
    drift = drift_penalty(real_scores, reduction=drift_reduction)
    loss = wasserstein + gp_weight * gp + drift_weight * drift
    return loss, {
        "d_loss": loss.item(),
        "wasserstein": -wasserstein.item(),  # report the estimated distance
        "gradient_penalty": gp.item(),
        "drift": drift.item(),
    }
