"""Objective metrics.

Two groups, deliberately kept apart:

**§3.4 of the paper** — `lsd` (eq. 5), `spectral_mse` (eq. 6), `snr` (eq. 7),
and ODG/DI from PEAQ (`peaq.py`, external binary). These reproduce the paper's
Table 2 and must not be "improved".

**Modern additions** — `si_sdr` and `multi_resolution_lsd`. Not in the paper;
included because eq. 7 is gain-sensitive and eq. 5 sees one resolution only.
They are reported alongside, never in place of, the paper's numbers. Any table
mixing the two must say which is which.

All functions take `(..., samples)` waveforms or `(..., 2, bins, frames)`
signed-sqrt spectra as noted, return plain floats or tensors, and hold no state
— so any logging backend can consume them.

Reference values from the paper's Table 2, for sanity-checking an
implementation against a known target (mono, per bitrate, best-of-20):

    mp3_16k  ODG -3.08  DI -1.67  LSD 10.98  MSE 0.40  SNR 13.69
    sto_16k  ODG -2.80  DI -1.19  LSD  3.72  MSE 0.26  SNR  9.51
    mp3_32k  ODG -3.04  DI -1.56  LSD  9.75  MSE 0.31  SNR 13.67
    sto_32k  ODG -2.74  DI -1.07  LSD  3.75  MSE 0.26  SNR  9.57
    mp3_64k  ODG -2.64  DI -0.86  LSD  4.89  MSE 0.07  SNR 17.85
    sto_64k  ODG -2.74  DI -1.02  LSD  3.59  MSE 0.17  SNR 11.51

Note the MP3 baseline beats the model on SNR at every bitrate: the model adds
plausible high frequencies that are not the original ones, which helps LSD and
perception but hurts a sample-wise residual. Expect the same shape.
"""

from __future__ import annotations

import torch

from .stft import power, stft

#: Floor on power before a log, in the paper's metrics. Eq. 5 divides two power
#: spectra and takes a log, so silent bins would give -inf. The paper does not
#: state a floor, which is one reason absolute LSD values are hard to match
#: across implementations — see REPRODUCTION.md §1.3.
LOG_FLOOR = 1e-10


# --------------------------------------------------------------- paper, §3.4


def lsd(clean: torch.Tensor, restored: torch.Tensor, floor: float = LOG_FLOOR) -> torch.Tensor:
    """Log-spectral distance, **eq. 5**.

        LSD = 1/L * sum_l sqrt( 1/W * sum_f [ 10*log10( P(l,f) / P_hat(l,f) ) ]^2 )

    Note the asymmetry of the reduction: root-mean-square **over frequency**,
    then a plain mean **over frames**. Averaging both at once, or taking the
    root at the end, gives a different number — a common way implementations
    disagree.

    Inputs are signed-sqrt spectra `(..., 2, bins, frames)`; `power` converts.
    """
    p, p_hat = power(clean).clamp_min(floor), power(restored).clamp_min(floor)
    per_bin = (10.0 * torch.log10(p / p_hat)) ** 2  # (..., bins, frames)
    per_frame = per_bin.mean(dim=-2).sqrt()  # rms over frequency
    return per_frame.mean(dim=-1)  # mean over frames


def spectral_mse(clean: torch.Tensor, restored: torch.Tensor) -> torch.Tensor:
    """MSE between the **square roots of the power spectra**, eq. 6.

    §3.4.3 gives the reason for preferring this to eq. 5: MP3 lowpasses, so the
    log in eq. 5 turns near-zero high bins into large negative numbers and
    "generally favors algorithms that add frequencies in the upper range (like
    the proposed method)". Eq. 6 is the fairer comparison, and the paper says so
    about its own method.
    """
    diff = power(clean).sqrt() - power(restored).sqrt()
    return (diff**2).mean(dim=(-2, -1))


def snr(clean: torch.Tensor, restored: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Signal-to-noise ratio in dB, **eq. 7**: `10*log10(||s||^2 / ||s - s_hat||^2)`.

    Time domain, so — as §3.4.4 says — "highly sensitive to phase information".
    It is also sensitive to overall gain, which `si_sdr` below is not.
    """
    noise = clean - restored
    return 10.0 * torch.log10(
        clean.pow(2).sum(dim=-1).clamp_min(eps) / noise.pow(2).sum(dim=-1).clamp_min(eps)
    )


# ------------------------------------------------ modern additions (NOT §3.4)


def si_sdr(clean: torch.Tensor, restored: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Scale-invariant SDR in dB (Le Roux et al., ICASSP 2019). **Not in the paper.**

    Eq. 7 punishes a restoration that is right in every way but quieter. SI-SDR
    projects the estimate onto the reference first, so a pure gain error scores
    identically. Worth having next to `snr` precisely because the generator's
    output level is not pinned by anything in the paper's objective — there is
    no reconstruction loss, so nothing forces G to match absolute amplitude.

    A large gap between `snr` and `si_sdr` means a gain problem, not a
    restoration problem.
    """
    clean = clean - clean.mean(dim=-1, keepdim=True)
    restored = restored - restored.mean(dim=-1, keepdim=True)
    scale = (restored * clean).sum(dim=-1, keepdim=True) / clean.pow(2).sum(
        dim=-1, keepdim=True
    ).clamp_min(eps)
    target = scale * clean
    noise = restored - target
    return 10.0 * torch.log10(
        target.pow(2).sum(dim=-1).clamp_min(eps) / noise.pow(2).sum(dim=-1).clamp_min(eps)
    )


def multi_resolution_lsd(
    clean_wav: torch.Tensor,
    restored_wav: torch.Tensor,
    n_ffts: tuple[int, ...] = (512, 2048, 8192),
    floor: float = LOG_FLOOR,
) -> torch.Tensor:
    """Eq. 5 averaged over several STFT resolutions. **Not in the paper.**

    The paper evaluates at the single resolution it trains on (2048/512). A
    restoration can look right there and still be wrong at others: coarse
    windows expose sustained tonal errors, fine windows expose smeared
    transients. Averaging three resolutions makes that visible without
    replacing the paper's number.

    Takes **waveforms**, since each resolution needs its own transform.

    Resolutions wider than the signal are dropped rather than raising: the
    generator's output is only `output_frames` long, so a short config (or a
    short excerpt) legitimately cannot support an 8192-sample window.
    """
    length = min(clean_wav.shape[-1], restored_wav.shape[-1])
    usable = [n for n in n_ffts if n < length]
    if not usable:
        raise ValueError(
            f"no STFT resolution in {n_ffts} fits a {length}-sample signal; "
            "pass a smaller n_ffts or evaluate longer excerpts"
        )
    scores = [
        lsd(
            stft(clean_wav, n_fft=n, hop=n // 4, n_bins=n // 2),
            stft(restored_wav, n_fft=n, hop=n // 4, n_bins=n // 2),
            floor=floor,
        )
        for n in usable
    ]
    return torch.stack(scores).mean(dim=0)


def _psd_sqrt(mat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Matrix square root of a symmetric positive-semidefinite matrix."""
    mat = 0.5 * (mat + mat.transpose(-1, -2))  # kill asymmetry from round-off
    vals, vecs = torch.linalg.eigh(mat)
    return (vecs * vals.clamp_min(eps).sqrt().unsqueeze(-2)) @ vecs.transpose(-1, -2)


def frechet_distance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Frechet distance between two sets of embeddings. **Not in the paper.**

    With FAD's embedder this is Frechet Audio Distance (Kilgour et al.,
    Interspeech 2019); the maths is the same as FID's::

        ||mu_a - mu_b||^2 + tr(S_a + S_b - 2*sqrt(S_a @ S_b))

    Each input is `(n_samples, n_features)`.

    Why it earns its place next to the paper's metrics: LSD, MSE and SNR all
    compare a restoration to *the* original sample by sample, so a restoration
    that is perfectly plausible but different is punished. That is precisely
    the case this model is built to produce — §4.1 concedes as much by taking
    the best of 20 draws. FAD compares *distributions*, so it asks the question
    the paper's metrics cannot: does the model's output sound like real music?

    `sqrt(S_a @ S_b)` is not symmetric, so it is computed through the identity
    `tr sqrt(S_a S_b) = tr sqrt(sqrt(S_a) S_b sqrt(S_a))`, whose argument is
    symmetric PSD. That keeps this dependency-free — no `scipy.linalg.sqrtm`.
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"expected (n, features) embeddings, got {tuple(a.shape)}/{tuple(b.shape)}"
        )
    if a.shape[0] < 2 or b.shape[0] < 2:
        raise ValueError("need at least 2 samples per set to estimate a covariance")

    a, b = a.double(), b.double()
    mu_a, mu_b = a.mean(0), b.mean(0)
    # atleast_2d: torch.cov collapses to a 0-dim scalar for a single feature,
    # which the matrix maths below cannot consume.
    cov_a, cov_b = torch.atleast_2d(torch.cov(a.T)), torch.atleast_2d(torch.cov(b.T))
    eye = torch.eye(cov_a.shape[0], dtype=cov_a.dtype, device=cov_a.device) * eps
    cov_a, cov_b = cov_a + eye, cov_b + eye

    root_a = _psd_sqrt(cov_a)
    cross = _psd_sqrt(root_a @ cov_b @ root_a)
    return (mu_a - mu_b).pow(2).sum() + torch.trace(cov_a + cov_b - 2 * cross)


#: Which metrics improve as they *rise*. Anything reporting or ranking metrics
#: must consult this — the paper's set mixes directions (§4.1: "higher values
#: are better for ODG, DI and SNR; lower values are better for LSD and MSE"),
#: and the best-of-N protocol in `evaluate.py` picks the wrong sample if this is
#: wrong.
HIGHER_IS_BETTER = {
    "lsd": False,
    "spectral_mse": False,
    "snr": True,
    "si_sdr": True,
    "multi_resolution_lsd": False,
    "odg": True,
    "di": True,
}
