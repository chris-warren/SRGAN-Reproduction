"""The paper's spectrogram representation: signed square-root complex STFT.

§3.3.1: "The main representation used in the proposed method are the complex
STFT components... The STFT is computed with window size 2048, and a hop size of
512. In addition, we perform non-linear scaling to all complex components... we
transform each complex STFT coefficient `h = a + ib` by taking the signed
square-root of each of its components", where

    sigma(r) = sign(r) * sqrt(|r|)                                       (eq. 4)

The scaling is applied to the real and imaginary parts **separately**, not to
the magnitude — hence "each of its components".

Framing follows REPRODUCTION.md §2: `center=False`, and enough samples for
exactly `segment_frames` frames. The Nyquist bin is dropped so the frequency
axis is 1024, matching Table 1 (§2.3).
"""

from __future__ import annotations

import torch


def signed_sqrt(x: torch.Tensor) -> torch.Tensor:
    """Eq. 4: `sign(r) * sqrt(|r|)`, applied elementwise."""
    return torch.sign(x) * torch.sqrt(torch.abs(x))


def signed_square(x: torch.Tensor) -> torch.Tensor:
    """Inverse of `signed_sqrt`."""
    return torch.sign(x) * x**2


def stft(wav: torch.Tensor, n_fft: int = 2048, hop: int = 512, n_bins: int = 1024) -> torch.Tensor:
    """Waveform `(..., samples)` -> `(..., 2, n_bins, frames)`, signed-sqrt scaled.

    Channel 0 is the real part, channel 1 the imaginary part. `center=False`
    because the paper's frame count fixes the architecture's time axis and
    centred framing would pad the signal (§2.5).
    """
    window = torch.hann_window(n_fft, device=wav.device, dtype=wav.dtype)
    spec = torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop,
        window=window,
        center=False,
        return_complex=True,
    )
    spec = spec[..., :n_bins, :]  # drop the Nyquist bin -> 1024 (§2.3)
    return torch.stack([signed_sqrt(spec.real), signed_sqrt(spec.imag)], dim=-3)


def istft(
    spec: torch.Tensor, n_fft: int = 2048, hop: int = 512, length: int | None = None
) -> torch.Tensor:
    """Inverse of `stft`, undoing the signed-sqrt scaling and restoring Nyquist.

    Overlap-add is done explicitly rather than via `torch.istft`, which refuses
    `center=False` input: a Hann window is zero at its endpoints, so the first
    and last frames fail its NOLA check even though the interior is fine.

    **Samples without full window coverage are returned as zeros**, not as a
    best effort. With `center=False` the first and last `n_fft - hop` samples
    are reached by fewer frames than the interior, and the frames that would
    have completed them do not exist — those samples are unrecoverable, not
    merely noisy. Dividing by the overlap-add window sum there means dividing
    by a number that goes to **exactly zero** at the boundary.

    This used to read `norm.clamp(min=1e-8)`, and the docstring claimed the
    edges "degrade gracefully". They did not: the clamp turned the boundary into
    a gain of up to 1e8, and because every waveform metric sums squared error
    over the whole signal, those few samples dominated all of them. Measured
    2026-10-07 — SI-SDR on a perturbed target read **-10.03 dB** against the
    **+22.39 dB** that the same perturbation scores on the valid interior, a
    32 dB error that made a working generator look catastrophic. See
    REPRODUCTION.md §4c.

    The zeros are deliberate and safe for the callers: `clean` and `restored`
    both pass through here, so both are zeroed identically and the metric is
    computed on exactly the reconstructable samples.
    """
    real, imag = signed_square(spec[..., 0, :, :]), signed_square(spec[..., 1, :, :])
    complex_spec = torch.complex(real, imag)
    pad = n_fft // 2 + 1 - complex_spec.shape[-2]
    if pad > 0:  # restore the dropped Nyquist bin as zeros
        complex_spec = torch.nn.functional.pad(complex_spec, (0, 0, 0, pad))

    window = torch.hann_window(n_fft, device=spec.device, dtype=real.dtype)
    frames = torch.fft.irfft(complex_spec, n=n_fft, dim=-2) * window.unsqueeze(-1)

    n_frames = frames.shape[-1]
    total = (n_frames - 1) * hop + n_fft
    flat = frames.reshape(-1, n_fft, n_frames)
    out = torch.nn.functional.fold(
        flat, output_size=(1, total), kernel_size=(1, n_fft), stride=(1, hop)
    ).reshape(*frames.shape[:-2], total)
    norm = torch.nn.functional.fold(
        (window**2).reshape(1, n_fft, 1).expand(1, n_fft, n_frames),
        output_size=(1, total),
        kernel_size=(1, n_fft),
        stride=(1, hop),
    ).reshape(total)
    # Dividing by `norm` is the correct normalisation wherever a sample has its
    # full complement of overlapping frames, ripple or not. Only the boundary is
    # unrecoverable: a sample at index i needs every frame f with
    # f*hop <= i < f*hop + n_fft, and for i < n_fft - hop some of those frames
    # would have f < 0 and do not exist. So the first and last `n_fft - hop`
    # samples are zeroed and the rest are normalised exactly.
    #
    # This criterion is POSITIONAL on purpose. Testing `norm` against its own
    # maximum instead looks equivalent and is not: the overlap-add weight is the
    # window **squared** (analysis window times synthesis window), and Hann
    # squared does not satisfy COLA at 50% overlap — it ripples between 0.5 and
    # 1.0. A max-threshold therefore kept only **2.8%** of samples at
    # `local.yaml`'s n_fft 1024 / hop 512, while passing every test in
    # `test_stft.py`, which uses 2048/512 where Hann squared *is* constant
    # at 1.5. Fixed 2026-10-13; see REPRODUCTION.md §4c.
    out = out / norm.clamp(min=1e-8)
    uncovered = n_fft - hop
    if uncovered > 0:
        covered = torch.ones(total, dtype=torch.bool, device=out.device)
        covered[:uncovered] = False
        covered[total - uncovered :] = False
        out = torch.where(covered, out, torch.zeros_like(out))
    return out[..., :length] if length is not None else out


def power(spec: torch.Tensor) -> torch.Tensor:
    """Power spectrum `|X|^2` of a signed-sqrt-scaled spectrogram."""
    real, imag = signed_square(spec[..., 0, :, :]), signed_square(spec[..., 1, :, :])
    return real**2 + imag**2


#: Floor under `sqrt`, whose derivative is infinite at zero. `clamp_min` gives
#: zero gradient below the floor, so it only ever needs to be *small* — and it
#: must be, because anything it clamps is replaced by a constant and stops
#: carrying information.
#:
#: This was 1e-12 and that was far too high. `power` here is the 4th power of
#: the stored signed-sqrt values, so it runs very small: at initialisation the
#: generator emits ~1e-4 and the median power is 1.2e-16, putting **100% of
#: bins** under a 1e-12 floor (77.8% for `faithful.yaml`). Two consequences,
#: neither of which shows up as an error:
#:   * `L_profile` (eq. 3) divides by the difference of two profiles. Both
#:     became the same constant, so the denominator was identically zero and
#:     the term collapsed to `theta*||z_i - z_j||/eps` ~ 1e11 — swamping the
#:     adversarial loss by 15 orders of magnitude.
#:   * The critic's second view (§3.1.1) became a constant image, so one of its
#:     two "independent critics" saw no signal at all.
#: Real audio is affected too, not just initialisation: a true component of
#: 1e-6 has power 1e-12, i.e. exactly at the old floor.
SQRT_FLOOR = 1e-24


def magnitude(spec: torch.Tensor) -> torch.Tensor:
    """Magnitude `|X| = sqrt(P)` — eq. 3's "Hadamard root of the power spectrum".

    Used by `L_profile`. Distinct from `sqrt_magnitude` below: the paper asks for
    different powers of the spectrum in the two places, and conflating them is an
    easy mistake since both are described as a square root of something.
    """
    return torch.sqrt(power(spec).clamp_min(SQRT_FLOOR)).unsqueeze(-3)


def sqrt_magnitude(spec: torch.Tensor) -> torch.Tensor:
    """`sqrt(|X|) = P^(1/4)` — the critic's second view (§3.1.1).

    §3.1.1: D gets "two different views on the input, (1) the signed square-root
    of the complex STFT components and (2) **the square-root of the magnitude
    spectrum** of the generator output (we found empirically that this resulted
    in more stable training than when using the log-magnitude spectrogram)".
    """
    return torch.sqrt(magnitude(spec).clamp_min(SQRT_FLOOR))
