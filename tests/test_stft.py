"""Checks on the paper's spectrogram representation (§3.3.1, eq. 4)."""

import pytest
import torch

from mp3ml.stft import (
    istft,
    magnitude,
    power,
    signed_sqrt,
    signed_square,
    sqrt_magnitude,
    stft,
)

N_FFT, HOP, SEGMENT = 2048, 512, 173_568


def test_signed_sqrt_is_invertible_and_sign_preserving() -> None:
    x = torch.tensor([-4.0, -0.25, 0.0, 0.25, 4.0])
    y = signed_sqrt(x)
    assert torch.allclose(y, torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0]))
    assert torch.allclose(signed_square(y), x, atol=1e-6)


def test_signed_sqrt_applies_per_component_not_to_magnitude() -> None:
    """Eq. 4 scales real and imaginary parts separately: sigma(a) + i*sigma(b)."""
    wav = torch.randn(SEGMENT) * 0.1
    spec = stft(wav, N_FFT, HOP, n_bins=1025)
    raw = torch.stft(
        wav, N_FFT, HOP, window=torch.hann_window(N_FFT), center=False, return_complex=True
    )
    assert torch.allclose(spec[0], signed_sqrt(raw.real), atol=1e-5)
    assert torch.allclose(spec[1], signed_sqrt(raw.imag), atol=1e-5)


def test_shape_matches_table_1() -> None:
    spec = stft(torch.randn(SEGMENT), N_FFT, HOP, n_bins=1024)
    assert spec.shape == (2, 1024, 336)


def test_overlap_add_is_exact_when_no_bins_are_dropped() -> None:
    """Isolates our OLA from the deliberate Nyquist drop — it must be exact."""
    torch.manual_seed(0)
    wav = torch.randn(SEGMENT) * 0.1
    back = istft(stft(wav, N_FFT, HOP, n_bins=1025), N_FFT, HOP, length=SEGMENT)
    interior = slice(N_FFT, SEGMENT - N_FFT)
    err = (back[interior] - wav[interior]).norm() / wav[interior].norm()
    assert err < 1e-5, f"overlap-add is lossy: {err:.2e}"


def test_dropping_nyquist_costs_only_that_bin() -> None:
    """The 1024-bin round trip loses exactly the Nyquist energy, nothing more."""
    torch.manual_seed(0)
    wav = torch.randn(SEGMENT) * 0.1
    back = istft(stft(wav, N_FFT, HOP, n_bins=1024), N_FFT, HOP, length=SEGMENT)
    interior = slice(N_FFT, SEGMENT - N_FFT)
    err = (back[interior] - wav[interior]).norm() / wav[interior].norm()
    raw = torch.stft(
        wav, N_FFT, HOP, window=torch.hann_window(N_FFT), center=False, return_complex=True
    )
    expected = (raw[1024].abs().pow(2).sum() / raw.abs().pow(2).sum()).sqrt()
    assert err == pytest.approx(expected.item(), rel=0.6)


def test_istft_zeros_samples_without_full_window_coverage() -> None:
    """The boundary must be zero, never a best effort.

    With `center=False` the first and last `n_fft - hop` samples are reached by
    fewer frames than the interior, and the overlap-add window sum there falls
    to exactly zero at the endpoints. Dividing by it with a small clamp — which
    this function did until 2026-10-07 — turns those samples into a gain of up
    to 1e8 (REPRODUCTION.md §4c).
    """
    torch.manual_seed(0)
    # BOTH overlap ratios the configs use. 2048/512 is 75% overlap, where the
    # squared window sums to a constant; 1024/512 is 50%, where it does NOT and
    # ripples 0.5-1.0. A magnitude-based coverage test passes the first and
    # keeps only 2.8% of the signal on the second — `local.yaml`'s setting.
    for n_fft, hop in ((2048, 512), (1024, 512)):
        samples = 200 * hop + n_fft
        spec = stft(torch.randn(samples) * 0.1, n_fft, hop, n_bins=n_fft // 2 + 1)
        back = istft(spec, n_fft, hop)
        uncovered = n_fft - hop
        label = f"n_fft={n_fft} hop={hop}"

        assert torch.all(back[:uncovered] == 0), f"{label}: leading boundary not zeroed"
        assert torch.all(back[-uncovered:] == 0), f"{label}: trailing boundary not zeroed"
        interior = back[uncovered:-uncovered]
        assert torch.all(interior != 0), f"{label}: zeroed inside the covered region"
        zeroed = 2 * uncovered / back.shape[-1]
        assert zeroed < 0.05, f"{label}: zeroed {100 * zeroed:.1f}% of the signal"


def test_waveform_error_tracks_spectral_error() -> None:
    """Waveform SNR after `istft` may beat the spectral SNR, but never trail it.

    Overlap-add is a least-squares pseudo-inverse: an arbitrary perturbation of
    a spectrogram is generally *not* the STFT of any signal, so the component
    orthogonal to the operator's range is discarded on inversion and the
    waveform error comes out **smaller** than the spectral error. The
    inequality is therefore one-sided, and that is the whole point — error
    cannot be manufactured by inverting.

    Before the boundary fix the waveform SNR trailed the spectral SNR by
    **32 dB**, which is impossible for a pseudo-inverse and was the signature
    of exploded edge samples. Every waveform metric was dominated by them while
    the spectral metrics could not see them, which made a working generator look
    catastrophic for two days (REPRODUCTION.md §4c). Asserted here rather than
    left to a probe script.
    """
    torch.manual_seed(0)
    wav = torch.randn(SEGMENT) * 0.1
    ref_spec = stft(wav, N_FFT, HOP, n_bins=1025)
    ref_wav = istft(ref_spec, N_FFT, HOP)
    for eps in (0.1, 0.03, 0.01):
        est_spec = ref_spec + eps * torch.randn_like(ref_spec)
        a, b = _as_complex(est_spec), _as_complex(ref_spec)
        spec_snr = 10 * torch.log10(b.abs().pow(2).sum() / (a - b).abs().pow(2).sum())
        est_wav = istft(est_spec, N_FFT, HOP)
        err = est_wav - ref_wav
        wav_snr = 10 * torch.log10(ref_wav.pow(2).sum() / err.pow(2).sum())
        assert wav_snr > spec_snr - 1.0, (
            f"eps={eps}: waveform SNR {wav_snr:.2f} dB trails spectral SNR "
            f"{spec_snr:.2f} dB — inversion cannot add error, so the istft "
            "boundary is leaking"
        )


def _as_complex(spec: torch.Tensor) -> torch.Tensor:
    return torch.complex(signed_square(spec[..., 0, :, :]), signed_square(spec[..., 1, :, :]))


def test_magnitude_and_sqrt_magnitude_are_different_transforms() -> None:
    """Eq. 3 wants sqrt(power) = |X|; §3.1.1 wants sqrt(|X|) = P^(1/4).

    Both are described as "a square root of" something, so conflating them is
    easy — and would silently change both L_profile and the critic's second view.
    """
    spec = stft(torch.randn(SEGMENT) * 0.1, N_FFT, HOP, n_bins=1024)
    p, mag, smag = power(spec), magnitude(spec), sqrt_magnitude(spec)
    assert mag.shape == smag.shape == (1, 1024, 336)
    assert (mag >= 0).all() and (smag >= 0).all()
    assert torch.allclose(mag.squeeze(-3), p.sqrt(), atol=1e-5)
    assert torch.allclose(smag, mag.sqrt(), atol=1e-5)
    assert not torch.allclose(mag, smag)


def test_sqrt_floor_does_not_swallow_quiet_signal() -> None:
    """The floor under `sqrt` must be small enough to stay out of the way.

    It was 1e-12, which is the power of a true component of 1e-6 — ordinary for
    a quiet bin, and *below* the whole output of an untrained generator (median
    power 1.2e-16). Everything under the floor becomes the same constant, so
    `L_profile`'s denominator went to exactly zero and the critic's magnitude
    view became a constant image. Neither raises.
    """
    from mp3ml.stft import SQRT_FLOOR

    quiet = torch.full((1, 2, 8, 4), 1e-3)  # true component 1e-6, power 2e-12
    assert (power(quiet) > SQRT_FLOOR).all(), "floor reaches ordinary quiet bins"

    # Two faintly different quiet spectra must stay distinguishable.
    a = torch.full((1, 2, 8, 4), 1e-4)
    b = a * 1.5
    assert not torch.allclose(magnitude(a), magnitude(b))
    assert not torch.allclose(sqrt_magnitude(a), sqrt_magnitude(b))


def test_magnitude_gradient_is_finite_at_zero() -> None:
    """Why the floor exists at all: sqrt'(0) is infinite."""
    spec = torch.zeros(1, 2, 4, 4, requires_grad=True)
    magnitude(spec).sum().backward()
    assert torch.isfinite(spec.grad).all()


def test_batched_input_is_supported() -> None:
    spec = stft(torch.randn(3, SEGMENT), N_FFT, HOP, n_bins=1024)
    assert spec.shape == (3, 2, 1024, 336)
    assert istft(spec, N_FFT, HOP, length=SEGMENT).shape == (3, SEGMENT)
