"""Checks on the objective metrics (§3.4, eq. 5-7) and the §4.1 protocol."""

from pathlib import Path

import pytest
import torch

from mp3ml.config import load_config
from mp3ml.evaluate import _best
from mp3ml.metrics import (
    HIGHER_IS_BETTER,
    lsd,
    multi_resolution_lsd,
    si_sdr,
    snr,
    spectral_mse,
)
from mp3ml.stft import stft

CFG = load_config(Path(__file__).resolve().parents[1] / "configs" / "local.yaml")


def _spec(wav: torch.Tensor) -> torch.Tensor:
    return stft(wav, n_fft=CFG.stft.n_fft, hop=CFG.stft.hop, n_bins=CFG.stft.n_bins)


@pytest.fixture
def wav() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(2, 40000) * 0.1


# ------------------------------------------------------------ identity cases


def test_identical_signals_score_perfectly(wav: torch.Tensor) -> None:
    """Every metric must bottom out (or top out) on an exact match."""
    spec = _spec(wav)
    assert torch.allclose(lsd(spec, spec), torch.zeros(2), atol=1e-5)
    assert torch.allclose(spectral_mse(spec, spec), torch.zeros(2), atol=1e-9)
    assert (snr(wav, wav) > 100).all()
    assert (si_sdr(wav, wav) > 100).all()
    assert torch.allclose(multi_resolution_lsd(wav, wav), torch.zeros(2), atol=1e-4)


def test_worse_approximations_score_worse(wav: torch.Tensor) -> None:
    torch.manual_seed(1)
    small = wav + torch.randn_like(wav) * 0.01
    large = wav + torch.randn_like(wav) * 0.10
    assert (lsd(_spec(wav), _spec(small)) < lsd(_spec(wav), _spec(large))).all()
    assert (spectral_mse(_spec(wav), _spec(small)) < spectral_mse(_spec(wav), _spec(large))).all()
    assert (snr(wav, small) > snr(wav, large)).all()
    assert (si_sdr(wav, small) > si_sdr(wav, large)).all()


# ------------------------------------------------------- equation structure


def test_snr_matches_equation_7_by_hand() -> None:
    """`10*log10(||s||^2 / ||s - s_hat||^2)`. A reference that is 10x the
    residual in amplitude is 100x in power, i.e. exactly 20 dB."""
    s = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    s_hat = torch.tensor([[9.0, 0.0, 0.0, 0.0]])  # residual 1.0, ratio 100
    assert snr(s, s_hat).item() == pytest.approx(20.0, abs=1e-4)


def test_lsd_reduces_rms_over_frequency_then_mean_over_frames() -> None:
    """Eq. 5's asymmetry: root-mean-square over frequency *inside* the sum,
    then a plain mean over frames. Averaging both axes together, or taking the
    root last, gives a different number — and both are common mistakes.

    Built so the two frames have very different errors; a symmetric reduction
    would give the RMS of the frame values, not their mean.
    """
    clean = torch.zeros(1, 2, 4, 2)
    clean[0, 0] = 1.0  # unit real part everywhere -> power 1
    restored = clean.clone()
    # Frame 0 left alone; frame 1 scaled so its power ratio is a clean 10 dB.
    restored[0, 0, :, 1] = float(10 ** (-10 / 40))  # signed-sqrt domain

    got = lsd(clean, restored).item()
    # frame 0 contributes 0 dB, frame 1 contributes 10 dB -> mean 5.0
    assert got == pytest.approx(5.0, abs=1e-3)

    frames = torch.tensor([0.0, 10.0])
    assert got != pytest.approx(frames.pow(2).mean().sqrt().item(), abs=1e-3)


def test_spectral_mse_uses_sqrt_of_power_not_power(wav: torch.Tensor) -> None:
    """Eq. 6 is MSE between `sqrt(P)`, i.e. magnitudes — not between powers.
    §3.4.3 gives the reason: the log in eq. 5 over-rewards adding high
    frequencies where MP3 has lowpassed to near zero."""
    from mp3ml.stft import power

    a, b = _spec(wav), _spec(wav * 0.5)
    want = ((power(a).sqrt() - power(b).sqrt()) ** 2).mean(dim=(-2, -1))
    assert torch.allclose(spectral_mse(a, b), want, atol=1e-9)

    wrong = ((power(a) - power(b)) ** 2).mean(dim=(-2, -1))
    assert not torch.allclose(spectral_mse(a, b), wrong)


# ------------------------------------------------------------- si-sdr vs snr


def test_si_sdr_ignores_gain_but_snr_does_not(wav: torch.Tensor) -> None:
    """The point of carrying both. The paper's objective has no reconstruction
    loss, so nothing pins G's output level; a quiet-but-correct restoration is
    heavily punished by eq. 7 and not at all by SI-SDR."""
    quiet = wav * 0.5
    assert (si_sdr(wav, quiet) > 100).all(), "SI-SDR must be gain-invariant"
    assert (snr(wav, quiet) < 15).all(), "SNR must see the gain error"


# ------------------------------------------------------- best-of-N protocol


def test_best_of_n_respects_each_metric_direction() -> None:
    """§4.1 takes the best of 20 draws *per metric*. Lower wins for LSD/MSE,
    higher for SNR/ODG/DI — getting a direction wrong silently reports the
    worst draw as the result."""
    stacked = torch.tensor([[1.0, 9.0], [5.0, 2.0], [3.0, 7.0]])  # (draws, items)
    assert torch.equal(_best(stacked, "lsd"), torch.tensor([1.0, 2.0]))
    assert torch.equal(_best(stacked, "snr"), torch.tensor([5.0, 9.0]))
    assert torch.equal(_best(stacked, "spectral_mse"), torch.tensor([1.0, 2.0]))
    assert torch.equal(_best(stacked, "odg"), torch.tensor([5.0, 9.0]))


def test_every_metric_declares_a_direction() -> None:
    """A metric missing from HIGHER_IS_BETTER would crash `_best` mid-run, or
    worse, be silently dropped from a report."""
    from mp3ml import evaluate as ev

    produced = {"lsd", "spectral_mse", "snr", "si_sdr", "multi_resolution_lsd", "odg", "di"}
    assert produced <= set(HIGHER_IS_BETTER)
    assert set(ev.EvalReport().means()) == set()


def test_paper_metrics_are_not_silently_replaced_by_modern_ones() -> None:
    """The paper's three equations must stay reportable on their own. If a
    refactor folded `lsd` into `multi_resolution_lsd`, Table 2 would stop being
    comparable."""
    import inspect

    from mp3ml import metrics

    for name in ("lsd", "spectral_mse", "snr"):
        assert callable(getattr(metrics, name))
    source = inspect.getsource(metrics.lsd)
    assert "n_ffts" not in source, "eq. 5 must stay single-resolution"
