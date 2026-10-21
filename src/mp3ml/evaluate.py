"""Evaluation harness implementing the paper's §4.1 protocol.

**The protocol is not "run the model once and measure".** §4.1:

> "for any input y taken from the test set, we sample 20 times with the
> corresponding generator using z_i ~ N(0, I), and for each objective metric,
> we take the best value of that set. Note that all objective metrics are
> computed by comparing the restored data with the original versions.
> Therefore, when picking samples to optimize a specific metric, we do not pick
> the sample with the best 'quality', but rather the restoration that best
> approximates the original."

Three consequences, all easy to get wrong:

1. **20 draws per input**, not one. A single-draw number is not comparable to
   Table 2 and will look far worse.
2. **Best is taken per metric independently** — the sample that wins on SNR
   need not be the one that wins on LSD. So the reported row is generally not
   achievable by any single output. That is the paper's choice, and reproducing
   it means reproducing that too.
3. **Direction matters**: higher is better for ODG/DI/SNR, lower for LSD/MSE
   (§4.1). `metrics.HIGHER_IS_BETTER` is the single source of truth.

The MP3 input is scored against the same reference, giving the `mp3_*` baseline
rows of Table 2.

**Deterministic models.** The paper's `det_*` rows come from a *separately
trained* generator "without noise input" — not from a stochastic model
evaluated at z = 0. `evaluate` handles a genuine deterministic model (one whose
config sets `noise_ch: 0`) by taking a single draw, but producing the paper's
comparison needs a second training run. See REPRODUCTION.md §1.3.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader

from . import peaq
from .config import ExperimentConfig, target_crop
from .data import WaveformPairs
from .metrics import (
    HIGHER_IS_BETTER,
    lsd,
    multi_resolution_lsd,
    si_sdr,
    snr,
    spectral_mse,
)
from .stft import istft

logger = logging.getLogger(__name__)

#: §4.1: "we sample 20 times with the corresponding generator".
PAPER_N_SAMPLES = 20


@dataclass
class EvalReport:
    """Per-item values plus their means. Keeps per-item data so the paper's
    Figure 4 violin plots stay reproducible, and so a logger can show spread."""

    model: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    baseline: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    n_items: int = 0
    n_samples: int = 0
    skipped: list[str] = field(default_factory=list)
    #: Set-level metrics (FAD). Not per-item, so they live apart from the
    #: best-of-N table — see `mp3ml.fad`.
    set_level: dict[str, float] = field(default_factory=dict)

    def means(self) -> dict[str, float]:
        out = {f"model/{k}": sum(v) / len(v) for k, v in self.model.items() if v}
        out.update({f"mp3/{k}": sum(v) / len(v) for k, v in self.baseline.items() if v})
        out.update(self.set_level)
        return out

    def table(self) -> str:
        rows = sorted({*self.model, *self.baseline})
        width = max((len(r) for r in rows), default=6)
        lines = [f"{'metric':<{width}}  {'mp3':>10}  {'model':>10}  {'better':>7}"]
        lines.append("-" * len(lines[0]))
        for name in rows:
            mp3 = self.baseline.get(name) or []
            mdl = self.model.get(name) or []
            fmt = lambda v: f"{sum(v) / len(v):>10.3f}" if v else f"{'-':>10}"  # noqa: E731
            arrow = "higher" if HIGHER_IS_BETTER.get(name, False) else "lower"
            lines.append(f"{name:<{width}}  {fmt(mp3)}  {fmt(mdl)}  {arrow:>7}")
        return "\n".join(lines)


def _spectral_metrics(clean: torch.Tensor, restored: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "lsd": lsd(clean, restored),
        "spectral_mse": spectral_mse(clean, restored),
    }


def _waveform_metrics(clean: torch.Tensor, restored: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "snr": snr(clean, restored),
        "si_sdr": si_sdr(clean, restored),
        "multi_resolution_lsd": multi_resolution_lsd(clean, restored),
    }


def _to_wav(spec: torch.Tensor, cfg: ExperimentConfig) -> torch.Tensor:
    return istft(spec, cfg.stft.n_fft, cfg.stft.hop)


def score_pair(
    clean_spec: torch.Tensor,
    restored_spec: torch.Tensor,
    cfg: ExperimentConfig,
    with_peaq: bool = False,
    samplerate: int = 44100,
) -> dict[str, torch.Tensor]:
    """Every metric for one batch of (clean, restored) spectrogram pairs.

    Returns per-item tensors of shape `(batch,)`, so the caller can pick the
    best draw per item without re-running anything.
    """
    clean_wav, restored_wav = _to_wav(clean_spec, cfg), _to_wav(restored_spec, cfg)
    out = _spectral_metrics(clean_spec, restored_spec)
    out.update(_waveform_metrics(clean_wav, restored_wav))
    if with_peaq:
        odg, di = [], []
        for c, r in zip(clean_wav.cpu().numpy(), restored_wav.cpu().numpy(), strict=True):
            a, b = peaq.odg_di(c, r, samplerate)
            odg.append(a)
            di.append(b)
        out["odg"] = torch.tensor(odg)
        out["di"] = torch.tensor(di)
    return out


def _best(stacked: torch.Tensor, name: str) -> torch.Tensor:
    """Best value per item across the sample axis (dim 0), per §4.1."""
    return stacked.max(dim=0).values if HIGHER_IS_BETTER[name] else stacked.min(dim=0).values


@torch.no_grad()
def evaluate(
    generator: torch.nn.Module,
    cfg: ExperimentConfig,
    split: str = "test",
    n_samples: int = PAPER_N_SAMPLES,
    batch_size: int = 4,
    device: str = "cuda",
    max_items: int | None = None,
    with_peaq: bool | None = None,
    with_fad: bool | None = None,
    stems: tuple[str, ...] | None = None,
    seed: int | None = 0,
) -> EvalReport:
    """Score `generator` on `split` using the §4.1 best-of-N protocol.

    `stems` defaults to `cfg.data.eval_stems` (mixtures only) — stems are
    training augmentation, and evaluating on them would not be comparable.
    """
    from . import fad
    from .train import sample_noise  # local import: avoids a cycle

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    # Fixed crops make the *audio* reproducible; this makes the **noise draws**
    # reproducible too, so a reported number can be reproduced exactly. The
    # protocol is unchanged — still N random draws per input (§4.1), just
    # reproducibly random. Pass seed=None for independent sampling.
    if seed is not None:
        torch.manual_seed(seed)
    generator = generator.to(dev).eval()
    report = EvalReport(n_samples=n_samples)

    if with_fad is None:
        with_fad = False  # opt-in: the first call downloads VGGish weights
    if with_fad and not fad.available():
        report.skipped.append("fad — VGGish could not be loaded (needs network on first use)")
        with_fad = False
    fad_pools: dict[str, list[torch.Tensor]] = defaultdict(list)

    if with_peaq is None:
        with_peaq = peaq.available()
    if not with_peaq:
        report.skipped.append(
            "odg/di — peaqb not on PATH; the paper's ODG and DI columns cannot be "
            "reproduced without it (see mp3ml.peaq for the build)"
        )
        logger.warning("PEAQ unavailable: ODG and DI will be missing from this report")

    deterministic = getattr(cfg.model, "noise_ch", 0) == 0
    if deterministic and n_samples != 1:
        logger.info("deterministic generator (noise_ch=0): using 1 draw, not %d", n_samples)
        n_samples = 1
        report.n_samples = 1

    # random_crop=False: a benchmark must score the same audio every run, or
    # two checkpoints cannot be compared and no number is reproducible.
    dataset = WaveformPairs(cfg, split=split, stems=stems or cfg.data.eval_stems, random_crop=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    lo, hi = target_crop(cfg)

    for batch in loader:
        if max_items is not None and report.n_items >= max_items:
            break
        y = batch["degraded"].to(dev)
        clean = batch["clean"].to(dev)

        # Baseline: the MP3 itself, cropped to the same window the model outputs.
        for name, value in score_pair(clean, y[..., lo:hi], cfg, with_peaq).items():
            report.baseline[name].extend(value.tolist())

        per_sample: dict[str, list[torch.Tensor]] = defaultdict(list)
        for draw in range(n_samples):
            z = None if deterministic else sample_noise(cfg, y.shape[0], dev)
            restored = generator(y, z)
            for name, value in score_pair(clean, restored, cfg, with_peaq).items():
                per_sample[name].append(value.cpu())
            # FAD pools one draw only: it compares distributions, so feeding it
            # 20 draws per input would inflate the generated set with
            # near-duplicates and bias the covariance.
            if with_fad and draw == 0:
                fad_pools["model"].append(_to_wav(restored, cfg).cpu())

        if with_fad:
            fad_pools["clean"].append(_to_wav(clean, cfg).cpu())
            fad_pools["mp3"].append(_to_wav(y[..., lo:hi], cfg).cpu())

        for name, values in per_sample.items():
            report.model[name].extend(_best(torch.stack(values), name).tolist())
        report.n_items += y.shape[0]

    if with_fad and fad_pools["clean"]:
        report.set_level.update(_fad_scores(fad_pools, report))

    logger.info("evaluated %d items x %d draws on %s", report.n_items, n_samples, split)
    return report


def _fad_scores(pools: dict[str, list[torch.Tensor]], report: EvalReport) -> dict[str, float]:
    """FAD of the model and of the MP3 baseline, both against the clean set.

    Reported as a pair on purpose: an absolute FAD means little, but
    `fad/model` below `fad/mp3` says the restoration moved the *distribution*
    toward real music, which is the claim the per-item metrics cannot make.
    """
    from . import fad
    from .metrics import frechet_distance

    # The whole computation is best-effort, not just the embedding step. FAD
    # needs at least two samples per set and VGGish emits roughly one frame per
    # 0.96 s, so a short config or a small `max_items` leaves fewer rows than
    # `frechet_distance` accepts -- and it raises. That used to happen *outside*
    # this try, discarding every per-item metric already accumulated by the
    # caller rather than degrading to a report without FAD (code review,
    # 2026-10-21).
    try:
        emb = {k: fad.embed(torch.cat(v, dim=0)) for k, v in pools.items()}
        return {
            "fad/model": float(frechet_distance(emb["clean"], emb["model"])),
            "fad/mp3": float(frechet_distance(emb["clean"], emb["mp3"])),
        }
    except Exception as exc:  # noqa: BLE001 - FAD is an extra, never the report
        logger.warning("FAD skipped: %s", exc)
        report.skipped.append(f"fad — {exc}")
        return {}
