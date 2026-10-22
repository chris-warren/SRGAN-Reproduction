"""Training loop for the GAN reproduction (§3.2, §3.2.1).

Paper-stated (§3.2): 40k iterations, batch size 12, Adam at lr 1e-3, gradient
penalty, a drift penalty on D's output for real data, He initialisation.

Paper-stated (§3.2.1) and easy to miss: **the generator is evaluated twice per
item**. "For each conditional input y_i to G ... we compute two outputs
G(x_k, {z_i, z_j}_k) using randomly sampled {z_i, z_j}_k ~ N(0, I), and use
those outputs to compute Lprofile, **as well as the common gradient update of
G**." So both outputs feed the Wasserstein term as well, not only `L_profile`.

Not stated by the paper, and therefore assumptions (REPRODUCTION.md §2):

* The critic's condition is the degraded input **centre-cropped** to G's output
  extent, so candidate and condition cover the same frames (§2.6, §6.1).
* The critic sees one noise draw, not both; the paper ties the two draws to G's
  update only.
* Critic and generator steps draw separate batches, as in Gulrajani's reference
  implementation.

`n_critic` used to be listed above as an assumption ("5 is the WGAN
convention"). It is **[P]** since the authors' email of 2026-09-29: the ratio is
**1:1**, which is `TrainConfig.n_critic = 1`. Do not "restore" 5.
* Mixed precision (`train.amp`). The paper says nothing about it; it is a
  memory concession for the local config, recorded as a deviation (§4).
"""

from __future__ import annotations

import copy
import json
import logging
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig, config_to_dict, target_crop
from .data import build_train_dataset
from .losses import critic_step_loss, profile_loss, wasserstein_generator_loss
from .models import Critic, Generator
from .tracking import Tracker

logger = logging.getLogger(__name__)


def noise_frames(cfg: ExperimentConfig) -> int:
    """Time extent `z` is broadcast over, once concatenated after Conv5 (Table 1).

    Conv5 is unpadded in time in G, so it has already removed `2 * dilations[0]`
    frames by the point the noise joins. `z` itself carries no time axis — see
    `sample_noise` — but this is the width it is repeated across.
    """
    return cfg.data.segment_frames - 2 * cfg.model.dilations[0]


def _grad_scaler(enabled: bool):
    """A CUDA `GradScaler` across torch versions.

    `torch.amp.GradScaler("cuda", ...)` is the device-first API from torch 2.3;
    before that the spelling is `torch.cuda.amp.GradScaler(...)`. Supporting
    both is three lines and buys a real constraint back: the Apollo replication
    runs in a container pinned to **torch 2.0.0** (torchaudio 2.0.1 needs
    FFmpeg 4 for `apply_codec`), and sharing one image for both models is
    operationally much simpler than maintaining two. This reproduction has no
    such pin of its own -- it calls `lame` and `ffmpeg` as CLIs and never
    imports torchaudio.
    """
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:  # torch 2.0-2.2: no device argument
            pass
    return torch.cuda.amp.GradScaler(enabled=enabled)


def sample_noise(cfg: ExperimentConfig, batch: int, device: torch.device) -> torch.Tensor:
    """`z ~ N(0, I)` as a **per-excerpt vector** `(batch, noise_ch)` (§3.1.1).

    Not a spatial map. §3.1.1: "The noise input z (to the generator G) is simply
    repeated in the two convolutional dimensions and concatenated to layer
    Conv5", and §3.2.1 calls them "input noise vectors". `models._broadcast`
    does the repeating.

    This matters well beyond the model: eq. 3 divides by `||z_i - z_j||`, and a
    64-dim vector gives ~11, where the 64x32x334 map we sampled before gave
    ~1170 — a 100x error in `L_profile`'s scale. See REPRODUCTION.md §1.1.
    """
    return torch.randn(batch, cfg.model.noise_ch, device=device)


def _endless(loader: DataLoader):
    """Yield batches forever — the paper counts iterations, not epochs."""
    while True:
        yield from loader


def non_finite(metrics: dict[str, float]) -> list[str]:
    """Names of any NaN/Inf metrics, sorted (.claude/rules/training.md).

    A diverged WGAN keeps producing numbers, so nothing else notices; the
    gradient penalty in particular goes to inf long before the loop errors.
    """
    return sorted(k for k, v in metrics.items() if not math.isfinite(v))


class Trainer:
    """Wires models, losses and data into the paper's training procedure."""

    def __init__(
        self,
        cfg: ExperimentConfig,
        run_dir: Path,
        device: str = "cuda",
        tracker: Tracker | None = None,
    ) -> None:
        if cfg.train.n_critic < 1:
            raise ValueError(f"n_critic must be >= 1, got {cfg.train.n_critic}")
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        torch.manual_seed(cfg.seed)

        # Device placement lives here and nowhere else (.claude/rules/training.md).
        if self.device.type == "cuda":
            # Autotune convolution algorithms. Every step has identical shapes,
            # which is the exact case this is for, and the model contains one
            # very unusual convolution -- `front.collapse` is
            # `Conv2d(38, 4096, kernel=(1024, 1))`, 53.6 GFLOP/sample. With
            # `benchmark=False` (the default) cuDNN picks an algorithm for it by
            # heuristic, and heuristics are least reliable on shapes like that.
            #
            # Measured 2026-10-22: faithful.yaml ran **71.6 s/step** at batch 4
            # on an H100, where scaling from local.yaml on a much weaker card
            # predicts 3-7 s. Gradient checkpointing was ruled out as the cause
            # by direct A/B (3.49 vs 3.20 s/step, an 8% effect).
            #
            # Caveat: algorithm choice changes floating-point summation order,
            # so runs are not bitwise reproducible across machines. Seeds still
            # fix the data order and the noise draws.
            torch.backends.cudnn.benchmark = True
        self.g = Generator(cfg).to(self.device)
        self.d = Critic(cfg).to(self.device)
        self.opt_g = self._make_opt(self.g, cfg.train.lr_g)
        self.opt_d = self._make_opt(self.d, cfg.train.lr_d)
        self.crop = target_crop(cfg)
        self.step = 0
        self.tracker = tracker
        self.deterministic = cfg.model.noise_ch == 0
        self._gp_over = 0

        # [I] Exponential moving average of G, as the authors' own GAN code
        # keeps — and which it evaluates with, rather than the live weights.
        # `validate()` and `scripts/evaluate.py` use this when it exists.
        self.g_ema: Generator | None = None
        if cfg.train.generator_ema:
            self.g_ema = copy.deepcopy(self.g).eval()
            for param in self.g_ema.parameters():
                param.requires_grad_(False)

        # AMP is off on CPU: autocast there covers a different set of ops and
        # GradScaler is a no-op, so a "working" CPU smoke test would prove nothing.
        self.amp = bool(cfg.train.amp) and self.device.type == "cuda"
        self.scaler_g = _grad_scaler(self.amp)
        self.scaler_d = _grad_scaler(self.amp)

        (self.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.json").write_text(json.dumps(config_to_dict(cfg), indent=2))
        self._write_notes()
        logger.info(
            "run %s | seed %d | device %s | amp %s | %.1f M params",
            cfg.name,
            cfg.seed,
            self.device,
            self.amp,
            sum(p.numel() for p in (*self.g.parameters(), *self.d.parameters())) / 1e6,
        )
        logger.info("train config: %s", json.dumps(asdict(cfg.train)))

    def _make_opt(self, model: torch.nn.Module, lr: float) -> torch.optim.Adam:
        return torch.optim.Adam(model.parameters(), lr=lr, betas=self.cfg.train.betas)

    def _reset_optimizers_if_configured(self) -> None:
        """[P] The authors: "the optimizer objects were recreated for each update
        in the original code, so their Adam moment estimates were not retained
        across batches."

        This is behaviourally significant, not bookkeeping. A fresh Adam step
        from zeroed moments has `m_hat = g` and `v_hat = g^2` after bias
        correction, so the update is `lr * g / (sqrt(g^2) + eps)` — **sign-SGD
        with step size exactly `lr`**, measured at exactly 1e-3 for gradients
        spanning 1e-3 to 1e6.

        Do not over-read that. Retained-state Adam with beta1=0 is also
        step-bounded: the same 1e9-fold gradient increase moves the weight only
        ~1.4x further. So this detail is worth reproducing for fidelity, but it
        is **not** the explanation for the divergence we saw (§4.6).

        Under this mode `betas` barely matter: they only enter through a single
        step's bias correction, which cancels.
        """
        if self.cfg.train.recreate_optimizers:
            self.opt_g = self._make_opt(self.g, self.cfg.train.lr_g)
            self.opt_d = self._make_opt(self.d, self.cfg.train.lr_d)

    def _write_notes(self) -> None:
        """`outputs/<run>/notes.md` (.claude/rules/experiments.md).

        `config.json` says what the settings were; this says what they *mean* —
        the deviations and placeholders a reader of the results needs, written
        at run start so a checkpoint is never orphaned from its caveats.
        """
        cfg, t = self.cfg, self.cfg.train
        caveats = []
        mode = "ORIGINAL behaviour" if t.recreate_optimizers else "modern baseline"
        caveats.append(
            f"- Fidelity mode: **{mode}**. recreate_optimizers={t.recreate_optimizers} "
            f"(True makes Adam behave as sign-SGD), gp_reduction={t.gp_reduction!r}. "
            "See REPRODUCTION.md section 3.3 — these are not comparable to each other."
        )
        if t.gp_reduction == "sum":
            caveats.append(
                f"- The gradient penalty is SUMMED over the batch, so its effective "
                f"weight is ~{t.batch_size}x gp_weight={t.gp_weight} "
                f"(~{t.batch_size * t.gp_weight:g}), as in the original code."
            )
        if self.amp:
            caveats.append("- fp16 autocast on (gradient penalty stays fp32) - section 4.3.")
        if t.grad_checkpoint:
            caveats.append("- gradient checkpointing on; gradient-identical, memory only.")
        if cfg.name != "faithful":
            caveats.append(f"- config `{cfg.name}` is not `faithful.yaml` - section 4.1.")

        (self.run_dir / "notes.md").write_text(
            f"# Run `{cfg.name}`\n\n"
            f"{cfg.notes.strip() or '(no notes set in the config)'}\n\n"
            f"- seed {cfg.seed}, device {self.device}, batch {t.batch_size}, "
            f"n_critic {t.n_critic}, {t.iterations} iterations\n"
            f"- bitrates {list(cfg.data.bitrates)}, {cfg.data.segment_frames} frames in\n\n"
            "## Caveats on any result from this run\n\n"
            + ("\n".join(caveats) if caveats else "- none recorded\n")
            + "\n\n(Written by Trainer at run start. Append findings below.)\n"
        )

    def _autocast(self):
        return torch.autocast("cuda", enabled=self.amp)

    def _condition(self, degraded: torch.Tensor) -> torch.Tensor:
        """The critic's conditioning input, cropped to G's output extent (§6.1)."""
        lo, hi = self.crop
        return degraded[..., lo:hi]

    # ---------------------------------------------------------------- steps

    def critic_step(self, batch: dict) -> dict[str, float]:
        y = batch["degraded"].to(self.device, non_blocking=True)
        real = batch["clean"].to(self.device, non_blocking=True)
        cond = self._condition(y)

        with torch.no_grad(), self._autocast():
            fake = self.g(y, sample_noise(self.cfg, y.shape[0], self.device))
        # The gradient penalty is a double-backward through D; run it in fp32 so
        # the second derivative is not computed on fp16 activations.
        loss, parts = critic_step_loss(
            self.d,
            real.float(),
            fake.float(),
            cond.float(),
            gp_weight=self.cfg.train.gp_weight,
            drift_weight=self.cfg.train.drift_weight,
            gp_reduction=self.cfg.train.gp_reduction,
            drift_reduction=self.cfg.train.drift_reduction,
        )
        # Nearly free, and the clearest read on whether the critic is winning:
        # a WGAN critic is unbounded, so its *scores* are hard to interpret, but
        # how often it ranks real above fake is not. ~0.5 means it cannot tell
        # them apart; a sustained 1.0 means it has won and G gets no gradient.
        with torch.no_grad():
            real_s = self.d(real.float(), cond.float()).mean(dim=-1)
            fake_s = self.d(fake.float(), cond.float()).mean(dim=-1)
            parts["critic_accuracy"] = (real_s > fake_s).float().mean().item()

        self._reset_optimizers_if_configured()
        self.opt_d.zero_grad(set_to_none=True)
        self.scaler_d.scale(loss).backward()
        self.scaler_d.step(self.opt_d)
        self.scaler_d.update()
        return parts

    @torch.no_grad()
    def _batch_quality(self, restored: torch.Tensor, clean: torch.Tensor) -> dict[str, float]:
        """Reference metrics on the **training** batch — how close the output is
        to its target, at training resolution.

        The GAN losses say whether G and D are balanced; these say whether the
        audio is actually improving, which is the thing to watch. Free of extra
        forward passes — both tensors are already in hand — so this runs on the
        logging cadence rather than every step.

        Not comparable to `scripts/evaluate.py`: that scores the *test* split
        with the paper's best-of-20 protocol. These are single-draw, on data the
        model is training on.
        """
        from .metrics import lsd, si_sdr, snr, spectral_mse
        from .stft import istft

        out = {
            "q_lsd": lsd(clean, restored).mean().item(),
            "q_spectral_mse": spectral_mse(clean, restored).mean().item(),
        }
        wav_c = istft(clean, self.cfg.stft.n_fft, self.cfg.stft.hop)
        wav_r = istft(restored, self.cfg.stft.n_fft, self.cfg.stft.hop)
        out["q_snr"] = snr(wav_c, wav_r).mean().item()
        out["q_si_sdr"] = si_sdr(wav_c, wav_r).mean().item()
        return out

    def generator_step(self, batch: dict, with_quality: bool = False) -> dict[str, float]:
        """§3.2.1: two noise draws, both feeding the Wasserstein term and L_profile."""
        y = batch["degraded"].to(self.device, non_blocking=True)
        cond = self._condition(y)
        b = y.shape[0]

        if self.deterministic:
            # §4.1's `det_*` models have no noise input, so the two draws of
            # §3.2.1 would be identical and L_profile is meaningless (its
            # numerator is ||z_i - z_j|| = 0). One pass, adversarial term only.
            with self._autocast():
                out_i = self.g(y, None)
            out_i = out_i.float()
            adversarial = wasserstein_generator_loss(self.d(out_i, cond.float()))
            profile = torch.zeros((), device=self.device)
        else:
            z_i = sample_noise(self.cfg, b, self.device)
            z_j = sample_noise(self.cfg, b, self.device)
            with self._autocast():
                out_i, out_j = self.g(y, z_i), self.g(y, z_j)

            out_i, out_j = out_i.float(), out_j.float()
            adversarial = 0.5 * (
                wasserstein_generator_loss(self.d(out_i, cond.float()))
                + wasserstein_generator_loss(self.d(out_j, cond.float()))
            )
            profile = profile_loss(
                out_i,
                out_j,
                z_i,
                z_j,
                theta_freq=self.cfg.train.profile_theta_freq,
                theta_rhyt=self.cfg.train.profile_theta_rhyt,
                p_freq=self.cfg.train.profile_p_freq,
                p_rhyt=self.cfg.train.profile_p_rhyt,
            )
        loss = adversarial + profile

        self._reset_optimizers_if_configured()
        self.opt_g.zero_grad(set_to_none=True)
        self.scaler_g.scale(loss).backward()
        self.scaler_g.step(self.opt_g)
        self.scaler_g.update()
        self._update_ema()
        stats = {
            "g_loss": loss.item(),
            "adversarial": adversarial.item(),
            "profile": profile.item(),
        }
        if with_quality:
            stats.update(
                self._batch_quality(out_i.detach(), batch["clean"].to(self.device).float())
            )
        return stats

    @torch.no_grad()
    def _update_ema(self) -> None:
        """`avg_p.mul_(decay).add_(1 - decay, p)` — the authors' update, run
        after every generator step. Buffers are copied outright; only
        parameters are averaged."""
        if self.g_ema is None:
            return
        decay = self.cfg.train.generator_ema
        for avg, live in zip(self.g_ema.parameters(), self.g.parameters(), strict=True):
            avg.mul_(decay).add_(live.detach(), alpha=1.0 - decay)
        for avg, live in zip(self.g_ema.buffers(), self.g.buffers(), strict=True):
            avg.copy_(live)

    @property
    def generator_for_eval(self) -> Generator:
        """The network to score. The authors evaluate the **averaged** generator
        (`return self.avgG(input)`), not the live weights."""
        return self.g_ema if self.g_ema is not None else self.g

    def check_divergence(self, metrics: dict[str, float]) -> None:
        """Abort on a sustained, finite, absurd gradient penalty.

        The NaN/Inf guard cannot see a diverged WGAN: a 60-step dry run reached
        a gradient penalty of 9.2e10 without ever producing a non-finite value,
        so nothing tripped and the run kept going (§4.6). A healthy penalty is
        `(||grad|| - 1)^2`, i.e. O(1).

        Patience matters as much as the limit — the same dry run spiked to 1643
        at step 14 and recovered to 0.004 by step 25, so a one-step trigger
        would kill runs that were fine.
        """
        limit = self.cfg.train.gp_divergence_limit
        self._gp_over = self._gp_over + 1 if metrics.get("gradient_penalty", 0.0) > limit else 0
        if self._gp_over >= self.cfg.train.gp_divergence_patience:
            self.save(tag="diverged")
            raise RuntimeError(
                f"gradient penalty above {limit:g} for {self._gp_over} consecutive "
                f"steps (now {metrics['gradient_penalty']:.4g}) — training has "
                "diverged. See REPRODUCTION.md 1.2; checkpoint saved as diverged.pt"
            )

    def train_step(self, batches, with_quality: bool = False) -> dict[str, float]:
        """One generator update preceded by `n_critic` critic updates."""
        for _ in range(self.cfg.train.n_critic):
            d_parts = self.critic_step(next(batches))
        return {**d_parts, **self.generator_step(next(batches), with_quality=with_quality)}

    # ---------------------------------------------------------------- loop

    def validate(self, n_items: int = 8, n_samples: int = 3) -> dict[str, float]:
        """Score a few validation excerpts with the §4.1 protocol, cheaply.

        Training losses say whether the critic and generator are balanced, not
        whether the audio is improving — a WGAN's Wasserstein curve can look
        healthy while the output is noise. These are the metrics that actually
        answer "is it working": LSD, spectral MSE, SNR and SI-SDR against both
        the clean reference and the MP3 baseline.

        Deliberately small (8 excerpts, 3 draws) so it can run often. It is a
        progress signal, **not** a Table 2 number — that needs the full test
        split at 20 draws via `scripts/evaluate.py`.
        """
        from .evaluate import evaluate  # local import: evaluate imports from here

        was_training = self.g.training
        report = evaluate(
            self.generator_for_eval,
            self.cfg,
            split="val",
            n_samples=n_samples,
            batch_size=min(self.cfg.train.batch_size, n_items),
            device=str(self.device),
            max_items=n_items,
            with_peaq=False,
            with_fad=False,
        )
        self.g.train(was_training)
        return report.means()

    def fit(
        self,
        log_every: int = 50,
        checkpoint_every: int = 1000,
        max_steps: int | None = None,
        num_workers: int = 4,
        eval_every: int = 0,
    ) -> None:
        dataset = build_train_dataset(self.cfg)
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=self.device.type == "cuda",
            persistent_workers=num_workers > 0,
        )
        batches = _endless(loader)
        total = max_steps or self.cfg.train.iterations
        logger.info("training to step %d over %d entries", total, len(dataset))

        started, first = time.time(), self.step
        while self.step < total:
            self.step += 1
            metrics = self.train_step(batches, with_quality=self.step % log_every == 0)

            self.check_divergence(metrics)

            # Fail loudly rather than letting a diverged run burn hours.
            bad = non_finite(metrics)
            if bad:
                self.save(tag="diverged")
                raise RuntimeError(
                    f"non-finite loss at step {self.step}: {bad} (checkpoint saved as diverged.pt)"
                )

            if self.tracker is not None:
                self.tracker.log(metrics, step=self.step, phase="train")

            if self.step % log_every == 0:
                rate = (self.step - first) / (time.time() - started)
                logger.info(
                    "step %6d/%d | W %9.4f | gp %8.4f | drift %8.4f | D_acc %.2f | "
                    "lsd %7.3f | snr %7.2f | %.2f it/s",
                    self.step,
                    total,
                    metrics["wasserstein"],
                    metrics["gradient_penalty"],
                    metrics["drift"],
                    metrics.get("critic_accuracy", float("nan")),
                    metrics.get("q_lsd", float("nan")),
                    metrics.get("q_snr", float("nan")),
                    rate,
                )
            if eval_every and self.step % eval_every == 0:
                scores = self.validate()
                if self.tracker is not None:
                    self.tracker.log(scores, step=self.step, phase="val")
                logger.info(
                    "step %6d | val: %s",
                    self.step,
                    "  ".join(f"{k} {v:.4g}" for k, v in sorted(scores.items())),
                )

            if self.step % checkpoint_every == 0:
                self.save()
        self.save(tag="final")
        if self.tracker is not None:
            self.tracker.finish()

    # ---------------------------------------------------------- checkpoints

    def _warn_on_config_drift(self, saved: dict | None) -> None:
        if not saved:
            logger.warning("checkpoint has no config recorded — cannot check for drift")
            return
        was, now = _flatten("", saved), _flatten("", config_to_dict(self.cfg))
        changed = [
            f"{k}: {was[k]!r} -> {now[k]!r}"
            for k in sorted(set(was) & set(now))
            if was[k] != now[k]
        ]
        if changed:
            logger.warning(
                "CONFIG CHANGED since this checkpoint (%d field(s)); resuming anyway:\n  %s",
                len(changed),
                "\n  ".join(changed),
            )

    def save(self, tag: str | None = None) -> Path:
        path = self.run_dir / "checkpoints" / f"{tag or f'step{self.step:06d}'}.pt"
        torch.save(
            {
                "step": self.step,
                "generator": self.g.state_dict(),
                "generator_ema": (self.g_ema.state_dict() if self.g_ema is not None else None),
                "critic": self.d.state_dict(),
                "opt_g": self.opt_g.state_dict(),
                "opt_d": self.opt_d.state_dict(),
                "scaler_g": self.scaler_g.state_dict(),
                "scaler_d": self.scaler_d.state_dict(),
                "config": config_to_dict(self.cfg),
            },
            path,
        )
        logger.info("checkpoint %s", path)
        return path

    def load(self, path: Path) -> None:
        """Resume a run, optimiser and scaler state included.

        Also reports any training hyperparameters that changed since the
        checkpoint was written. Architecture changes fail loudly on the
        `state_dict` load below, but a changed `lr`, `gp_weight` or
        `recreate_optimizers` would be applied **silently** to old weights —
        and on 2026-09-29 every one of those values changed at once, which is
        exactly when a half-resumed run is easiest to create and hardest to
        notice. Changing them deliberately on resume is legitimate, so this
        warns rather than raises.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._warn_on_config_drift(ckpt.get("config"))
        self.g.load_state_dict(ckpt["generator"])
        if self.g_ema is not None and ckpt.get("generator_ema") is not None:
            self.g_ema.load_state_dict(ckpt["generator_ema"])
        self.d.load_state_dict(ckpt["critic"])
        self.opt_g.load_state_dict(ckpt["opt_g"])
        self.opt_d.load_state_dict(ckpt["opt_d"])
        self.scaler_g.load_state_dict(ckpt["scaler_g"])
        self.scaler_d.load_state_dict(ckpt["scaler_d"])
        self.step = ckpt["step"]
        logger.info("resumed %s at step %d", path, self.step)


def _flatten(prefix: str, value) -> dict[str, object]:
    if isinstance(value, dict):
        out: dict[str, object] = {}
        for k, v in value.items():
            out.update(_flatten(f"{prefix}.{k}" if prefix else str(k), v))
        return out
    return {prefix: value}


def latest_checkpoint(run_dir: Path) -> Path | None:
    """Newest numbered checkpoint in `run_dir`, or None if there is none."""
    found = sorted((Path(run_dir) / "checkpoints").glob("step*.pt"))
    return found[-1] if found else None
