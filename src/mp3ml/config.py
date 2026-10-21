"""Experiment configuration for the GAN reproduction.

Two configs are maintained in `configs/`: `faithful.yaml` reproduces Table 1 of
Lattner & Nistal exactly and is sized for cloud hardware, and `local.yaml` is a
reduced variant that fits an 8 GB card for development. Everything that differs
between them is a field here, so the deviation is explicit rather than a code
branch.

`estimate_size` reports parameters and memory without importing torch, so a
config can be checked against a VRAM budget before committing to a run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import yaml

#: Bytes per parameter for weights + gradients + Adam's two moments, fp32.
ADAM_BYTES_PER_PARAM = 16


@dataclass
class StftConfig:
    """Paper §3.3.1: complex STFT, signed square-root scaled."""

    n_fft: int = 2048
    hop: int = 512
    #: Bins actually fed to the model. The paper uses 1024, i.e. n_fft//2
    #: (dropping the Nyquist bin so the frequency axis is a power of two).
    n_bins: int = 1024
    scaling: str = "signed_sqrt"


@dataclass
class ModelConfig:
    """Table 1. `conv4_out` must equal `latent_ch * latent_freq` (the Reshape1 step)."""

    early_ch: tuple[int, int] = (18, 38)
    #: Conv4 collapses the whole frequency axis with a (n_bins, 1) kernel. This
    #: single layer dominates the parameter count.
    conv4_out: int = 4096
    latent_ch: int = 128
    latent_freq: int = 32
    stack_ch: int = 256
    noise_ch: int = 64
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 1, 2, 4, 8, 16)
    #: §3.1.2 eq. 2: one conv emits twice the maps, split in half; PReLU on the
    #: gated half, sigmoid on the gating half. NOT nn.GLU — the paper states
    #: linear gated units "resulted in unstable training". Conv6-Conv14 only.
    gating: str = "prelu_sigmoid"
    #: [I] PGGAN's **equalized learning rate**: initialise weights to N(0, 1)
    #: and multiply each layer's *output* by He's constant at runtime, instead
    #: of baking that constant into the weights.
    #:
    #: Not in the paper, which says only "He's initialization is performed for
    #: all the layers" — literally true of both. But the authors' own GAN
    #: codebase (`SonyCSLParis/Comparing-Representations-...`, reference [67] of
    #: this paper) defaults to `equalizedlR=True`, and the measurement that
    #: motivates it is stark: with baked-in He, our layers' weight scales span
    #: 35x, so at lr 1e-3 `Conv4` moves **10.2% of its own weight scale per
    #: step** while `Conv1` moves 0.3%. Equalized LR makes that uniform.
    #: See REPRODUCTION.md §1.2.
    equalized_lr: bool = True
    #: [P] Settled by the authors (§6.2): "the two convolutional groups are
    #: indeed grouped convolutions throughout the critic ... joined only in the
    #: final output layer". A `groups=2` convolution has half the weights of an
    #: ungrouped one, hence 0.5. We previously read this as two separate
    #: full-width stacks (2.0), which was wrong — a fourfold error in the
    #: critic's parameter count. See §2.1.
    critic_group_factor: float = 0.5

    def __post_init__(self) -> None:
        # A 3-tap conv at dilation d, padded by d, has both outer taps land on
        # padding for every row unless d < latent_freq. The paper's schedule goes
        # up to 16 on a 32-row axis; shrinking the axis below that silently
        # degenerates those layers to a single tap in frequency. `local.yaml`
        # had exactly this bug with latent_freq=16 until 2026-10-07.
        widest = max(self.dilations)
        if widest >= self.latent_freq:
            raise ValueError(
                f"latent_freq ({self.latent_freq}) must exceed the widest dilation "
                f"({widest}), or those layers see only their centre tap in "
                "frequency and the dilation schedule is wasted"
            )
        if self.latent_ch * self.latent_freq != self.conv4_out:
            raise ValueError(
                f"conv4_out ({self.conv4_out}) must equal latent_ch * latent_freq "
                f"({self.latent_ch} * {self.latent_freq} = "
                f"{self.latent_ch * self.latent_freq}) — Reshape1 depends on it"
            )


@dataclass
class DataConfig:
    root: Path = Path("/home/chris/thesis/data/mp3ml-waveform")
    bitrates: tuple[int, ...] = (16, 32, 64)
    #: 4 s at 44.1 kHz with hop 512 gives 336 frames (paper §3.3).
    segment_frames: int = 336
    #: Stems are training augmentation; evaluate on mixtures only.
    train_stems: tuple[str, ...] = ("mixture", "bass", "drums", "other", "vocals")
    eval_stems: tuple[str, ...] = ("mixture",)
    #: Which on-disk dataset to read. `"waveform"` is this project's own
    #: pre-aligned (clean, degraded) pairs from `scripts/build_waveform_dataset.py`.
    #: `"apollo"` reads the Apollo preprocessed corpus and applies MP3 at
    #: training time, so the two models can be compared on identical audio —
    #: see `mp3ml.apollo`, which also documents why only 64 kbit/s is usable.
    source: str = "waveform"

    def __post_init__(self) -> None:
        if self.source not in ("waveform", "apollo"):
            raise ValueError(f"data.source must be 'waveform' or 'apollo', got {self.source!r}")


@dataclass
class TrainConfig:
    """Paper §3.2 gives iterations, batch size, optimiser, lr, GP and init.

    It does not give Adam betas, the GP weight, the drift-penalty weight, the
    critic:generator ratio, or theta for L_profile. **All of those came from the
    authors by email on 2026-09-29** and are now [P] — see REPRODUCTION.md §1.1.
    Nothing in this dataclass is a free assumption any more except the two
    fidelity switches (`recreate_optimizers`, `gp_reduction`), which exist
    because the authors offered a choice.
    """

    #: §3.2: "trained for 40k iterations and a batch size of 12".
    iterations: int = 40_000
    batch_size: int = 12
    #: §3.2: "the ADAM optimizer with a learning rate of 1e-3".
    lr_g: float = 1e-3
    lr_d: float = 1e-3
    #: [P] From the authors (§6.2), not the paper: "Adam betas: (0, 0.99)".
    betas: tuple[float, float] = (0.0, 0.99)
    #: [P] From the authors: "Critic-to-generator update ratio: 1:1".
    n_critic: int = 1
    #: [P] From the authors: "Gradient-penalty weight: 5". The paper specifies
    #: the penalty (Gulrajani [66]) but never its weight.
    gp_weight: float = 5.0
    #: [P] From the authors: the original code **summed** the gradient penalty
    #: over the batch rather than averaging it, so at batch 12 the effective
    #: weight is ~12x larger. "sum" reproduces that; "mean" is the author's own
    #: suggested improvement for a robust baseline (§4.8).
    gp_reduction: str = "sum"
    #: [I] The authors' own code reduces the **drift** penalty the same way:
    #: `(predRealD[:,-1]**2).sum() * epsilonD`. We averaged it until 2026-10-04,
    #: which at batch 4 makes it 4x weaker than theirs (12x at the paper's batch
    #: 12). The first real run showed `drift` climbing 0.99 -> 809, which is the
    #: drift the term exists to prevent, so this is worth testing directly.
    drift_reduction: str = "sum"
    #: §3.2: "a loss term that penalizes the magnitudes of the output of D for
    #: real input data, preventing the loss from drifting" — the PGGAN drift
    #: term. [P] From the authors: "Drift penalty weight: 0.5". Far above the
    #: 1e-3 PGGAN default we had assumed.
    drift_weight: float = 0.5
    #: §3.2: "He's initialization is performed for all the layers".
    init: str = "he"
    #: Eq. 3 writes a single scalar `theta`, but [P] the authors used **two**:
    #: "the final stochastic 32 kbit/s setup used separate weights rather than a
    #: single shared theta: 1e-5 for the rhythm/time profile and 1e-4 for the
    #: frequency profile". See §2.8 — our measurement had put the balanced range
    #: at 1e-5 to 1e-4, which is exactly where these land.
    profile_theta_freq: float = 1e-4
    profile_theta_rhyt: float = 1e-5
    profile_p_freq: float = 1.3
    profile_p_rhyt: float = 1.6
    #: §3.2.1: "for each conditional input y_i ... we compute two outputs
    #: G(x_k, {z_i, z_j}_k)". G runs twice per item, doubling its cost.
    noise_draws: int = 2
    #: [A] Engineering guard, nothing to do with the paper. A WGAN's gradient
    #: penalty is `(||grad|| - 1)^2` and sits below ~1 when healthy, so a
    #: sustained value this large means the run is gone. Needed because NaN/Inf
    #: checks do not fire on "finite but absurd": a 60-step dry run reached a
    #: gradient penalty of 9.2e10 without tripping anything (§4.6).
    gp_divergence_limit: float = 1e4
    #: Consecutive steps above the limit before aborting. A single spike is
    #: normal — a dry run hit 1643 at step 14 and recovered to 0.004.
    gp_divergence_patience: int = 10
    #: [P] From the authors: "the optimizer objects were recreated for each
    #: update in the original code, so their Adam moment estimates were not
    #: retained across batches". That is not a quirk — a fresh Adam step from
    #: zeroed moments gives `m_hat = g`, `v_hat = g^2`, so the update is
    #: `lr * g / (sqrt(g^2) + eps)`, i.e. **sign-SGD with step size exactly lr**
    #: — measured at exactly 1e-3 for gradients from 1e-3 to 1e6.
    #: This does NOT by itself explain our divergence: retained-state Adam with
    #: beta1=0 is also step-bounded, moving only ~1.4x further on a gradient 1e9
    #: times larger. The likelier cause was our assumed beta1=0.5 momentum (§4.6).
    #: True reproduces it; False is the author's suggested improvement (§4.8).
    #: [I] Exponential moving average of the generator's weights, as the
    #: authors' own GAN code does (`avg_p.mul_(0.999).add_(0.001, p.data)`), and
    #: which it uses for **evaluation output** rather than the live weights.
    #: Not in the paper; not a layer, so Table 1's silence does not rule it out
    #: — the same admissibility argument as equalized learning rate.
    #: EMA generators usually produce smoother, better output, so evaluating
    #: live weights may understate results. 0 disables.
    generator_ema: float = 0.999
    recreate_optimizers: bool = True
    amp: bool = True
    grad_checkpoint: bool = False
    optimizer: str = "adam"


@dataclass
class ExperimentConfig:
    name: str = "unnamed"
    notes: str = ""
    seed: int = 0
    stft: StftConfig = field(default_factory=StftConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_config(path: Path) -> ExperimentConfig:
    """Load a YAML config, filling anything absent from the dataclass defaults."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    sections = {
        "stft": StftConfig,
        "model": ModelConfig,
        "data": DataConfig,
        "train": TrainConfig,
    }
    kwargs: dict = {k: raw[k] for k in ("name", "notes", "seed") if k in raw}
    for key, cls in sections.items():
        payload = dict(raw.get(key) or {})
        for f_name, f_type in cls.__annotations__.items():
            if f_name in payload and "tuple" in str(f_type):
                payload[f_name] = tuple(payload[f_name])
        if key == "data" and "root" in payload:
            payload["root"] = Path(payload["root"])
        kwargs[key] = cls(**payload)
    return ExperimentConfig(**kwargs)


def describe_architecture(cfg: ExperimentConfig) -> list[dict]:
    """Derive every layer's output shape from `cfg`, mirroring Table 1.

    Used by the faithfulness test: with `faithful.yaml` this must reproduce
    Table 1's output sizes exactly, including the 336 -> 212 time shrink caused
    by using no time padding in G (§3.1.1).
    """
    m, st, d = cfg.model, cfg.stft, cfg.data
    e1, e2 = m.early_ch
    f, t = st.n_bins, d.segment_frames
    rows: list[dict] = [{"layer": "Input", "ch": 2, "freq": f, "time": t}]

    # Conv1-3: dilated in both axes, padded, so shapes are preserved (Table 1).
    for i, (co, dil) in enumerate(((e1, 1), (e2, 2), (e2, 4)), 1):
        rows.append({"layer": f"Conv{i}", "ch": co, "freq": f, "time": t, "dilation": dil})

    # Conv4 collapses the whole frequency axis with a (n_bins, 1) kernel.
    rows.append({"layer": "Conv4", "ch": m.conv4_out, "freq": 1, "time": t})
    rows.append({"layer": "Reshape1", "ch": m.latent_ch, "freq": m.latent_freq, "time": t})
    rows.append({"layer": "ReMap", "ch": m.stack_ch, "freq": m.latent_freq, "time": t})

    # Dilated stack. No time padding in G, so each 3x3 conv at dilation `dil`
    # removes 2*dil frames. Conv5 is ungated (§3.1.2 gates Conv6-Conv14).
    for i, dil in enumerate(m.dilations):
        t -= 2 * dil
        name = f"Conv{5 + i}"
        rows.append(
            {"layer": name, "ch": m.stack_ch, "freq": m.latent_freq, "time": t, "dilation": dil}
        )
        if i == 0:
            rows.append(
                {
                    "layer": "NoiseConcat",
                    "ch": m.stack_ch + m.noise_ch,
                    "freq": m.latent_freq,
                    "time": t,
                }
            )
        else:
            rows.append(
                {"layer": f"SelfGating{5 + i}", "ch": m.latent_ch, "freq": m.latent_freq, "time": t}
            )

    rows.append({"layer": "Reshape2", "ch": m.conv4_out, "freq": 1, "time": t})
    for name, co in (("DeConv4", e2), ("DeConv3", e2), ("DeConv2", e1), ("DeConv1", 2)):
        rows.append({"layer": name, "ch": co, "freq": st.n_bins, "time": t})
    rows.append({"layer": "Output", "ch": 2, "freq": st.n_bins, "time": t})
    return rows


def segment_samples(cfg: ExperimentConfig) -> int:
    """Audio samples needed for exactly `segment_frames` STFT frames (center=False).

    The paper calls segments "4-second-long" *and* "336 frames"; at n_fft 2048 /
    hop 512 those differ (4 s is 341 frames). We match the frame count, since the
    architecture's time chain is stated in frames. See REPRODUCTION.md §2.
    """
    return (cfg.data.segment_frames - 1) * cfg.stft.hop + cfg.stft.n_fft


def target_crop(cfg: ExperimentConfig) -> tuple[int, int]:
    """Frame slice of the target matching G's output (§2.6).

    G is unpadded in time, so each dilated 3x3 conv removes `d` frames per side;
    its output aligns with the centre of the input, not the start.
    """
    pad = sum(cfg.model.dilations)
    return pad, pad + output_frames(cfg)


def output_frames(cfg: ExperimentConfig) -> int:
    """Frames surviving G's unpadded dilated stack (§3.1.1): 336 -> 212 in the paper."""
    return cfg.data.segment_frames - sum(2 * d for d in cfg.model.dilations)


def _conv(in_ch: int, out_ch: int, k_freq: int, k_time: int = 1) -> int:
    return in_ch * out_ch * k_freq * k_time + out_ch


def estimate_size(cfg: ExperimentConfig) -> dict:
    """Parameter counts for `cfg`, and a **lower bound** on memory.

    Parameter counts are exact. The memory figures are **not** a prediction:
    `peak_estimate_gb` counts weights, gradients, Adam state and the generator's
    early-layer activations only. It ignores the dilated stack, the critic's two
    branches, the doubled maps inside every self-gated conv, autograd's stored
    intermediates, and cuDNN workspace.

    Measured against reality it under-reports by roughly **5x**: for
    `local.yaml` it gives 2.10 GB at batch 12, where a real critic+generator
    step OOMs an 8 GB card and batch 6 already peaks at 7.50 GB.

    Use `scripts/check_config.py --measure` for an authoritative number; this
    function is for sizing parameters and for a quick "obviously too big" check.
    """
    m, s, d = cfg.model, cfg.stft, cfg.data
    e1, e2 = m.early_ch
    n_bins, frames = s.n_bins, d.segment_frames

    front = _conv(2, e1, 3, 3) + _conv(e1, e2, 3, 3) + _conv(e2, e2, 3, 3)
    collapse = _conv(e2, m.conv4_out, n_bins, 1)
    stack = (
        _conv(m.latent_ch, m.stack_ch, 1, 1)
        + _conv(m.stack_ch, m.stack_ch, 3, 3)
        + _conv(m.stack_ch + m.noise_ch, m.stack_ch, 3, 3)
        + (len(m.dilations) - 2) * _conv(m.latent_ch, m.stack_ch, 3, 3)
    )
    expand = (
        _conv(m.conv4_out, e2, n_bins, 1)
        + _conv(e2, e2, 3, 3)
        + _conv(e2, e1, 3, 3)
        + _conv(e1, 2, 3, 3)
    )
    generator = front + collapse + stack + expand

    # The critic's front end takes 8 channels, not 2 (§2.2): MP3 complex,
    # candidate complex, and 2-channel magnitudes of each.
    critic_front = _conv(8, e1, 3, 3) + _conv(e1, e2, 3, 3) + _conv(e2, e2, 3, 3)
    # Everything up to and including Conv15 is grouped; only Conv16 joins.
    grouped = critic_front + collapse + stack + _conv(m.latent_ch, m.stack_ch, 3, 3)
    critic = int(grouped * m.critic_group_factor) + _conv(m.stack_ch, 1, m.latent_freq, 1)

    total = generator + critic
    act_per_sample = (e1 + 2 * e2) * n_bins * frames * 4
    return {
        "generator_params": generator,
        "critic_params": critic,
        "total_params": total,
        "conv4_params": collapse,
        "weights_gb": total * 4 / 1e9,
        "adam_state_gb": total * ADAM_BYTES_PER_PARAM / 1e9,
        "activations_per_sample_mb": act_per_sample / 1e6,
        "activations_batch_gb": act_per_sample * cfg.train.batch_size / 1e9,
        # Lower bound only — see the docstring. Real peaks run ~5x higher.
        "peak_estimate_gb": (total * ADAM_BYTES_PER_PARAM + act_per_sample * cfg.train.batch_size)
        / 1e9,
    }


def fits_in(cfg: ExperimentConfig, vram_gb: float, headroom: float = 0.9) -> tuple[bool, str]:
    """Whether `cfg` is **obviously** too big for `vram_gb`, with a one-line reason.

    A `True` here does not mean it fits — the underlying estimate is a lower
    bound that under-reports by roughly 5x. Only `--measure` settles it.
    """
    est = estimate_size(cfg)
    budget = vram_gb * headroom
    peak = est["peak_estimate_gb"]
    if cfg.train.amp:
        peak -= est["activations_batch_gb"] * 0.5  # fp16 activations
    verdict = peak <= budget
    return verdict, (
        f"peak ~{peak:.2f} GB vs {budget:.2f} GB usable "
        f"({est['total_params'] / 1e6:.0f} M params, batch {cfg.train.batch_size})"
    )


def single_bitrate(cfg: ExperimentConfig, bitrate: int) -> ExperimentConfig:
    """A copy of `cfg` restricted to one bitrate, renamed to match.

    §4.1 trains "for each compression rate ... a stochastic generator and a
    deterministic generator", so Table 2's rows come from one model per rate.
    The configs carry all three bitrates because the mixed-bitrate model is a
    useful variant in its own right (a real restorer does not know its input's
    bitrate) — but it is *not* what the paper measures. See REPRODUCTION.md
    §2.10 and §4.4.
    """
    if bitrate not in cfg.data.bitrates:
        raise ValueError(f"{bitrate} not in this config's bitrates {list(cfg.data.bitrates)}")
    out = replace(
        cfg,
        name=f"{cfg.name}_{bitrate}k",
        data=replace(cfg.data, bitrates=(bitrate,)),
    )
    return out


def deterministic(cfg: ExperimentConfig) -> ExperimentConfig:
    """A copy of `cfg` with the noise input removed, renamed to match.

    §4.1: "for each compression rate, we train a stochastic generator (with
    noise input z) and **a deterministic generator (without noise input)**".
    Table 2's `det_*` rows come from these, and they are a *separately trained
    network* — not a stochastic model evaluated at z = 0 (REPRODUCTION.md
    §2.10).

    With `noise_ch = 0` the Conv5 concatenation is skipped, so Conv6 takes 256
    inputs rather than 320 — which is exactly what Table 1's `[256]` bracket
    specifies for the critic, and what the generator becomes without noise.
    `L_profile` is then meaningless and is disabled (it would be 0 anyway,
    since its numerator is ||z_i - z_j||).
    """
    return replace(
        cfg,
        name=f"{cfg.name}_det",
        model=replace(cfg.model, noise_ch=0),
    )


def config_to_dict(cfg: ExperimentConfig) -> dict:
    d = asdict(cfg)
    d["data"]["root"] = str(d["data"]["root"])
    return d
