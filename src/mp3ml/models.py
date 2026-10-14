"""Generator and critic, wired from Table 1 of Lattner & Nistal.

Details that are easy to get wrong, each with its source:

* **Self-gating is not `nn.GLU`.** Eq. 2 is `PReLU(x*W + a) · σ(x*V + b)`; §3.1.2
  says linear gated units "resulted in unstable training". Implemented as one
  convolution emitting twice the gated width, split in half.
* **Gating covers Conv6–Conv14 only** (§3.1.2). Conv5 is ungated, which is why
  Table 1 shows it at 256 maps and the noise concat reaching 320.
* **G is unpadded in time, D is padded** — Table 1's padding column reads
  `freq, (G_time)[D_time]`, e.g. `1, (0)[1]`. So G shrinks 336 → 212 while D
  holds 212 throughout.
* **Frequency is padded in both**, by `dilation`, so the 32-bin latent axis is
  preserved through the dilated stack.
* **Conv4 collapses frequency** with a `(n_bins, 1)` kernel (§3.1.3); Reshape1
  splits its 4096 responses into 32 groups of 128 (Fig. 2). DeConv4 inverts it.
* **He initialisation** throughout (§3.2).

The critic's conditioning input is our documented assumption, not the paper's —
see REPRODUCTION.md §1.1 and §6.1.
"""

from __future__ import annotations

import math

import torch
import torch.utils.checkpoint as cp
from torch import nn

from .config import ExperimentConfig
from .stft import sqrt_magnitude


def _broadcast(noise: torch.Tensor, like: torch.Size) -> torch.Tensor:
    """Repeat a per-excerpt noise vector `(B, C)` over frequency and time.

    §3.1.1: the noise is "simply repeated in the two convolutional dimensions".
    `expand` does this without allocating, and autograd sums the gradients back
    onto the vector — which is what makes `z` a global style control rather
    than per-cell dither.
    """
    if noise.ndim == 2:
        noise = noise[:, :, None, None]
    return noise.expand(noise.shape[0], noise.shape[1], like[2], like[3])


def he_init(module: nn.Module) -> None:
    """§3.2: "He's initialization is performed for all the layers"."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="leaky_relu", a=0.25)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def equalized_lr_init(module: nn.Module) -> None:
    """PGGAN's equalized learning rate, as the authors' own GAN code does it.

    Weights start at N(0, 1) and the layer's **output** is scaled by He's
    constant at runtime, so the optimiser sees unit-scale weights everywhere.
    That matters because a fixed learning rate otherwise perturbs each layer by
    a different *relative* amount: measured on `local.yaml`, baked-in He gives a
    35x spread, with Conv4 moving 10.2% of its own weight scale in one step at
    lr 1e-3 while Conv1 moves 0.3%.

    Still "He's initialization" in the paper's sense (§3.2) — the same constant,
    applied at a different point. See REPRODUCTION.md §1.2.
    """
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        # prod(weight.shape[1:]) matches the authors' getLayerNormalizationFactor.
        fan_in = int(torch.tensor(module.weight.shape[1:]).prod().item())
        scale = math.sqrt(2.0 / fan_in)
        nn.init.normal_(module.weight, 0.0, 1.0)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        # Scales the bias too, exactly as the reference implementation does.
        module.register_forward_hook(lambda _m, _i, out: out * scale)


def init_weights(model: nn.Module, cfg: ExperimentConfig) -> None:
    model.apply(equalized_lr_init if cfg.model.equalized_lr else he_init)


class SelfGatedConv2d(nn.Module):
    """Eq. 2: `PReLU(x*W + a) · sigmoid(x*V + b)`.

    One convolution emits `2 * out_ch` maps; the halves are the gated and gating
    branches (§3.1.2: "we use only one layer with twice as many output maps as
    input maps, which are then used for self-gating").
    """

    def __init__(
        self, in_ch: int, out_ch: int, dilation: int, pad_time: bool, groups: int = 1
    ) -> None:
        super().__init__()
        self.groups = groups
        self.conv = nn.Conv2d(
            in_ch,
            2 * out_ch,
            kernel_size=3,
            dilation=dilation,
            padding=(dilation, dilation if pad_time else 0),
            groups=groups,
        )
        self.act = nn.PReLU(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.groups == 1:
            gated, gate = y.chunk(2, dim=1)
        else:
            # A grouped conv lays its output out as [group0 maps, group1 maps].
            # A plain chunk(2) would therefore split *between* the groups, making
            # group 1 the gate for group 0 — mixing the two independent critics
            # that groups=2 exists to keep apart. Split inside each group instead.
            b, c, f, t = y.shape
            per_group = y.view(b, self.groups, c // self.groups, f, t)
            gated, gate = per_group.chunk(2, dim=2)
            gated = gated.reshape(b, -1, f, t)
            gate = gate.reshape(b, -1, f, t)
        return self.act(gated) * torch.sigmoid(gate)


class FrequencyAggregation(nn.Module):
    """Conv1–Conv4 plus Reshape1 — the front end shared by G and D (§3.1.3)."""

    def __init__(self, cfg: ExperimentConfig, in_ch: int, groups: int = 1) -> None:
        super().__init__()
        e1, e2 = cfg.model.early_ch
        # Conv1-3: padded in BOTH axes (Table 1: "1,1" / "2,2" / "4,4").
        self.early = nn.Sequential(
            nn.Conv2d(in_ch, e1, 3, dilation=1, padding=1, groups=groups),
            nn.PReLU(e1),
            nn.Conv2d(e1, e2, 3, dilation=2, padding=2, groups=groups),
            nn.PReLU(e2),
            nn.Conv2d(e2, e2, 3, dilation=4, padding=4, groups=groups),
            nn.PReLU(e2),
        )
        # Conv4: one kernel spanning the whole frequency axis, convolving in time.
        self.collapse = nn.Conv2d(e2, cfg.model.conv4_out, (cfg.stft.n_bins, 1), groups=groups)
        self.collapse_act = nn.PReLU(cfg.model.conv4_out)
        self.latent_ch, self.latent_freq = cfg.model.latent_ch, cfg.model.latent_freq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.collapse_act(self.collapse(self.early(x)))  # (B, 4096, 1, T)
        b, _, _, t = x.shape
        return x.reshape(b, self.latent_ch, self.latent_freq, t)  # Reshape1


class DilatedStack(nn.Module):
    """Conv5–Conv14. Conv5 is ungated; Conv6 onward are self-gated (§3.1.2).

    `checkpoint` recomputes the gated layers' activations during the backward
    pass instead of storing them. It is mathematically an identity — gradients
    are unchanged, which `tests/test_train.py` asserts — and exists only so the
    generator's two forward passes (§3.2.1) fit an 8 GB card.
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        pad_time: bool,
        noise_ch: int = 0,
        checkpoint: bool = False,
        skip: bool = False,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.checkpoint = checkpoint
        self.skip = skip
        m = cfg.model
        dil = m.dilations
        self.remap = nn.Conv2d(m.latent_ch, m.stack_ch, 1, groups=groups)
        self.remap_act = nn.PReLU(m.stack_ch)
        # Conv5: plain conv, no gating, so it keeps stack_ch maps.
        self.conv5 = nn.Conv2d(
            m.stack_ch,
            m.stack_ch,
            3,
            dilation=dil[0],
            padding=(dil[0], dil[0] if pad_time else 0),
            groups=groups,
        )
        self.conv5_act = nn.PReLU(m.stack_ch)
        self.noise_ch = noise_ch
        gated_in = [m.stack_ch + noise_ch] + [m.latent_ch] * (len(dil) - 2)
        self.gated = nn.ModuleList(
            SelfGatedConv2d(ci, m.latent_ch, d, pad_time, groups=groups)
            for ci, d in zip(gated_in, dil[1:], strict=True)
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        x = self.conv5_act(self.conv5(self.remap_act(self.remap(x))))
        if self.noise_ch:
            if noise is None:
                noise = torch.randn(x.shape[0], self.noise_ch, device=x.device, dtype=x.dtype)
            # §3.1.1: "The noise input z (to the generator G) is **simply repeated
            # in the two convolutional dimensions** and concatenated to layer
            # Conv5." So z is a per-excerpt *vector*, broadcast over frequency and
            # time — not an independent value per cell. §3.2.1 agrees, calling
            # them "input noise vectors {z_i, z_j}".
            x = torch.cat([x, _broadcast(noise, x.shape)], dim=1)
        for layer in self.gated:
            if self.checkpoint and torch.is_grad_enabled():
                # use_reentrant=False is required: the critic's gradient penalty
                # back-propagates twice through this graph.
                out = cp.checkpoint(layer, x, use_reentrant=False)
            else:
                out = layer(x)
            # §3.1.1: "skip connections for D in the convolutional layers
            # Conv6 - Conv14". Additive, because Table 1's In Maps for Conv7-14
            # stay at 128 — a concatenating skip would double them to 256.
            # Conv6 is skipped automatically: its input is stack_ch (256) while
            # its output is latent_ch (128), so there is nothing to add.
            x = out + x if (self.skip and out.shape == x.shape) else out
        return x


class Generator(nn.Module):
    """Table 1, generator column. `(B, 2, n_bins, 336)` -> `(B, 2, n_bins, 212)`."""

    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        e1, e2 = cfg.model.early_ch
        self.cfg = cfg
        self.front = FrequencyAggregation(cfg, in_ch=2)
        self.stack = DilatedStack(
            cfg,
            pad_time=False,
            noise_ch=cfg.model.noise_ch,
            checkpoint=cfg.train.grad_checkpoint,
        )
        self.conv4_out = cfg.model.conv4_out
        # DeConv4 inverts Conv4; Table 1's In/Out columns are written in terms of
        # the convolution being inverted, so read the Output Size column instead.
        self.expand = nn.ConvTranspose2d(cfg.model.conv4_out, e2, (cfg.stft.n_bins, 1))
        self.expand_act = nn.PReLU(e2)
        self.late = nn.Sequential(
            nn.ConvTranspose2d(e2, e2, 3, dilation=4, padding=4),
            nn.PReLU(e2),
            nn.ConvTranspose2d(e2, e1, 3, dilation=2, padding=2),
            nn.PReLU(e1),
            nn.ConvTranspose2d(e1, 2, 3, dilation=1, padding=1),
            # Table 1 lists PReLU on DeConv1 too — the generator's output is
            # activated, not raw. Unusual for a signed spectrogram (PReLU scales
            # negatives by a learnable slope, so it is asymmetric), but it is
            # what the table specifies; the separate "(Output)" row carries no
            # non-linearity of its own.
            nn.PReLU(2),
        )
        init_weights(self, cfg)

    def forward(self, y: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        x = self.stack(self.front(y), noise)  # (B, 128, 32, 212)
        b, _, _, t = x.shape
        x = x.reshape(b, self.conv4_out, 1, t)  # Reshape2
        return self.late(self.expand_act(self.expand(x)))


class Critic(nn.Module):
    """Table 1, critic column: ONE stack of grouped convolutions, joined at the
    output layer.

    **The layout below comes from the authors, not the paper.** Stefan Lattner
    replied to our questions (2026-09-29; §6.2) and settled both of the things
    Table 1 left ambiguous:

    > "The critic receives the MP3 by direct channel-wise concatenation. Its
    > effective input is eight channels: the two complex channels of the MP3,
    > the two complex channels of the original/restored candidate, then
    > two-channel versions of the MP3 magnitude and candidate magnitude. The two
    > convolutional groups are indeed grouped convolutions throughout the
    > critic: one group processes the complex MP3/candidate pair and the other
    > processes the magnitude MP3/candidate pair. They are joined only in the
    > final output layer. So the table's two-channel input description is
    > incomplete/ambiguous rather than indicating that the MP3 enters
    > elsewhere."

    So the input is::

        ch 0-1   condition (MP3), signed-sqrt complex     -.
        ch 2-3   candidate, signed-sqrt complex            |- group 0
        ch 4-5   condition magnitude, widened to 2 ch     -.
        ch 6-7   candidate magnitude, widened to 2 ch      |- group 1

    PyTorch splits a grouped convolution's channels contiguously, so this order
    is what puts the complex pair in one group and the magnitude pair in the
    other. The magnitudes are widened from 1 channel to 2 so both groups have
    identical shape, which is what `groups=2` requires — and what makes the
    author's "two-channel versions" phrase add up to eight.

    This replaces our earlier reading of two separate full-width stacks. That
    was wrong, and Table 1 had said so: `[Conv16]` takes **256** inputs, which
    only works if the two groups together are 256 wide, not 2 x 256. See §2.1.
    """

    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        m = cfg.model
        self.front = FrequencyAggregation(cfg, in_ch=8, groups=2)
        self.stack = DilatedStack(cfg, pad_time=True, skip=True, groups=2)
        # Conv15: still grouped — "grouped convolutions throughout the critic".
        self.head = nn.Sequential(
            nn.Conv2d(m.latent_ch, m.stack_ch, 3, dilation=1, padding=1, groups=2),
            nn.PReLU(m.stack_ch),
        )
        # Conv16: groups=1. "They are joined only in the final output layer."
        self.join = nn.Conv2d(m.stack_ch, 1, (m.latent_freq, 1))
        init_weights(self, cfg)

    def forward(self, candidate: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # `expand` rather than `repeat`: the widened magnitude channels are
        # identical copies, so there is nothing to allocate.
        cond_mag = sqrt_magnitude(condition).expand(-1, 2, -1, -1)
        cand_mag = sqrt_magnitude(candidate).expand(-1, 2, -1, -1)
        x = torch.cat([condition, candidate, cond_mag, cand_mag], dim=1)
        x = self.head(self.stack(self.front(x)))
        return self.join(x).squeeze(1).squeeze(1)
