"""Checks that the model matches Table 1 — shapes, wiring and non-linearities."""

import inspect
from pathlib import Path

import pytest
import torch
from torch import nn

from mp3ml.config import load_config, output_frames, target_crop
from mp3ml.models import Critic, Generator, SelfGatedConv2d
from mp3ml.train import sample_noise

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CFG = load_config(CONFIG_DIR / "local.yaml")  # same wiring, smaller; CPU-friendly


@pytest.fixture(scope="module")
def models() -> tuple[Generator, Critic]:
    torch.manual_seed(0)
    return Generator(CFG), Critic(CFG)


def test_generator_shrinks_time_to_the_table_1_value(models) -> None:
    g, _ = models
    y = torch.randn(2, 2, CFG.stft.n_bins, CFG.data.segment_frames)
    with torch.no_grad():
        out = g(y)
    assert out.shape == (2, 2, CFG.stft.n_bins, output_frames(CFG))


def test_critic_emits_one_score_per_frame(models) -> None:
    """§3.1.1: D is convolutional in time, "as many loss outputs as there are
    spectrogram frames", averaged for the Wasserstein loss."""
    g, d = models
    y = torch.randn(2, 2, CFG.stft.n_bins, CFG.data.segment_frames)
    lo, hi = target_crop(CFG)
    with torch.no_grad():
        score = d(g(y), y[..., lo:hi])
    assert score.shape == (2, output_frames(CFG))


def test_generator_output_is_activated() -> None:
    """Table 1 lists PReLU on DeConv1, so G's final layer is activated.

    Easy to omit — the separate "(Output)" row has no non-linearity, which reads
    as though the last convolution is raw.
    """
    g = Generator(CFG)
    assert isinstance(g.late[-1], nn.PReLU), "missing the PReLU Table 1 puts on DeConv1"


def test_critic_output_layer_has_no_activation() -> None:
    """Table 1's [Conv16] non-linearity column is "-": a Wasserstein critic must
    be unbounded, so no squashing at the output."""
    d = Critic(CFG)
    assert isinstance(d.join, nn.Conv2d)


def test_self_gating_is_prelu_times_sigmoid_not_glu() -> None:
    """Eq. 2. A standard GLU uses a linear gated half; §3.1.2 says linear units
    "resulted in unstable training", so PReLU is required."""
    layer = SelfGatedConv2d(4, 8, dilation=1, pad_time=True)
    assert isinstance(layer.act, nn.PReLU)
    assert layer.conv.out_channels == 16, "one conv must emit twice the gated width"
    x = torch.randn(1, 4, 8, 8)
    with torch.no_grad():
        gated, gate = layer.conv(x).chunk(2, dim=1)
        assert torch.allclose(layer(x), layer.act(gated) * torch.sigmoid(gate))


def test_critic_takes_eight_channels_in_two_groups(models) -> None:
    """[P] From the authors (REPRODUCTION.md 1.1): "Its effective input is eight
    channels: the two complex channels of the MP3, the two complex channels of
    the original/restored candidate, then two-channel versions of the MP3
    magnitude and candidate magnitude. The two convolutional groups are indeed
    grouped convolutions throughout the critic ... joined only in the final
    output layer."

    We previously built two separate full-width stacks with 4-channel inputs.
    Table 1 had already contradicted that — `[Conv16]` takes 256 inputs, which
    only works if the two groups together are 256 wide.
    """
    _, d = models
    assert d.front.early[0].in_channels == 8
    assert d.front.early[0].groups == 2
    assert d.stack.conv5.groups == 2
    assert d.head[0].groups == 2
    # Conv16 joins: ungrouped, and Table 1's stack_ch inputs (not 2x).
    assert d.join.groups == 1
    assert d.join.in_channels == CFG.model.stack_ch


def test_every_critic_conv_is_grouped_except_the_output_layer(models) -> None:
    """ "Grouped convolutions **throughout** the critic" — a single ungrouped
    layer anywhere in the stack would let the two views mix early, which is the
    thing the design exists to prevent."""
    _, d = models
    ungrouped = [
        name
        for name, mod in d.named_modules()
        if isinstance(mod, nn.Conv2d) and mod.groups == 1 and name != "join"
    ]
    assert ungrouped == [], f"ungrouped convs in the critic stack: {ungrouped}"


def test_critic_generator_split_matches_author_channel_order(models) -> None:
    """PyTorch splits grouped channels contiguously, so the concat order is what
    puts the complex pair in group 0 and the magnitude pair in group 1. If the
    order were wrong the model would still run and train — silently mixing a
    magnitude channel into the complex critic."""
    import mp3ml.models as mm

    source = inspect.getsource(mm.Critic.forward)
    order = source.index("condition, candidate, cond_mag, cand_mag")
    assert order > 0, "channel order must be [cond, cand, cond_mag, cand_mag]"


def test_noise_changes_the_output(models) -> None:
    """§3.2: the stochastic generator must actually depend on z."""
    g, _ = models
    y = torch.randn(1, 2, CFG.stft.n_bins, CFG.data.segment_frames)
    with torch.no_grad():
        assert not torch.allclose(g(y), g(y))


def test_generator_is_deterministic_given_noise(models) -> None:
    g, _ = models
    y = torch.randn(1, 2, CFG.stft.n_bins, CFG.data.segment_frames)
    frames = CFG.data.segment_frames - 2 * CFG.model.dilations[0]
    z = torch.randn(1, CFG.model.noise_ch, CFG.model.latent_freq, frames)
    with torch.no_grad():
        assert torch.allclose(g(y, z), g(y, z))


def test_parameter_count_matches_the_estimator(models) -> None:
    from mp3ml.config import estimate_size

    g, d = models
    est = estimate_size(CFG)
    assert sum(p.numel() for p in g.parameters()) == pytest.approx(
        est["generator_params"], rel=0.01
    )
    assert sum(p.numel() for p in d.parameters()) == pytest.approx(est["critic_params"], rel=0.05)


def test_generator_accepts_variable_length_input() -> None:
    """§3.1.1: "At test time, G is applied to **variable-length** and
    potentially relatively long input sequences (e.g., full songs). In such a
    setting, G does not perform very well if trained on short excerpts with
    zero-padding in the time dimension. Therefore, we do not use zero-padding
    for G during training."

    So variable-length inference is the *reason* for the no-padding rule, not a
    side effect — which makes it a requirement on the implementation, not just a
    nice property. Every length must shrink by exactly sum(2*dilations).
    """
    from mp3ml.config import load_config
    from mp3ml.train import sample_noise

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml")
    g = Generator(cfg)
    shrink = sum(2 * d for d in cfg.model.dilations)
    z = sample_noise(cfg, 1, torch.device("cpu"))
    for frames in (cfg.data.segment_frames, 200, 400):
        out = g(torch.randn(1, 2, cfg.stft.n_bins, frames), z)
        assert out.shape[-1] == frames - shrink, f"{frames} frames in"


def test_generator_rejects_input_too_short_for_the_stack() -> None:
    """Below sum(2*dilations)+1 frames there is nothing left to output. Better a
    clear error than a zero-width tensor propagating into the loss."""
    from mp3ml.config import load_config
    from mp3ml.train import sample_noise

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml")
    g = Generator(cfg)
    z = sample_noise(cfg, 1, torch.device("cpu"))
    with pytest.raises(RuntimeError):
        g(torch.randn(1, 2, cfg.stft.n_bins, 64), z)


def test_critic_groups_stay_independent_end_to_end() -> None:
    """The whole point of `groups=2`: one group sees the complex pair, the other
    the magnitude pair, and they must not mix until the output layer.

    This is checked by perturbation rather than by reading the code, because
    every plausible way to break it is invisible. `Reshape1` folds 4096 channels
    into `latent_ch x latent_freq`, and if that reshape interleaved the groups
    instead of keeping them contiguous, the two critics would silently blend and
    nothing would fail. Likewise a plain `chunk(2)` in the self-gating would make
    group 1 the gate for group 0.
    """
    from mp3ml.stft import sqrt_magnitude

    torch.manual_seed(0)
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml")
    d = Critic(cfg).eval()
    bins, frames = cfg.stft.n_bins, output_frames(cfg)
    cand, cond = (torch.randn(1, 2, bins, frames) * 0.1 for _ in range(2))

    x = torch.cat(
        [
            cond,
            cand,
            sqrt_magnitude(cond).expand(-1, 2, -1, -1),
            sqrt_magnitude(cand).expand(-1, 2, -1, -1),
        ],
        dim=1,
    )
    stack = lambda inp: d.head(d.stack(d.front(inp)))  # noqa: E731
    base = stack(x)
    half = base.shape[1] // 2

    bumped_mag = x.clone()
    bumped_mag[:, 4:8] += 1.0
    bumped_cplx = x.clone()
    bumped_cplx[:, 0:4] += 1.0
    mag_out, cplx_out = stack(bumped_mag), stack(bumped_cplx)

    assert torch.allclose(base[:, :half], mag_out[:, :half], atol=1e-6), (
        "perturbing the magnitude channels changed the complex group"
    )
    assert torch.allclose(base[:, half:], cplx_out[:, half:], atol=1e-6), (
        "perturbing the complex channels changed the magnitude group"
    )
    # ...and each perturbation must reach its OWN group, or the test is vacuous.
    assert not torch.allclose(base[:, half:], mag_out[:, half:], atol=1e-6)
    assert not torch.allclose(base[:, :half], cplx_out[:, :half], atol=1e-6)


def test_generator_output_is_centred_on_the_target_crop() -> None:
    """§2.6 derived `target_crop` from the receptive-field centre. This measures
    it: output frame `k` must be centred on input frame `k + sum(dilations)`.

    Probed away from the edges — frame 0's receptive field is clipped at index 0,
    which shifts its apparent centre and would make this look wrong.
    """
    import copy

    from mp3ml.train import sample_noise

    torch.manual_seed(0)
    cfg = copy.deepcopy(load_config(Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"))
    cfg.data.segment_frames = 300  # wide enough that the probes do not clip
    g = Generator(cfg)
    bins, out_frames = cfg.stft.n_bins, output_frames(cfg)
    pad = sum(cfg.model.dilations)

    for k in (out_frames // 4, out_frames // 2):
        y = torch.zeros(1, 2, bins, cfg.data.segment_frames, requires_grad=True)
        g.zero_grad()
        g(y, sample_noise(cfg, 1, torch.device("cpu")))[0, 0, bins // 2, k].backward()
        used = y.grad.abs().sum(dim=(0, 1, 2)).nonzero().flatten()
        centre = (used.min().item() + used.max().item()) / 2
        assert abs(centre - (pad + k)) < 0.6, f"output[{k}] centred at {centre}, want {pad + k}"


def test_no_parameter_is_left_without_gradient() -> None:
    """A dead branch — a layer built but never called, or detached by accident —
    shows up as a parameter that never receives a gradient. Nothing else catches
    it: the model trains, just with a piece of itself frozen at initialisation.
    """
    from mp3ml.losses import critic_step_loss, profile_loss, wasserstein_generator_loss
    from mp3ml.train import sample_noise

    torch.manual_seed(0)
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml")
    g, d = Generator(cfg), Critic(cfg)
    bins = cfg.stft.n_bins
    y = torch.randn(2, 2, bins, cfg.data.segment_frames) * 0.1
    real = torch.randn(2, 2, bins, output_frames(cfg)) * 0.1
    lo, hi = target_crop(cfg)
    cond = y[..., lo:hi]
    cpu = torch.device("cpu")

    z_i, z_j = sample_noise(cfg, 2, cpu), sample_noise(cfg, 2, cpu)
    out_i, out_j = g(y, z_i), g(y, z_j)
    gen_loss = 0.5 * (
        wasserstein_generator_loss(d(out_i, cond)) + wasserstein_generator_loss(d(out_j, cond))
    ) + profile_loss(out_i, out_j, z_i, z_j)
    g.zero_grad()
    d.zero_grad()
    gen_loss.backward()
    dead = [n for n, p in g.named_parameters() if p.grad is None or not p.grad.any()]
    assert dead == [], f"generator parameters with no gradient: {dead}"

    g.zero_grad()
    d.zero_grad()
    critic_step_loss(d, real, out_i.detach(), cond)[0].backward()
    dead = [n for n, p in d.named_parameters() if p.grad is None or not p.grad.any()]
    assert dead == [], f"critic parameters with no gradient: {dead}"


def test_equalized_lr_gives_every_layer_the_same_relative_step_size() -> None:
    """[I] PGGAN's equalized learning rate, from the authors' own GAN codebase
    (REPRODUCTION.md 1.2). Weights start at N(0,1) and He's constant scales the
    layer output at runtime.

    The point is not the initialisation itself but the *relative* step size: a
    fixed learning rate perturbs a layer in proportion to 1/weight_scale, and
    with He baked into the weights that varies 35x across this model — Conv4
    moves 10.2% of its own scale per step at lr 1e-3 while Conv1 moves 0.3%.
    """
    import copy

    cfg = copy.deepcopy(load_config(Path(__file__).resolve().parents[1] / "configs" / "local.yaml"))

    def spread(equalized: bool) -> float:
        cfg.model.equalized_lr = equalized
        torch.manual_seed(0)
        g = Generator(cfg)
        stds = [
            m.weight.std().item()
            for m in g.modules()
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))
        ]
        return max(stds) / min(stds)

    assert spread(True) < 1.5, "equalized LR should put every layer near unit scale"
    assert spread(False) > 10.0, "baked-in He is expected to span an order of magnitude"


def test_equalized_lr_preserves_output_scale() -> None:
    """Scaling the output by He's constant must leave activations in the same
    regime as baked-in He — otherwise it is a different model, not a
    reparametrisation. Checked at the generator output, after the whole stack.
    """
    import copy

    cfg = copy.deepcopy(load_config(Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"))
    from mp3ml.train import sample_noise

    y = torch.randn(1, 2, cfg.stft.n_bins, cfg.data.segment_frames) * 0.1
    scales = {}
    for equalized in (True, False):
        cfg.model.equalized_lr = equalized
        torch.manual_seed(0)
        g = Generator(cfg)
        with torch.no_grad():
            out = g(y, sample_noise(cfg, 1, torch.device("cpu")))
        scales[equalized] = out.abs().mean().item()
    ratio = scales[True] / scales[False]
    assert 0.02 < ratio < 50, f"output scale shifted {ratio:.3g}x — not a reparametrisation"


def test_critic_matches_table_1_row_by_row() -> None:
    """Table 1's critic column, checked cell by cell against the built modules.

    The generator has had this check since early on; the critic never did, which
    is part of why the two-separate-stacks reading survived as long as it did.
    `[Conv16]`'s 256 inputs were the clue that it was wrong — two full-width
    stacks would need 512 — and that cell is asserted here explicitly.
    """
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "faithful.yaml")
    d = Critic(cfg)
    bins = cfg.stft.n_bins

    def spec(conv):
        return (conv.in_channels, conv.out_channels, tuple(conv.kernel_size))

    assert spec(d.front.early[0]) == (8, 18, (3, 3))  # 8 ch: authors, section 6.2
    assert spec(d.front.early[2]) == (18, 38, (3, 3))
    assert spec(d.front.early[4]) == (38, 38, (3, 3))
    assert spec(d.front.collapse) == (38, 4096, (bins, 1))
    assert spec(d.stack.remap) == (128, 256, (1, 1))
    assert spec(d.stack.conv5) == (256, 256, (3, 3))
    # Table 1 Conv6 reads "(320)[256]": 320 in G (after the noise concat),
    # 256 in D, which has no noise input.
    assert d.stack.gated[0].conv.in_channels == 256
    assert spec(d.head[0]) == (128, 256, (3, 3))  # [Conv15]
    assert spec(d.join) == (256, 1, (cfg.model.latent_freq, 1))  # [Conv16]


def test_target_crop_matches_the_generators_real_receptive_field() -> None:
    """`target_crop`'s offset must be what G *actually* does, not what we assume.

    `test_target_crop_is_centred` only checks the offset is symmetric, which
    verifies the config's arithmetic against itself. If G's real frame
    reduction were asymmetric, every loss would be computed against a
    misaligned target and nothing in the suite would say so — the loss would
    simply be higher and the model would take the blame. That is the exact
    failure pattern behind two earlier bugs (REPRODUCTION.md §5).

    Measured by gradient: which input frames influence a given output frame.
    The input is deliberately **longer than a training segment** so the
    receptive field is not truncated at the boundary — clamped edges make the
    apparent centre drift, which is misleading enough that it briefly looked
    like a 7-frame bug when this was first measured.
    """
    cfg = load_config(CONFIG_DIR / "smoke.yaml")
    torch.manual_seed(0)
    g = Generator(cfg)
    long_frames = 400
    y = torch.randn(1, 2, cfg.stft.n_bins, long_frames, requires_grad=True)
    # `sample_noise` is a per-excerpt vector (batch, noise_ch), so it is
    # independent of the input length; `models._broadcast` does the repeating.
    z = sample_noise(cfg, 1, torch.device("cpu"))

    assert g(y, z).shape[-1] == long_frames - sum(2 * d for d in cfg.model.dilations)

    offsets = set()
    for out_frame in (100, 150, 200):
        y.grad = None
        g(y, z)[0, :, :, out_frame].abs().sum().backward()
        touched = (y.grad[0].abs().sum(dim=(0, 1)) > 0).nonzero().flatten()
        first, last = int(touched[0]), int(touched[-1])
        assert first > 0 and last < long_frames - 1, "field is clamped; lengthen the input"
        offsets.add((first + last) / 2 - out_frame)

    assert len(offsets) == 1, f"offset varies across output frames: {offsets}"
    assert offsets.pop() == target_crop(cfg)[0]
