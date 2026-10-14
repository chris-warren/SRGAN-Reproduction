"""Checks on experiment configs, especially the constraints that break silently."""

from pathlib import Path

import pytest
import yaml

from mp3ml.config import (
    ExperimentConfig,
    ModelConfig,
    describe_architecture,
    estimate_size,
    fits_in,
    load_config,
    output_frames,
    segment_samples,
    target_crop,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_every_shipped_config_loads() -> None:
    paths = sorted(CONFIG_DIR.glob("*.yaml"))
    assert paths, "no configs found"
    for path in paths:
        assert load_config(path).name


def test_reshape_constraint_is_enforced() -> None:
    """conv4_out must equal latent_ch * latent_freq or Reshape1 is impossible.

    Getting this wrong produces a shape error deep in the model rather than at
    config load, so it is checked eagerly.
    """
    with pytest.raises(ValueError, match="Reshape1"):
        ModelConfig(conv4_out=4096, latent_ch=64, latent_freq=32)


def test_latent_freq_must_exceed_the_widest_dilation() -> None:
    """A 3-tap conv at dilation d >= latent_freq is a 1-tap conv in frequency.

    Table 1's schedule tops out at dilation 16 on a 32-row latent axis. A
    reduced config that shrinks the axis to 16 keeps every shape and parameter
    count legal while silently deleting the frequency context of four layers,
    so it is rejected at load (REPRODUCTION.md §3.1).
    """
    with pytest.raises(ValueError, match="widest dilation"):
        ModelConfig(conv4_out=2048, latent_ch=128, latent_freq=16)


def test_shipped_configs_satisfy_the_reshape_constraint() -> None:
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        m = load_config(path).model
        assert m.latent_ch * m.latent_freq == m.conv4_out, path


def test_faithful_matches_the_paper() -> None:
    """Table 1 values. If any of these drift, the reproduction is no longer faithful."""
    cfg = load_config(CONFIG_DIR / "faithful.yaml")
    assert cfg.stft.n_fft == 2048
    assert cfg.stft.hop == 512
    assert cfg.stft.n_bins == 1024
    assert cfg.data.segment_frames == 336  # 4 s, paper section 3.3
    assert cfg.model.conv4_out == 4096
    assert cfg.model.latent_ch == 128
    assert cfg.model.latent_freq == 32
    assert cfg.model.stack_ch == 256
    assert cfg.model.noise_ch == 64
    assert cfg.train.profile_p_freq == 1.3  # paper eq. 3
    assert cfg.train.profile_p_rhyt == 1.6  # paper eq. 3


def test_conv4_dominates_the_parameter_count() -> None:
    """The frequency-collapsing layer is why the faithful model needs cloud hardware."""
    est = estimate_size(load_config(CONFIG_DIR / "faithful.yaml"))
    assert est["conv4_params"] > 150e6
    # 404 M since the critic became grouped (REPRODUCTION.md 1.1); it was 649 M
    # while we wrongly built two separate full-width critic stacks.
    assert est["total_params"] > 350e6
    assert est["conv4_params"] / est["total_params"] > 0.35


def test_local_fits_an_8gb_card_and_faithful_does_not() -> None:
    assert fits_in(load_config(CONFIG_DIR / "local.yaml"), 8.0)[0]
    assert not fits_in(load_config(CONFIG_DIR / "faithful.yaml"), 8.0)[0]


def test_local_is_flagged_as_not_comparable() -> None:
    """local.yaml must say so in its notes — it is not a reproduction."""
    notes = load_config(CONFIG_DIR / "local.yaml").notes.lower()
    assert "not" in notes and "comparable" in notes


def test_yaml_overrides_defaults(tmp_path) -> None:
    path = tmp_path / "x.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "probe",
                "stft": {"n_bins": 256},
                "model": {"conv4_out": 512, "latent_ch": 16, "latent_freq": 32},
                "train": {"batch_size": 3},
            }
        )
    )
    cfg = load_config(path)
    assert cfg.name == "probe"
    assert cfg.stft.n_bins == 256
    assert cfg.train.batch_size == 3
    assert cfg.stft.hop == ExperimentConfig().stft.hop  # untouched default


def test_smaller_frequency_axis_shrinks_conv4_proportionally() -> None:
    full = estimate_size(load_config(CONFIG_DIR / "faithful.yaml"))
    small = estimate_size(load_config(CONFIG_DIR / "local.yaml"))
    assert small["conv4_params"] < full["conv4_params"] / 4


#: Table 1 of Lattner & Nistal, generator column, training case (336 frames, no
#: time padding). Every output size, verbatim from the paper. If `faithful.yaml`
#: or `describe_architecture` drifts from this, the reproduction is broken.
TABLE_1_GENERATOR = [
    ("Input", 2, 1024, 336),
    ("Conv1", 18, 1024, 336),
    ("Conv2", 38, 1024, 336),
    ("Conv3", 38, 1024, 336),
    ("Conv4", 4096, 1, 336),
    ("Reshape1", 128, 32, 336),
    ("ReMap", 256, 32, 336),
    ("Conv5", 256, 32, 334),
    ("NoiseConcat", 320, 32, 334),
    ("Conv6", 256, 32, 330),
    ("SelfGating6", 128, 32, 330),
    ("Conv7", 256, 32, 322),
    ("SelfGating7", 128, 32, 322),
    ("Conv8", 256, 32, 306),
    ("SelfGating8", 128, 32, 306),
    ("Conv9", 256, 32, 274),
    ("SelfGating9", 128, 32, 274),
    ("Conv10", 256, 32, 272),
    ("SelfGating10", 128, 32, 272),
    ("Conv11", 256, 32, 268),
    ("SelfGating11", 128, 32, 268),
    ("Conv12", 256, 32, 260),
    ("SelfGating12", 128, 32, 260),
    ("Conv13", 256, 32, 244),
    ("SelfGating13", 128, 32, 244),
    ("Conv14", 256, 32, 212),
    ("SelfGating14", 128, 32, 212),
    ("Reshape2", 4096, 1, 212),
    ("DeConv4", 38, 1024, 212),
    ("DeConv3", 38, 1024, 212),
    ("DeConv2", 18, 1024, 212),
    ("DeConv1", 2, 1024, 212),
    ("Output", 2, 1024, 212),
]


def test_faithful_config_reproduces_table_1_exactly() -> None:
    """The strongest faithfulness check we have: every layer, every output size."""
    rows = describe_architecture(load_config(CONFIG_DIR / "faithful.yaml"))
    assert len(rows) == len(TABLE_1_GENERATOR)
    for row, (name, ch, freq, time) in zip(rows, TABLE_1_GENERATOR, strict=True):
        got = (row["layer"], row["ch"], row["freq"], row["time"])
        assert got == (name, ch, freq, time), f"Table 1 mismatch: {got} != {(name, ch, freq, time)}"


def test_time_shrink_matches_the_paper() -> None:
    """§3.1.1: no time padding in G, so 4 s (336 frames) yields 212 output frames."""
    assert output_frames(load_config(CONFIG_DIR / "faithful.yaml")) == 212


def test_local_keeps_the_papers_time_structure() -> None:
    """local.yaml may shrink frequency, width and batch, but not the time axis.

    Batch size deviates (4 vs the paper's 12) because a measured training step
    OOMs above batch 6 on an 8 GB card — see REPRODUCTION.md section 3.1.
    """
    local = load_config(CONFIG_DIR / "local.yaml")
    faithful = load_config(CONFIG_DIR / "faithful.yaml")
    assert local.model.dilations == faithful.model.dilations
    assert local.data.segment_frames == faithful.data.segment_frames
    assert output_frames(local) == output_frames(faithful) == 212


def test_paper_training_values_are_locked() -> None:
    """§3.2 states these outright — drift here means we stopped reproducing the paper."""
    t = load_config(CONFIG_DIR / "faithful.yaml").train
    assert t.iterations == 40_000
    assert t.batch_size == 12
    assert t.lr_g == 1e-3 and t.lr_d == 1e-3
    assert t.init == "he"
    assert t.noise_draws == 2  # §3.2.1: two noise draws per item


def test_gating_is_not_a_standard_glu() -> None:
    """§3.1.2 / eq. 2: PReLU on the gated half, not linear. nn.GLU would be wrong."""
    assert load_config(CONFIG_DIR / "faithful.yaml").model.gating == "prelu_sigmoid"


def test_segment_samples_yields_exactly_the_frame_count() -> None:
    """§2.5: we match 336 frames, not the nominal 4 seconds."""
    cfg = load_config(CONFIG_DIR / "faithful.yaml")
    n = segment_samples(cfg)
    frames = (n - cfg.stft.n_fft) // cfg.stft.hop + 1  # center=False
    assert frames == cfg.data.segment_frames == 336
    assert n == 173_568


def test_target_crop_is_centred() -> None:
    """§2.6: G's output aligns with the centre of its input, not the start."""
    cfg = load_config(CONFIG_DIR / "faithful.yaml")
    start, stop = target_crop(cfg)
    assert (start, stop) == (62, 274)
    assert stop - start == output_frames(cfg) == 212
    assert start == cfg.data.segment_frames - stop  # symmetric crop
