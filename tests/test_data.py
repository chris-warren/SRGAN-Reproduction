"""The training Dataset's contract, against a synthetic Apollo tree.

Builds a miniature `apollo_v1` layout in `tmp_path` rather than touching the
real 273 GB corpus, so these stay fast and runnable anywhere — the project's
test rule. The real corpus is exercised only by the benchmark script.
"""

import shutil

import numpy as np
import pytest
import torch

h5py = pytest.importorskip("h5py", reason="needs the `apollo` extra")

from mp3ml.apollo import WINDOW_SAMPLES, ApolloPairs, index_windows  # noqa: E402
from mp3ml.config import load_config, output_frames  # noqa: E402
from mp3ml.data import expected_shapes  # noqa: E402

CONFIG_DIR = __import__("pathlib").Path(__file__).resolve().parents[1] / "configs"

needs_codec = pytest.mark.skipif(
    shutil.which("lame") is None or shutil.which("ffmpeg") is None,
    reason="needs lame and ffmpeg on PATH",
)


@pytest.fixture
def apollo_root(tmp_path):
    """A two-stem, few-window Apollo tree with real broadband audio."""
    rng = np.random.default_rng(0)
    t = np.arange(WINDOW_SAMPLES) / 44100
    for stem, n_files, n_windows in (("vocals", 2, 3), ("mixture", 1, 2)):
        d = tmp_path / "hdf5_train" / stem
        d.mkdir(parents=True)
        for f in range(n_files):
            data = np.empty((n_windows, 2, WINDOW_SAMPLES), dtype=np.float32)
            for w in range(n_windows):
                for c in range(2):
                    x = 0.3 * np.sin(2 * np.pi * (220 + 40 * w + 7 * c) * t)
                    x += 0.05 * rng.standard_normal(WINDOW_SAMPLES)
                    data[w, c] = np.clip(x, -1, 1)
            with h5py.File(d / f"synthetic__track{f}__{stem}.hdf5", "w") as h:
                h.create_dataset("data", data=data)
    return tmp_path


@pytest.fixture
def cfg(apollo_root):
    """The smoke config, pointed at the synthetic tree."""
    import dataclasses

    c = load_config(CONFIG_DIR / "smoke.yaml")
    return dataclasses.replace(
        c,
        data=dataclasses.replace(
            c.data,
            source="apollo",
            root=apollo_root,
            bitrates=(64,),
            train_stems=("vocals", "mixture"),
        ),
    )


def test_index_covers_every_window_and_channel(cfg) -> None:
    """2 vocals files x 3 windows + 1 mixture file x 2 windows, both channels."""
    assert len(index_windows(cfg.data.root)) == (2 * 3 + 1 * 2) * 2


def test_index_can_be_restricted_to_stems(cfg) -> None:
    only = index_windows(cfg.data.root, stems=("mixture",))
    assert len(only) == 1 * 2 * 2
    assert {w.stem for w in only} == {"mixture"}


def test_index_rejects_a_different_window_length(cfg, tmp_path) -> None:
    """A changed `apollo_v1` revision must fail loudly, not silently mis-crop."""
    d = cfg.data.root / "hdf5_train" / "wrong"
    d.mkdir(parents=True)
    with h5py.File(d / "synthetic__bad__wrong.hdf5", "w") as h:
        h.create_dataset("data", data=np.zeros((1, 2, 1000), dtype=np.float32))
    with pytest.raises(ValueError, match="samples per window"):
        index_windows(cfg.data.root, stems=("wrong",))


def test_rejects_bitrates_that_resample(cfg) -> None:
    import dataclasses

    bad = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, bitrates=(32,)))
    with pytest.raises(ValueError, match="resample"):
        ApolloPairs(bad)


def test_rejects_a_non_training_split(cfg) -> None:
    with pytest.raises(ValueError, match="hdf5_train"):
        ApolloPairs(cfg, split="val")


@needs_codec
def test_shapes_match_the_config(cfg) -> None:
    item = ApolloPairs(cfg)[0]
    for name, shape in expected_shapes(cfg).items():
        assert tuple(item[name].shape) == shape, name
    assert item["bitrate"].item() == 64


@needs_codec
def test_target_is_the_centre_crop_not_the_front(cfg) -> None:
    """G is unpadded in time, so its output aligns with the input's centre.

    A front crop would offset target from prediction by `sum(dilations)`
    frames and every loss would be computed against the wrong audio.
    """
    item = ApolloPairs(cfg)[0]
    lo = sum(cfg.model.dilations)
    assert item["clean"].shape[-1] == output_frames(cfg)
    assert torch.allclose(item["clean"], item["clean_full"][..., lo : lo + output_frames(cfg)])
    assert not torch.allclose(item["clean"], item["clean_full"][..., : output_frames(cfg)])


@needs_codec
def test_offsets_vary_between_reads(cfg) -> None:
    """Whole windows are stored precisely so the crop can move every epoch."""
    ds = ApolloPairs(cfg, random_crop=True)
    torch.manual_seed(0)
    seen = {ApolloPairs.__getitem__(ds, 0)["clean"].sum().item() for _ in range(6)}
    assert len(seen) > 1, "random_crop=True returned the same window every read"


@needs_codec
def test_fixed_crops_are_reproducible_and_still_differ_between_items(cfg) -> None:
    """Evaluation must score identical audio across runs, but not one window."""
    a = ApolloPairs(cfg, random_crop=False, seed=7)
    b = ApolloPairs(cfg, random_crop=False, seed=7)
    assert a[0]["clean"].sum().item() == b[0]["clean"].sum().item()
    assert a[0]["clean"].sum().item() != a[3]["clean"].sum().item()


@needs_codec
def test_degraded_differs_from_clean_but_is_finite(cfg) -> None:
    """A pass-through would make every metric downstream meaningless."""
    item = ApolloPairs(cfg)[0]
    for v in item.values():
        assert torch.isfinite(v).all()
    full, deg = item["clean_full"], item["degraded"]
    rel = (deg - full).norm() / full.norm()
    assert 1e-3 < rel < 1.0, f"relative difference {rel:.4f} is implausible"
