"""Checks on the training Dataset. Needs the built corpus; skips without it."""

from pathlib import Path

import pytest
import torch

from mp3ml.config import load_config, output_frames, target_crop
from mp3ml.data import WaveformPairs, expected_shapes

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "faithful.yaml"
CFG = load_config(CONFIG)

pytestmark = pytest.mark.skipif(
    not (CFG.data.root / "index.json").exists(),
    reason="waveform dataset not built — run scripts/build_waveform_dataset.py",
)


@pytest.fixture(scope="module")
def train() -> WaveformPairs:
    return WaveformPairs(CFG, split="train")


def test_shapes_match_the_config(train: WaveformPairs) -> None:
    item = train[0]
    for key, want in expected_shapes(CFG).items():
        assert tuple(item[key].shape) == want, key


def test_target_is_the_centre_crop_not_the_front(train: WaveformPairs) -> None:
    """§2.6: G's output is centred on its input, so a front crop would offset
    target from prediction by 62 frames (~0.7 s) — silent, and invisible to
    magnitude-only metrics."""
    item = train[0]
    lo, hi = target_crop(CFG)
    assert torch.equal(item["clean"], item["clean_full"][..., lo:hi])
    assert not torch.equal(item["clean"], item["clean_full"][..., : hi - lo])


def test_offsets_vary_between_reads(train: WaveformPairs) -> None:
    """The dataset stores whole tracks so chunk boundaries can differ every
    epoch. A previous version seeded from torch.initial_seed(), constant within
    a process, so every read returned the identical window."""
    sums = {train[0]["degraded"].sum().item() for _ in range(4)}
    assert len(sums) == 4, "offset sampling is not random"


def test_offsets_are_reproducible_from_a_seed(train: WaveformPairs) -> None:
    torch.manual_seed(7)
    first = train[0]["degraded"].sum().item()
    torch.manual_seed(7)
    assert train[0]["degraded"].sum().item() == first


def test_splits_are_disjoint_by_track() -> None:
    """Leakage check: no track may appear in two splits."""
    tracks = {
        split: {e["track"] for e in WaveformPairs(CFG, split=split).entries}
        for split in ("train", "val", "test")
    }
    assert not tracks["train"] & tracks["val"]
    assert not tracks["train"] & tracks["test"]
    assert not tracks["val"] & tracks["test"]


def test_eval_split_can_be_restricted_to_mixtures() -> None:
    """Stems are training augmentation; evaluation uses mixtures only."""
    ds = WaveformPairs(CFG, split="test", stems=CFG.data.eval_stems)
    assert {e["stem"] for e in ds.entries} == {"mixture"}


def test_batches_are_finite_and_correctly_shaped(train: WaveformPairs) -> None:
    from torch.utils.data import DataLoader

    batch = next(iter(DataLoader(train, batch_size=4, shuffle=True)))
    assert batch["degraded"].shape == (4, 2, CFG.stft.n_bins, CFG.data.segment_frames)
    assert batch["clean"].shape == (4, 2, CFG.stft.n_bins, output_frames(CFG))
    assert torch.isfinite(batch["degraded"]).all()
    assert torch.isfinite(batch["clean"]).all()


def test_degraded_differs_from_clean(train: WaveformPairs) -> None:
    """Sanity: if these matched, the pairing or the crop would be wrong."""
    item = train[0]
    lo, hi = target_crop(CFG)
    rel = (item["degraded"][..., lo:hi] - item["clean"]).norm() / item["clean"].norm()
    assert 0.01 < rel < 1.0, f"implausible degraded/clean difference: {rel:.4f}"


def test_evaluation_crops_are_reproducible() -> None:
    """A benchmark must score the same audio every run.

    Training deliberately draws a random window per access — that is why whole
    tracks are stored. Evaluation inherited that, which meant two runs of
    `scripts/evaluate.py` on one checkpoint returned different numbers, and two
    checkpoints were compared on *different audio*. Found in the 2026-09-29
    sweep; nothing failed, the metrics were just quietly incomparable.
    """
    ds = WaveformPairs(CFG, split="test", stems=CFG.data.eval_stems, random_crop=False)
    reads = {round(float(ds[0]["degraded"].sum()), 6) for _ in range(4)}
    assert len(reads) == 1, "fixed-crop dataset returned different windows"


def test_fixed_crops_still_differ_between_entries() -> None:
    """Reproducible must not mean "always the track centre" — that would bias
    every metric toward the middle of songs."""
    ds = WaveformPairs(CFG, split="test", stems=CFG.data.eval_stems, random_crop=False)
    a, b = float(ds[0]["degraded"].sum()), float(ds[1]["degraded"].sum())
    assert a != b


def test_training_crops_stay_random() -> None:
    """The fix must not disable the per-epoch variation training relies on."""
    ds = WaveformPairs(CFG, split="train")
    reads = {round(float(ds[0]["degraded"].sum()), 6) for _ in range(4)}
    assert len(reads) > 1, "training crops stopped varying"


def test_evaluate_uses_fixed_crops() -> None:
    """The harness must opt in — the default is random, for training."""
    import inspect

    from mp3ml import evaluate as ev

    assert "random_crop=False" in inspect.getsource(ev.evaluate)
