"""The training Dataset, and the shapes it must produce.

The reproduction trains exclusively from the **Apollo corpus** so that it and
the Apollo replication can be scored on identical audio, so the Dataset itself
lives in `mp3ml.apollo`. This module keeps the shape contract, which is derived
from the config and asserted in tests.

An earlier `WaveformPairs` read this project's own pre-aligned (clean, degraded)
waveform pairs. That phase moved to `/home/chris/thesis/mdct-granule-pairs` on
2026-10-21; the built 58 GB corpus still exists and is the only dataset here
containing 16 kbit/s, which Apollo's lacks.
"""

from __future__ import annotations

from .config import ExperimentConfig, output_frames


def expected_shapes(cfg: ExperimentConfig) -> dict[str, tuple[int, ...]]:
    """Shapes the training Dataset must produce, derived from the config."""
    bins, frames = cfg.stft.n_bins, cfg.data.segment_frames
    return {
        "degraded": (2, bins, frames),
        "clean": (2, bins, output_frames(cfg)),
        "clean_full": (2, bins, frames),
    }


def build_train_dataset(cfg: ExperimentConfig):
    """The training Dataset. Kept as a function so the Trainer stays agnostic."""
    from .apollo import ApolloPairs

    return ApolloPairs(cfg, split="train")
