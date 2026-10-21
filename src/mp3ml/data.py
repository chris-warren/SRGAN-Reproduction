"""Dataset over the waveform pairs, yielding the paper's spectrogram segments.

Returns, per item:

    degraded  (2, n_bins, segment_frames)   G's conditional input, y
    clean     (2, n_bins, output_frames)    target, centre-cropped (§2.6)
    clean_full(2, n_bins, segment_frames)   uncropped, for the critic

Segments are drawn at random offsets within a track (see DATASET.md — the
arrays are stored whole precisely so chunk boundaries can vary every epoch).
The clean target is **centre-cropped** to match G's unpadded output: G's
`output[i]` is centred on `input[i + sum(dilations)]`, so a front crop would
offset target from prediction by 62 frames. See REPRODUCTION.md §1.3.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import ExperimentConfig, output_frames, segment_samples, target_crop
from .stft import stft


class WaveformPairs(Dataset):
    """Spectrogram pairs for one split. Returns dicts of plain tensors.

    Shapes: `degraded` (2, n_bins, segment_frames), `clean` (2, n_bins,
    output_frames), `clean_full` (2, n_bins, segment_frames).
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        split: str,
        stems: tuple[str, ...] | None = None,
        seed: int = 0,
        random_crop: bool = True,
    ) -> None:
        self.cfg = cfg
        self.split = split
        self.seg_samples = segment_samples(cfg)
        self.crop = target_crop(cfg)
        self.seed = seed
        #: Training wants a different window every epoch — that is why whole
        #: tracks are stored. **Evaluation must not**: with random crops, two
        #: runs of the same checkpoint score different audio, so results are
        #: irreproducible and two checkpoints cannot be compared. `evaluate()`
        #: sets this False, which derives a fixed offset per entry from `seed`.
        self.random_crop = random_crop

        index = json.loads((cfg.data.root / "index.json").read_text())
        allowed = stems if stems is not None else cfg.data.train_stems
        self.entries = [
            e
            for e in index["entries"]
            if e["split"] == split
            and e["stem"] in allowed
            and e["bitrate_kbps"] in cfg.data.bitrates
            and e["n_samples"] >= self.seg_samples
        ]
        if not self.entries:
            raise ValueError(f"no entries for split={split!r}, stems={allowed}")
        # Opened lazily so DataLoader workers each get their own handles.
        self._arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def _load(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        if i not in self._arrays:
            e = self.entries[i]
            root = self.cfg.data.root
            self._arrays[i] = (
                np.load(root / e["clean_path"], mmap_mode="r"),
                np.load(root / e["degraded_path"], mmap_mode="r"),
            )
        return self._arrays[i]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        clean_arr, degraded_arr = self._load(i)
        # Random offset per access, so chunk boundaries differ every epoch —
        # the reason the dataset stores whole tracks rather than fixed chunks.
        # Drawn from torch's global RNG, which DataLoader re-seeds per worker per
        # epoch, so offsets vary across epochs yet stay reproducible from a seed.
        # (An earlier version seeded from torch.initial_seed(), which is constant
        # within a process, so every read of an item returned the same offset.)
        span = len(clean_arr) - self.seg_samples + 1
        if span <= 1:
            offset = 0
        elif self.random_crop:
            offset = int(torch.randint(0, span, (1,)).item())
        else:
            # Reproducible but still varied across entries: a per-item generator
            # seeded from (seed, index). Always taking the track centre would
            # bias every metric toward the middle of songs.
            gen = torch.Generator().manual_seed(self.seed * 1_000_003 + i)
            offset = int(torch.randint(0, span, (1,), generator=gen).item())
        sl = slice(offset, offset + self.seg_samples)

        s = self.cfg.stft
        to_spec = lambda a: stft(  # noqa: E731
            # np.array copies: a memmap slice is read-only and torch warns on it
            torch.from_numpy(np.array(a[sl], dtype=np.int16)).float() / 32768.0,
            n_fft=s.n_fft,
            hop=s.hop,
            n_bins=s.n_bins,
        )
        clean_full, degraded = to_spec(clean_arr), to_spec(degraded_arr)
        lo, hi = self.crop
        return {
            "degraded": degraded,
            "clean": clean_full[..., lo:hi],
            "clean_full": clean_full,
            "bitrate": torch.tensor(self.entries[i]["bitrate_kbps"]),
        }


def expected_shapes(cfg: ExperimentConfig) -> dict[str, tuple[int, ...]]:
    """Shapes `WaveformPairs` must produce, derived from the config."""
    bins, frames = cfg.stft.n_bins, cfg.data.segment_frames
    return {
        "degraded": (2, bins, frames),
        "clean": (2, bins, output_frames(cfg)),
        "clean_full": (2, bins, frames),
    }


def build_train_dataset(cfg: ExperimentConfig):
    """The training Dataset named by `cfg.data.source`.

    Both options return the same item contract, so the Trainer does not care
    which corpus it is reading — `ApolloPairs` exists so this reproduction and
    the Apollo replication can be scored on identical audio.
    """
    if cfg.data.source == "apollo":
        from .apollo import ApolloPairs

        return ApolloPairs(cfg, split="train")
    return WaveformPairs(cfg, split="train")
