"""Writing granule pairs to disk as per-track memmappable arrays.

Layout, one directory per bitrate and one per track inside it::

    <root>/
      index.json
      bitrate=128/
        <track name>/
          xr.npy        (n_granules, n_channels, 576) float32  clean target
          xr_hat.npy    (n_granules, n_channels, 576) float32  degraded input
          sideinfo.npy  (n_granules, n_channels)      structured
          audio.mp3     the compressed audio itself
          meta.json     encoder settings + build-time validation

Arrays are stored whole per track rather than pre-chunked, so chunk length stays
a training hyperparameter and a DataLoader can sample random offsets (different
chunk boundaries every epoch) instead of being locked to fixed cuts. Load with
``np.load(path, mmap_mode="r")`` and slice.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .oracle import BLOCK_TYPE_NAMES, check_alignment, dump_granules

#: Per-granule side info kept alongside the coefficients. The encoder-side and
#: decoder-side copies are verified equal at build time (see `meta["alignment"]`),
#: so only one copy of each is stored.
SIDEINFO_DTYPE = np.dtype(
    [
        ("block_type", "i1"),
        ("mixed_block_flag", "i1"),
        ("scalefac_scale", "i1"),
        ("preflag", "i1"),
        ("global_gain", "<i2"),
        ("big_values", "<i2"),
        ("subblock_gain", "i1", (3,)),
    ]
)


@dataclass(frozen=True)
class TrackEntry:
    """One (track, bitrate) directory, as recorded in the index."""

    track: str
    bitrate_kbps: int
    channels: int
    n_granules: int
    n_samples: int
    split: str
    path: str
    source: str
    duration_s: float


def _granule_view(records: np.ndarray, field: str, channels: int) -> np.ndarray:
    """Reshape flat records to (n_granules, n_channels, ...).

    mp3dump emits records frame-major, then granule, then channel, so channel is
    the fastest-varying axis and this reshape is exact.
    """
    flat = records[field]
    return flat.reshape(-1, channels, *flat.shape[1:])


def write_track(
    pcm: np.ndarray,
    out_dir: Path,
    bitrate_kbps: int,
    channels: int,
    samplerate: int = 44100,
) -> dict:
    """Dump one (track, bitrate) pair into `out_dir` and return its metadata.

    Raises if the encoder- and decoder-side side info disagree: that means the
    granule join is misaligned and every pair in the track would be invalid.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        granules = dump_granules(
            pcm,
            bitrate_kbps=bitrate_kbps,
            out_prefix=Path(tmp) / "g",
            samplerate=samplerate,
            channels=channels,
        )
        records = granules.records

        alignment = check_alignment(records)
        if alignment["block_type"] < 1.0 or alignment["global_gain"] < 1.0:
            raise RuntimeError(
                f"granule misalignment in {out_dir.name} @ {bitrate_kbps}k: "
                f"{alignment} — refusing to write invalid pairs"
            )

        np.save(out_dir / "xr.npy", _granule_view(records, "xr", channels))
        np.save(out_dir / "xr_hat.npy", _granule_view(records, "xr_hat", channels))

        n_granules = len(records) // channels
        side = np.empty((n_granules, channels), dtype=SIDEINFO_DTYPE)
        for name in SIDEINFO_DTYPE.names:
            side[name] = _granule_view(records, name, channels)
        np.save(out_dir / "sideinfo.npy", side)

        shutil.copyfile(granules.mp3_path, out_dir / "audio.mp3")

        meta = dict(granules.meta)
        meta["alignment"] = alignment
        meta["block_types"] = granules.block_type_counts()
        meta["n_granules"] = n_granules
        meta["mp3_bytes"] = (out_dir / "audio.mp3").stat().st_size
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    return meta


def load_track(track_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Memory-map one track's arrays: (xr, xr_hat, sideinfo, meta).

    Nothing is read into memory here — slice the returned arrays to pull in only
    the granules a batch needs.
    """
    xr = np.load(track_dir / "xr.npy", mmap_mode="r")
    xr_hat = np.load(track_dir / "xr_hat.npy", mmap_mode="r")
    side = np.load(track_dir / "sideinfo.npy", mmap_mode="r")
    meta = json.loads((track_dir / "meta.json").read_text())
    return xr, xr_hat, side, meta


@dataclass(frozen=True)
class Chunk:
    """A slice of granules. `xr`/`xr_hat` are (n_granules, n_channels, 576) float32.

    When several chunks are read at once a leading chunk axis is added, giving
    (n_chunks, chunk_granules, n_channels, 576).
    """

    xr: np.ndarray
    xr_hat: np.ndarray
    sideinfo: np.ndarray
    offset: int | np.ndarray


def chunk_count(n_granules: int, chunk_granules: int) -> int:
    """How many whole chunks of `chunk_granules` fit in a track.

    A trailing partial chunk is dropped so every chunk has the same shape.
    """
    if chunk_granules <= 0:
        raise ValueError("chunk_granules must be positive")
    return n_granules // chunk_granules


class TrackReader:
    """Read granule chunks from one built track directory.

    Chunk length is chosen here, at read time, not baked into the files, so it
    stays a training hyperparameter::

        r = TrackReader(track_dir, chunk_granules=64)
        r.chunk(7)                 # chunk number 7
        r.chunks([7, 8, 12])       # several, stacked on a leading axis
        r.window(offset=1234, length=64)   # arbitrary offset, not chunk-aligned

    Nothing is read until a chunk is requested; the arrays are memory-mapped and
    only the requested granules are faulted in.
    """

    def __init__(self, track_dir: Path, chunk_granules: int) -> None:
        self.track_dir = Path(track_dir)
        self.chunk_granules = chunk_granules
        self.xr, self.xr_hat, self.sideinfo, self.meta = load_track(self.track_dir)
        self.n_granules = int(self.xr.shape[0])
        self.n_channels = int(self.xr.shape[1])
        self.n_chunks = chunk_count(self.n_granules, chunk_granules)

    def window(self, offset: int, length: int | None = None) -> Chunk:
        """Granules [offset, offset+length). Use for random-offset sampling."""
        length = self.chunk_granules if length is None else length
        if offset < 0 or offset + length > self.n_granules:
            raise IndexError(
                f"window [{offset}, {offset + length}) outside "
                f"{self.n_granules} granules in {self.track_dir.name}"
            )
        sl = slice(offset, offset + length)
        return Chunk(
            xr=np.asarray(self.xr[sl]),
            xr_hat=np.asarray(self.xr_hat[sl]),
            sideinfo=np.asarray(self.sideinfo[sl]),
            offset=offset,
        )

    def chunk(self, i: int) -> Chunk:
        """Chunk number `i`, i.e. granules [i*chunk_granules, (i+1)*chunk_granules)."""
        if not -self.n_chunks <= i < self.n_chunks:
            raise IndexError(f"chunk {i} out of range (0..{self.n_chunks - 1})")
        i %= self.n_chunks
        return self.window(i * self.chunk_granules)

    def chunks(self, ids: list[int] | np.ndarray) -> Chunk:
        """Several chunks stacked on a leading axis, in the order given."""
        parts = [self.chunk(int(i)) for i in ids]
        if not parts:
            raise ValueError("no chunk ids given")
        return Chunk(
            xr=np.stack([p.xr for p in parts]),
            xr_hat=np.stack([p.xr_hat for p in parts]),
            sideinfo=np.stack([p.sideinfo for p in parts]),
            offset=np.array([p.offset for p in parts]),
        )


def build_chunk_index(
    root: Path,
    chunk_granules: int,
    split: str | None = None,
    bitrates: list[int] | None = None,
) -> np.ndarray:
    """Every addressable chunk in the dataset, as a structured array.

    Fields: ``entry`` (index into `read_index(root)`), ``chunk``, ``bitrate_kbps``.
    Lets a Dataset's ``__getitem__(i)`` map straight to a (track, chunk) pair
    with no directory scanning.
    """
    entries = read_index(root)
    rows = []
    for e_i, e in enumerate(entries):
        if split is not None and e.split != split:
            continue
        if bitrates is not None and e.bitrate_kbps not in bitrates:
            continue
        for c in range(chunk_count(e.n_granules, chunk_granules)):
            rows.append((e_i, c, e.bitrate_kbps))
    return np.array(rows, dtype=[("entry", "<i4"), ("chunk", "<i4"), ("bitrate_kbps", "<i4")])


def write_index(root: Path, entries: list[TrackEntry]) -> Path:
    """Write `index.json` so a DataLoader needs no directory scan at startup."""
    path = root / "index.json"
    payload = {
        "n_entries": len(entries),
        "block_type_names": BLOCK_TYPE_NAMES,
        "sideinfo_dtype": [
            [name, str(SIDEINFO_DTYPE[name].base), list(SIDEINFO_DTYPE[name].shape)]
            for name in SIDEINFO_DTYPE.names
        ],
        "entries": [vars(e) for e in entries],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def read_index(root: Path) -> list[TrackEntry]:
    payload = json.loads((root / "index.json").read_text())
    return [TrackEntry(**e) for e in payload["entries"]]
