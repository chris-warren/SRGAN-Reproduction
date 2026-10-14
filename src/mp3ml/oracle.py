"""Run the native LAME dumper and load its granule records.

The dumper (``native/lame_dump/mp3dump``) encodes mono PCM with LAME and reads
two arrays straight out of LAME's analysis path:

``xr``      the encoder's pre-quantization coefficients (the clean target)
``xr_hat``  the decoder's dequantized coefficients (the degraded input)

Both are 576 values per granule per channel at the same point in the MP3 signal
chain — post-alias-reduction, pre-quantization — so they differ only by the
quantization and Huffman round trip. See ``native/lame_dump/README.md``.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Must match record_t in native/lame_dump/mp3dump.c field for field.
RECORD_DTYPE = np.dtype(
    [
        ("frame_num", "<i4"),
        ("granule", "<i4"),
        ("channel", "<i4"),
        ("block_type", "<i4"),  # encoder side
        ("dec_block_type", "<i4"),  # decoder side, from the bitstream
        ("mixed_block_flag", "<i4"),
        ("global_gain", "<i4"),  # decoder side, from the bitstream
        ("enc_global_gain", "<i4"),  # encoder side
        ("scalefac_scale", "<i4"),
        ("preflag", "<i4"),
        ("big_values", "<i4"),
        ("subblock_gain", "<i4", (3,)),
        ("xr", "<f4", (576,)),
        ("xr_hat", "<f4", (576,)),
    ]
)

BLOCK_TYPE_NAMES = {0: "long", 1: "start", 2: "short", 3: "stop"}

_REPO_ROOT = Path(__file__).resolve().parents[2]
DUMPER = _REPO_ROOT / "native" / "lame_dump" / "mp3dump"


@dataclass(frozen=True)
class GranuleSet:
    """Granule records for one (audio, bitrate) pair, plus the encoder's MP3."""

    records: np.ndarray
    meta: dict
    mp3_path: Path
    pcm_path: Path | None = None

    @property
    def xr(self) -> np.ndarray:
        """Clean targets, shape (n_records, 576) float32."""
        return self.records["xr"]

    @property
    def xr_hat(self) -> np.ndarray:
        """Degraded inputs, shape (n_records, 576) float32."""
        return self.records["xr_hat"]

    def block_type_counts(self) -> dict[str, int]:
        values, counts = np.unique(self.records["block_type"], return_counts=True)
        return {
            BLOCK_TYPE_NAMES.get(int(v), str(v)): int(c)
            for v, c in zip(values, counts, strict=True)
        }

    def save_npz(self, path: Path) -> None:
        fields = {name: self.records[name] for name in self.records.dtype.names}
        np.savez_compressed(path, meta=json.dumps(self.meta), **fields)


def dump_granules(
    pcm: np.ndarray,
    bitrate_kbps: int,
    out_prefix: Path,
    samplerate: int = 44100,
    analysis: bool = True,
    write_pcm: bool = False,
    allow_resample: bool = False,
    channels: int = 1,
) -> GranuleSet:
    """Encode mono int16 `pcm` at `bitrate_kbps` and return its granule records.

    Writes ``{out_prefix}.bin``, ``.json`` and ``.mp3``, plus ``.pcm`` (mpglib's
    own decode) when `write_pcm` is set. With ``analysis=False`` LAME's analysis
    path is off, so no records are produced — that mode exists only to compare
    the resulting bitstream against stock LAME's.
    """
    if not DUMPER.exists():
        raise FileNotFoundError(f"{DUMPER} not built — run native/lame_dump/build.sh")
    if pcm.dtype != np.int16:
        raise ValueError(f"expected int16 PCM, got {pcm.dtype}")
    if channels == 1 and pcm.ndim != 1:
        raise ValueError(f"expected mono PCM of shape (n,), got {pcm.shape}")
    if channels == 2 and (pcm.ndim != 2 or pcm.shape[1] != 2):
        raise ValueError(f"expected stereo PCM of shape (n, 2), got {pcm.shape}")

    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # stdout is discarded: this LAME tree carries a stray debug printf in
    # newmdct.c. All real output goes to the three files.
    cmd = [
        str(DUMPER),
        "--bitrate",
        str(bitrate_kbps),
        "--samplerate",
        str(samplerate),
        "--out-prefix",
        str(out_prefix),
        "--channels",
        str(channels),
    ]
    if not analysis:
        cmd.append("--no-analysis")
    if write_pcm:
        cmd.append("--pcm-out")

    proc = subprocess.run(
        cmd,
        input=np.ascontiguousarray(pcm, dtype="<i2").tobytes(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"mp3dump failed: {proc.stderr.decode(errors='replace').strip()}")
    logger.info("%s", proc.stderr.decode(errors="replace").strip())

    meta = json.loads(Path(f"{out_prefix}.json").read_text())
    if meta["record_bytes"] != RECORD_DTYPE.itemsize:
        raise RuntimeError(
            f"record size mismatch: mp3dump says {meta['record_bytes']}, "
            f"RECORD_DTYPE is {RECORD_DTYPE.itemsize} — the C struct and dtype "
            "have drifted apart"
        )
    records = np.fromfile(f"{out_prefix}.bin", dtype=RECORD_DTYPE)
    if len(records) != meta["records"]:
        raise RuntimeError(f"expected {meta['records']} records, read {len(records)}")

    # LAME silently resamples below 64 kbps CBR (48k -> 32 kHz) and at low VBR
    # qualities, and can switch to MPEG-2 LSF (576-sample frames, 1 granule).
    # xr then covers a different frequency range and no longer corresponds 1:1
    # to the input samples, so pairs from such a run must not be mixed into a
    # 44.1 kHz dataset without knowing it happened.
    if not allow_resample and meta["out_samplerate"] != samplerate:
        raise RuntimeError(
            f"LAME resampled {samplerate} Hz -> {meta['out_samplerate']} Hz at "
            f"{bitrate_kbps} kbps (framesize {meta['framesize']}, "
            f"mode_gr {meta['mode_gr']}). Stay at >=64 kbps CBR, or pass "
            "allow_resample=True and record out_samplerate with the data."
        )

    return GranuleSet(
        records=records,
        meta=meta,
        mp3_path=Path(f"{out_prefix}.mp3"),
        pcm_path=Path(f"{out_prefix}.pcm") if write_pcm else None,
    )


def check_alignment(records: np.ndarray) -> dict[str, float]:
    """Fraction of records whose encoder- and decoder-side side info agree.

    The encoder fields (``block_type``, ``enc_global_gain``) and the decoder
    fields (``dec_block_type``, ``global_gain``) are written by different code
    paths at different times and describe the same granule. Anything below 1.0
    means the frame join is misaligned and the xr/xr_hat pairing is invalid.
    """
    return {
        "block_type": float((records["block_type"] == records["dec_block_type"]).mean()),
        "global_gain": float((records["enc_global_gain"] == records["global_gain"]).mean()),
    }
