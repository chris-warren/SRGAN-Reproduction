"""Fast checks on the LAME oracle. Synthetic audio only — no corpus, no network."""

import numpy as np
import pytest

from mp3ml.oracle import RECORD_DTYPE, check_alignment, dump_granules

pytestmark = pytest.mark.skipif(
    not __import__("mp3ml.oracle", fromlist=["DUMPER"]).DUMPER.exists(),
    reason="mp3dump not built — run native/lame_dump/build.sh",
)

SAMPLERATE = 44100


@pytest.fixture(scope="module")
def tone() -> np.ndarray:
    t = np.arange(SAMPLERATE // 2) / SAMPLERATE
    x = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.15 * np.sin(2 * np.pi * 3011 * t)
    return np.round(x * 32767).astype(np.int16)


@pytest.fixture(scope="module")
def granules(tone: np.ndarray, tmp_path_factory: pytest.TempPathFactory):
    out = tmp_path_factory.mktemp("oracle") / "tone"
    return dump_granules(tone, bitrate_kbps=320, out_prefix=out, samplerate=SAMPLERATE)


def test_record_dtype_matches_c_struct(granules) -> None:
    assert granules.meta["record_bytes"] == RECORD_DTYPE.itemsize


def test_shapes_and_finiteness(granules) -> None:
    assert granules.xr.shape[1] == 576
    assert granules.xr_hat.shape == granules.xr.shape
    assert len(granules.records) == granules.meta["mode_gr"] * granules.meta["emitted_frames"]
    assert np.isfinite(granules.xr).all()
    assert np.isfinite(granules.xr_hat).all()


def test_two_granules_per_frame(granules) -> None:
    assert granules.meta["mode_gr"] == 2
    for frame in np.unique(granules.records["frame_num"]):
        assert sorted(granules.records["granule"][granules.records["frame_num"] == frame]) == [0, 1]


def test_side_info_alignment_is_exact(granules) -> None:
    """Encoder- and decoder-side side info must agree for every granule.

    These fields are written by different code paths at different times, so
    anything below 1.0 means the frame join is off and the pairs are invalid.
    """
    assert check_alignment(granules.records) == {"block_type": 1.0, "global_gain": 1.0}


def test_degradation_is_small_but_nonzero_at_320k(granules) -> None:
    xr = granules.xr.astype(np.float64)
    xr_hat = granules.xr_hat.astype(np.float64)
    rms = np.linalg.norm(xr - xr_hat) / np.linalg.norm(xr)
    assert 0.0 < rms < 1e-2, f"unexpected degradation at 320 kbps: {rms:.3e}"


def test_xr_and_xr_hat_share_a_scale(granules) -> None:
    """The pair must need no rescaling — they are the same representation."""
    xr = granules.xr.astype(np.float64)
    xr_hat = granules.xr_hat.astype(np.float64)
    scale = (xr * xr_hat).sum() / (xr_hat * xr_hat).sum()
    assert abs(scale - 1.0) < 1e-3, f"best-fit scale {scale:.6f} is not ~1"


def test_metadata_records_lame_preprocessing(granules) -> None:
    """LAME's per-bitrate input scale and lowpass must be recorded.

    xr is the transform of LAME's pre-processed input, not of the pristine
    original, so every dump has to carry these or cross-bitrate comparisons are
    silently wrong (0.95x at <=160 kbps, 0.97x at 192, 1.0x at 320).
    """
    assert granules.meta["input_scale"] == pytest.approx(1.0)  # 320 kbps
    assert granules.meta["lowpass_hz"] == 20500
    assert granules.meta["analysis"] == 1


def test_input_scale_varies_with_bitrate(tone: np.ndarray, tmp_path) -> None:
    scales = {
        br: dump_granules(tone, br, tmp_path / f"b{br}", SAMPLERATE).meta["input_scale"]
        for br in (320, 192, 128)
    }
    assert scales[320] == pytest.approx(1.00, abs=1e-6)
    assert scales[192] == pytest.approx(0.97, abs=1e-6)
    assert scales[128] == pytest.approx(0.95, abs=1e-6)


def test_no_resampling_at_supported_bitrates(tone: np.ndarray, tmp_path) -> None:
    """CBR at >=64 kbps must keep 44.1 kHz, 1152-sample frames, 2 granules."""
    for br in (320, 192, 128, 64):
        meta = dump_granules(tone, br, tmp_path / f"r{br}", SAMPLERATE).meta
        assert meta["out_samplerate"] == SAMPLERATE, br
        assert meta["framesize"] == 1152, br
        assert meta["mode_gr"] == 2, br


def test_resampling_is_refused_not_silent(tone: np.ndarray, tmp_path) -> None:
    """Below 64 kbps LAME drops the sample rate; that must fail loudly.

    At 48 kbps LAME resamples to 32 kHz, so xr covers a different frequency
    range and stops corresponding 1:1 to the input samples. Mixing such pairs
    into a 44.1 kHz dataset unnoticed would silently corrupt it.
    """
    with pytest.raises(RuntimeError, match="resampled"):
        dump_granules(tone, 48, tmp_path / "r48", SAMPLERATE)


def test_rejects_stereo_input(tone: np.ndarray, tmp_path) -> None:
    with pytest.raises(ValueError, match="mono"):
        dump_granules(np.stack([tone, tone]), 320, tmp_path / "x", SAMPLERATE)


def test_rejects_float_input(tmp_path) -> None:
    with pytest.raises(ValueError, match="int16"):
        dump_granules(np.zeros(1152, dtype=np.float32), 320, tmp_path / "x", SAMPLERATE)
