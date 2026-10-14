"""Checks on the on-disk dataset layout. Synthetic audio only."""

import numpy as np
import pytest

from mp3ml.dataset import (
    SIDEINFO_DTYPE,
    TrackEntry,
    TrackReader,
    build_chunk_index,
    chunk_count,
    load_track,
    read_index,
    write_index,
    write_track,
)
from mp3ml.oracle import DUMPER, dump_granules

pytestmark = pytest.mark.skipif(
    not DUMPER.exists(), reason="mp3dump not built — run native/lame_dump/build.sh"
)

SAMPLERATE = 44100


@pytest.fixture(scope="module")
def stereo_tone() -> np.ndarray:
    """Half a second of stereo audio with different content per channel."""
    t = np.arange(SAMPLERATE // 2) / SAMPLERATE
    left = 0.3 * np.sin(2 * np.pi * 440 * t)
    right = 0.3 * np.sin(2 * np.pi * 660 * t)
    return np.round(np.stack([left, right], axis=1) * 32767).astype(np.int16)


@pytest.fixture(scope="module")
def built(stereo_tone, tmp_path_factory):
    out = tmp_path_factory.mktemp("ds") / "bitrate=128" / "tone"
    meta = write_track(stereo_tone, out, bitrate_kbps=128, channels=2)
    return out, meta


def test_files_written(built) -> None:
    out, _ = built
    for name in ("xr.npy", "xr_hat.npy", "sideinfo.npy", "audio.mp3", "meta.json"):
        assert (out / name).exists(), name


def test_arrays_memmap_with_expected_shape(built) -> None:
    out, meta = built
    xr, xr_hat, side, _ = load_track(out)
    assert xr.shape == (meta["n_granules"], 2, 576)
    assert xr_hat.shape == xr.shape
    assert xr.dtype == np.float32
    assert side.shape == (meta["n_granules"], 2)
    assert side.dtype == SIDEINFO_DTYPE
    assert isinstance(xr, np.memmap)


def test_channels_are_not_duplicated(built) -> None:
    """The two channels carry different audio, so their coefficients must differ.

    A reshape bug that collapsed the channel axis would silently make them equal.
    """
    out, _ = built
    xr, _, _, _ = load_track(out)
    assert not np.array_equal(xr[:, 0, :], xr[:, 1, :])


def test_stored_arrays_match_a_fresh_dump(stereo_tone, built, tmp_path) -> None:
    out, _ = built
    xr, xr_hat, _, _ = load_track(out)
    fresh = dump_granules(stereo_tone, 128, tmp_path / "fresh", channels=2)
    np.testing.assert_array_equal(np.asarray(xr), fresh.records["xr"].reshape(-1, 2, 576))
    np.testing.assert_array_equal(np.asarray(xr_hat), fresh.records["xr_hat"].reshape(-1, 2, 576))


def test_meta_records_alignment_and_preprocessing(built) -> None:
    _, meta = built
    assert meta["alignment"] == {"block_type": 1.0, "global_gain": 1.0}
    assert meta["out_samplerate"] == SAMPLERATE
    assert meta["channels"] == 2
    assert "input_scale" in meta and "lowpass_hz" in meta


def test_random_offset_chunking(built) -> None:
    """Chunk length is a read-time choice, not baked into the files."""
    out, _ = built
    xr, xr_hat, _, _ = load_track(out)
    n = xr.shape[0]
    for chunk in (1, 8, 32):
        assert n >= chunk, f"fixture too short for chunk {chunk}"
        offset = min(n // 3, n - chunk)
        assert np.asarray(xr[offset : offset + chunk]).shape == (chunk, 2, 576)
        assert np.asarray(xr_hat[offset : offset + chunk]).shape == (chunk, 2, 576)


def test_index_round_trip(tmp_path) -> None:
    entries = [
        TrackEntry("t1", 128, 2, 100, 57600, "train", "bitrate=128/t1", "/src/t1.wav", 1.3),
        TrackEntry("t2", 320, 2, 200, 115200, "val", "bitrate=320/t2", "/src/t2.wav", 2.6),
    ]
    write_index(tmp_path, entries)
    assert read_index(tmp_path) == entries


def test_misaligned_pairs_are_refused(monkeypatch, stereo_tone, tmp_path) -> None:
    """A granule-join failure must abort the write, not produce a bad track."""
    import mp3ml.dataset as ds

    monkeypatch.setattr(ds, "check_alignment", lambda r: {"block_type": 0.9, "global_gain": 1.0})
    with pytest.raises(RuntimeError, match="misalignment"):
        write_track(stereo_tone, tmp_path / "bad", bitrate_kbps=128, channels=2)


def test_chunk_addressing(built) -> None:
    """Explicit chunk numbers, and chunk i == the matching window."""
    out, _ = built
    r = TrackReader(out, chunk_granules=8)
    assert r.n_chunks == r.n_granules // 8

    c = r.chunk(2)
    assert c.xr.shape == (8, 2, 576)
    assert c.offset == 16
    np.testing.assert_array_equal(c.xr, r.window(16, 8).xr)


def test_chunks_are_contiguous_and_ordered(built) -> None:
    """chunk(i) must be exactly the granules a full read would give at that offset."""
    out, _ = built
    r = TrackReader(out, chunk_granules=8)
    xr, _, _, _ = load_track(out)
    for i in (0, 1, r.n_chunks - 1):
        np.testing.assert_array_equal(r.chunk(i).xr, np.asarray(xr[i * 8 : (i + 1) * 8]))


def test_multiple_chunks_stack_in_the_order_given(built) -> None:
    out, _ = built
    r = TrackReader(out, chunk_granules=8)
    ids = [3, 0, 1]
    got = r.chunks(ids)
    assert got.xr.shape == (3, 8, 2, 576)
    np.testing.assert_array_equal(got.offset, [24, 0, 8])
    for k, i in enumerate(ids):
        np.testing.assert_array_equal(got.xr[k], r.chunk(i).xr)


def test_chunk_length_is_a_read_time_choice(built) -> None:
    """The same files serve any chunk length — nothing is baked in."""
    out, _ = built
    for length in (1, 4, 16):
        r = TrackReader(out, chunk_granules=length)
        assert r.chunk(0).xr.shape == (length, 2, 576)


def test_out_of_range_chunk_raises(built) -> None:
    out, _ = built
    r = TrackReader(out, chunk_granules=8)
    with pytest.raises(IndexError):
        r.chunk(r.n_chunks)
    with pytest.raises(IndexError):
        r.window(r.n_granules - 2, 8)


def test_chunk_count_drops_partial_tail() -> None:
    assert chunk_count(100, 8) == 12  # 96 granules used, 4 dropped
    assert chunk_count(7, 8) == 0


def test_chunk_index_addresses_every_chunk(built, tmp_path) -> None:
    """build_chunk_index is the documented DataLoader entry point, so it must be exact."""
    out, meta = built
    root = out.parent.parent
    entry = TrackEntry(
        track="tone",
        bitrate_kbps=128,
        channels=2,
        n_granules=meta["n_granules"],
        n_samples=meta["input_samples"],
        split="train",
        path=str(out.relative_to(root)),
        source="synthetic",
        duration_s=0.5,
    )
    write_index(root, [entry])

    idx = build_chunk_index(root, chunk_granules=8)
    assert len(idx) == chunk_count(meta["n_granules"], 8)
    assert set(idx.dtype.names) == {"entry", "chunk", "bitrate_kbps"}
    assert (idx["entry"] == 0).all()
    assert list(idx["chunk"]) == list(range(len(idx)))

    # every indexed chunk must actually be readable
    reader = TrackReader(root / entry.path, chunk_granules=8)
    for row in (idx[0], idx[len(idx) // 2], idx[-1]):
        assert reader.chunk(int(row["chunk"])).xr.shape == (8, 2, 576)


def test_chunk_index_filters(built, tmp_path) -> None:
    out, meta = built
    root = out.parent.parent
    write_index(
        root,
        [
            TrackEntry(
                track="tone",
                bitrate_kbps=128,
                channels=2,
                n_granules=meta["n_granules"],
                n_samples=meta["input_samples"],
                split="val",
                path=str(out.relative_to(root)),
                source="synthetic",
                duration_s=0.5,
            )
        ],
    )
    assert len(build_chunk_index(root, 8, split="train")) == 0
    assert len(build_chunk_index(root, 8, split="val")) > 0
    assert len(build_chunk_index(root, 8, bitrates=[320])) == 0
    assert len(build_chunk_index(root, 8, bitrates=[128])) > 0
