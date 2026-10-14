"""Checks on metric logging and the FAD distance.

The tracking contract that matters: **JSONL is written whether or not a backend
works**. A run that loses an hour of GPU time because a logging service refused
a connection is the failure mode these tests exist to prevent.
"""

import os
from pathlib import Path

import pytest
import torch

from mp3ml.config import load_config, single_bitrate
from mp3ml.metrics import frechet_distance
from mp3ml.tracking import Tracker, read_jsonl

CFG = load_config(Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml")


# ---------------------------------------------------------------- tracking


def test_jsonl_is_written_without_any_backend(tmp_path: Path) -> None:
    t = Tracker(tmp_path / "run", CFG, backend="none")
    t.log({"d_loss": 1.5, "gp": 0.2}, step=1)
    t.log({"d_loss": 1.0, "gp": 0.1}, step=2)
    t.finish()

    rows = read_jsonl(tmp_path / "run")
    assert [r["step"] for r in rows] == [1, 2]
    assert rows[0]["d_loss"] == 1.5
    assert all(r["phase"] == "train" for r in rows)


def test_phases_are_recorded_and_filterable(tmp_path: Path) -> None:
    """Train, val and test curves must be separable offline — that is what
    makes the JSONL a usable record rather than a dump."""
    t = Tracker(tmp_path / "run", CFG)
    t.log({"lsd": 5.0}, step=1, phase="train")
    t.log({"lsd": 6.0}, step=1, phase="val")
    t.log({"lsd": 7.0}, step=1, phase="test")

    assert len(read_jsonl(tmp_path / "run")) == 3
    assert [r["lsd"] for r in read_jsonl(tmp_path / "run", phase="val")] == [6.0]


def test_tensors_are_reduced_to_plain_scalars(tmp_path: Path) -> None:
    """JSON cannot hold a tensor; a silent crash here would kill a run."""
    t = Tracker(tmp_path / "run", CFG)
    t.log({"a": torch.tensor(2.0), "b": torch.tensor([1.0, 3.0])}, step=1)
    row = read_jsonl(tmp_path / "run")[0]
    assert row["a"] == pytest.approx(2.0)
    assert row["b"] == pytest.approx(2.0), "multi-element tensors are averaged"


def test_a_broken_backend_degrades_to_jsonl(tmp_path: Path, monkeypatch) -> None:
    """The whole point of JSONL-first. Simulates wandb.init raising."""
    import wandb

    def boom(*_args, **_kwargs):
        raise RuntimeError("no network")

    monkeypatch.setattr(wandb, "init", boom)
    t = Tracker(tmp_path / "run", CFG, backend="wandb")
    assert t.backend == "none", "must fall back, not raise"
    t.log({"d_loss": 1.0}, step=1)
    assert len(read_jsonl(tmp_path / "run")) == 1


def test_unknown_backend_is_rejected_early(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        Tracker(tmp_path / "run", CFG, backend="mlflow")


def test_audio_is_written_to_disk_regardless_of_backend(tmp_path: Path) -> None:
    import soundfile as sf

    t = Tracker(tmp_path / "run", CFG)
    t.log_audio("restored", torch.randn(4000) * 0.1, samplerate=16000, step=7)
    out = tmp_path / "run" / "audio" / "step000007_restored.wav"
    assert out.exists()
    assert len(sf.read(out)[0]) == 4000


# -------------------------------------------------------------------- FAD


def test_frechet_distance_is_zero_for_identical_sets() -> None:
    torch.manual_seed(0)
    x = torch.randn(200, 16)
    assert frechet_distance(x, x).item() == pytest.approx(0.0, abs=1e-6)


def test_frechet_distance_matches_the_closed_form_in_one_dimension() -> None:
    """For 1-D Gaussians the distance is `(m1-m2)^2 + (s1-s2)^2` in std units.
    Built from exact moments so the check is arithmetic, not statistical."""
    a = torch.tensor([[-1.0], [1.0]])  # mean 0, var 2 (torch.cov uses n-1)
    b = torch.tensor([[4.0], [8.0]])  # mean 6, var 8
    want = (0.0 - 6.0) ** 2 + (2.0**0.5 - 8.0**0.5) ** 2
    assert frechet_distance(a, b).item() == pytest.approx(want, rel=1e-4)


def test_frechet_distance_grows_as_distributions_separate() -> None:
    torch.manual_seed(0)
    ref = torch.randn(300, 8)
    near = torch.randn(300, 8) + 0.1
    far = torch.randn(300, 8) + 2.0
    assert frechet_distance(ref, near) < frechet_distance(ref, far)


def test_frechet_distance_is_symmetric() -> None:
    torch.manual_seed(0)
    a, b = torch.randn(150, 8), torch.randn(150, 8) * 1.5 + 0.3
    assert frechet_distance(a, b).item() == pytest.approx(frechet_distance(b, a).item(), rel=1e-6)


def test_frechet_distance_rejects_unusable_input() -> None:
    with pytest.raises(ValueError, match="embeddings"):
        frechet_distance(torch.randn(10), torch.randn(10))
    with pytest.raises(ValueError, match="at least 2 samples"):
        frechet_distance(torch.randn(1, 8), torch.randn(5, 8))


# ------------------------------------------------------------- per-bitrate


def test_single_bitrate_narrows_and_renames() -> None:
    """§2.10: the paper trains one model per rate. The rename keeps run
    directories from colliding across the three runs."""
    cfg = single_bitrate(CFG, 32)
    assert cfg.data.bitrates == (32,)
    assert cfg.name == "smoke_32k"
    assert CFG.data.bitrates == (16, 32, 64), "must not mutate the original"


def test_single_bitrate_rejects_a_rate_the_config_lacks() -> None:
    with pytest.raises(ValueError, match="not in this config"):
        single_bitrate(CFG, 128)


# ------------------------------------------------------------- .env loading


def test_env_file_is_loaded_but_never_overrides_the_environment(tmp_path, monkeypatch) -> None:
    """`.env` holds the W&B key. Values already exported must win, so a shell
    export still overrides the file and CI cannot pick up a developer's key."""
    from mp3ml.tracking import load_env_file

    (tmp_path / ".env").write_text(
        "# a comment\n\nWANDB_API_KEY=from-file\nALREADY_SET=from-file\nMALFORMED\n"
    )
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("ALREADY_SET", "from-shell")

    applied = load_env_file(tmp_path)

    assert os.environ["WANDB_API_KEY"] == "from-file"
    assert os.environ["ALREADY_SET"] == "from-shell", "the file must not override a real export"
    assert "WANDB_API_KEY" in applied and "ALREADY_SET" not in applied


def test_env_file_is_found_from_a_subdirectory(tmp_path, monkeypatch) -> None:
    """Runs are often launched from elsewhere in the tree."""
    from mp3ml.tracking import load_env_file

    (tmp_path / ".env").write_text("DEEP_KEY=found\n")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    monkeypatch.delenv("DEEP_KEY", raising=False)
    assert load_env_file(nested) == ["DEEP_KEY"]


def test_missing_env_file_is_not_an_error(tmp_path) -> None:
    from mp3ml.tracking import load_env_file

    assert load_env_file(tmp_path / "nowhere") == []


def test_env_loader_returns_names_not_values(tmp_path, monkeypatch) -> None:
    """It handles a secret — the return value is logged, so it must not carry
    the key itself."""
    from mp3ml.tracking import load_env_file

    (tmp_path / ".env").write_text("WANDB_API_KEY=supersecret\n")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    applied = load_env_file(tmp_path)
    assert applied == ["WANDB_API_KEY"]
    assert "supersecret" not in str(applied)
