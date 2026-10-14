"""Metric logging: JSONL always, W&B optionally on top.

**JSONL is the record; the backend is a viewer.** Every scalar goes to
`outputs/<run>/metrics.jsonl` first, one JSON object per line, whether or not a
backend is configured. That way history survives a backend swap, an expired
account or an offline cloud box, and can be re-rendered into anything later —
which matters because this project will accumulate a theta sweep plus six
models (3 bitrates x {stochastic, deterministic}).

The backend is chosen by name so nothing else in the codebase imports `wandb`:

    tracker = Tracker(run_dir, cfg, backend="wandb")
    tracker.log({"d_loss": 1.2}, step=10, phase="train")
    tracker.log_audio("restored", wav, samplerate=44100, step=10)
    tracker.finish()

A missing or broken backend **degrades to JSONL with a warning** rather than
killing a run — losing an hour of GPU time because a logging service refused a
connection is not a trade worth making.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import ExperimentConfig, config_to_dict

logger = logging.getLogger(__name__)

BACKENDS = ("none", "wandb")

#: Searched upward from the working directory, so a run started from a
#: subdirectory still finds it.
ENV_FILE = ".env"


def load_env_file(start: Path | None = None) -> list[str]:
    """Read `.env` into `os.environ`, without adding a dependency for one key.

    **Values already in the environment win**, so `export WANDB_API_KEY=...`
    still overrides the file and CI never picks up a developer's key by
    accident. Returns the names it set, never the values — this handles a
    secret, so nothing here should end up in a log.
    """
    here = (start or Path.cwd()).resolve()
    for folder in (here, *here.parents):
        path = folder / ENV_FILE
        if not path.is_file():
            continue
        applied = []
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("\"'")
            if key and value and key not in os.environ:
                os.environ[key] = value
                applied.append(key)
        return applied
    return []


class Tracker:
    """Writes metrics to JSONL, and mirrors them to a backend if one is set."""

    def __init__(
        self,
        run_dir: Path,
        cfg: ExperimentConfig,
        backend: str = "none",
        project: str = "mp3-restoration",
        group: str | None = None,
    ) -> None:
        if backend not in BACKENDS:
            raise ValueError(f"unknown backend {backend!r}; choose from {BACKENDS}")
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "metrics.jsonl"
        self.cfg = cfg
        self.backend = backend
        self._wandb = None
        self._started = time.time()

        if backend == "wandb":
            self._wandb = self._init_wandb(project, group)
            if self._wandb is None:
                self.backend = "none"

    def _init_wandb(self, project: str, group: str | None):
        applied = load_env_file()
        if applied:
            logger.info("loaded %s from %s", ", ".join(sorted(applied)), ENV_FILE)
        if not os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_MODE") != "offline":
            logger.warning(
                "no WANDB_API_KEY (checked the environment and %s) — logging to %s only. "
                "Add the key from https://wandb.ai/authorize, or set WANDB_MODE=offline "
                "to record locally and `wandb sync` later.",
                ENV_FILE,
                self.path,
            )
            return None
        try:
            import wandb
        except ImportError:
            logger.warning(
                "wandb not installed — logging to %s only. `pip install -e '.[tracking]'`",
                self.path,
            )
            return None
        try:
            return wandb.init(
                project=project,
                name=self.run_dir.name,
                group=group,
                config=config_to_dict(self.cfg),
                dir=str(self.run_dir),
            )
        except Exception as exc:  # noqa: BLE001 - any failure here must not end the run
            logger.warning("wandb init failed (%s) — logging to %s only", exc, self.path)
            return None

    # ------------------------------------------------------------------ log

    def log(self, metrics: dict[str, Any], step: int, phase: str = "train") -> None:
        """Record one row. `phase` is 'train', 'val' or 'test'.

        Keys are namespaced as `<phase>/<key>` for the backend, so train and val
        curves for the same metric land on one chart, but kept flat in the JSONL
        alongside an explicit `phase` field, which is easier to filter offline.
        """
        clean = {k: _scalar(v) for k, v in metrics.items()}
        row = {
            "step": step,
            "phase": phase,
            "wall_time": round(time.time() - self._started, 3),
            **clean,
        }
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")

        if self._wandb is not None:
            self._wandb.log({f"{phase}/{k}": v for k, v in clean.items()}, step=step)

    def log_audio(
        self, name: str, wav: torch.Tensor | np.ndarray, samplerate: int, step: int
    ) -> None:
        """Log an audio example. **The reason a backend is worth having at all**:
        for a restoration model, listening tells you far more than LSD does.

        Written to `<run>/audio/` regardless, so examples survive without a
        backend.
        """
        data = _to_numpy(wav)
        out = self.run_dir / "audio" / f"step{step:06d}_{name}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            import soundfile as sf

            sf.write(out, data, samplerate, subtype="PCM_16")
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not write %s: %s", out, exc)

        if self._wandb is not None:
            import wandb

            self._wandb.log({f"audio/{name}": wandb.Audio(data, sample_rate=samplerate)}, step=step)

    def log_table(self, name: str, report_table: str, step: int) -> None:
        """Log a rendered text table (e.g. `EvalReport.table()`) as an artifact."""
        out = self.run_dir / f"{name}_step{step:06d}.txt"
        out.write_text(report_table)
        if self._wandb is not None:
            self._wandb.summary[f"{name}_latest"] = report_table

    def finish(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()


def _scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else value.mean().item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _to_numpy(wav: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().numpy()
    return np.asarray(wav, dtype=np.float32).squeeze()


def read_jsonl(run_dir: Path, phase: str | None = None) -> list[dict]:
    """Read a run's metric history back. The point of writing JSONL: plotting,
    re-export to another backend, and cross-run comparison all work offline."""
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in rows if phase is None or r.get("phase") == phase]
