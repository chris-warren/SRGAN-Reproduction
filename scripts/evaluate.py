"""Score a checkpoint with the paper's §4.1 protocol.

    python scripts/evaluate.py outputs/local/checkpoints/final.pt
    python scripts/evaluate.py CKPT --split val --samples 20 --max-items 64
    python scripts/evaluate.py CKPT --json outputs/local/eval.json

Reports the model against the MP3 baseline on the same excerpts, which is the
comparison Table 2 makes. Remember the protocol takes the **best of N draws per
metric**, so a `--samples 1` run is not comparable to the paper.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from mp3ml.config import ExperimentConfig, load_config
from mp3ml.evaluate import PAPER_N_SAMPLES, evaluate
from mp3ml.models import Generator


def _config_for(ckpt_path: Path, override: Path | None) -> ExperimentConfig:
    """Prefer the config the checkpoint was trained with, not whatever is on disk now."""
    if override is not None:
        return load_config(override)
    beside = ckpt_path.parent.parent / "config.json"
    if not beside.exists():
        raise SystemExit(
            f"no config.json beside {ckpt_path} and no --config given; refusing to "
            "guess an architecture for a checkpoint"
        )
    import tempfile

    import yaml

    raw = json.loads(beside.read_text())
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(raw, f)
        return load_config(Path(f.name))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--config", type=Path, help="override the run's own config.json")
    p.add_argument("--split", default="val", choices=("val", "test"))
    p.add_argument(
        "--data-root",
        type=Path,
        help="override the corpus path in the config (also settable with "
        "$APOLLO_DATA). The YAML points at a workstation drive that will not "
        "exist on a cluster.",
    )
    p.add_argument(
        "--bitrate",
        type=int,
        help="which rate's clips to score. Only 64 is usable from this corpus "
        "(see mp3ml.apollo); defaults to the config's first rate.",
    )
    p.add_argument(
        "--aligned-only",
        action="store_true",
        help="drop the clips Apollo's codec_simu left misaligned instead of "
        "correcting them from the manifest's recorded lag (the default)",
    )
    p.add_argument("--samples", type=int, default=PAPER_N_SAMPLES, help="draws per input (§4.1)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-items", type=int, help="cap for a quick look")
    p.add_argument("--device", default="cuda")
    p.add_argument("--json", type=Path, help="write the full report here")
    p.add_argument("--no-peaq", action="store_true", help="skip ODG/DI even if peaqb is present")
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for the noise draws, so a reported number reproduces exactly. "
        "Use -1 for independent sampling.",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = _config_for(args.checkpoint, args.config)
    if args.data_root is not None:
        import dataclasses

        cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, root=args.data_root))
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    g = Generator(cfg)
    g.load_state_dict(ckpt["generator"])
    print(f"{args.checkpoint} — step {ckpt.get('step', '?')}, config {cfg.name}")

    report = evaluate(
        g,
        cfg,
        split=args.split,
        n_samples=args.samples,
        batch_size=args.batch_size,
        device=args.device,
        max_items=args.max_items,
        with_peaq=False if args.no_peaq else None,
        bitrate=args.bitrate,
        aligned_only=args.aligned_only,
        seed=None if args.seed < 0 else args.seed,
    )

    print(f"\n{args.split}: {report.n_items} excerpts x {report.n_samples} draws\n")
    print(report.table())
    for note in report.skipped:
        print(f"\nskipped: {note}")
    if report.n_samples == 1:
        print("\nNOTE: 1 draw — not comparable to the paper's best-of-20 Table 2.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "checkpoint": str(args.checkpoint),
                    "step": ckpt.get("step"),
                    "config": cfg.name,
                    "split": args.split,
                    "n_items": report.n_items,
                    "n_samples": report.n_samples,
                    "means": report.means(),
                    "per_item": {"model": dict(report.model), "mp3": dict(report.baseline)},
                    "skipped": report.skipped,
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
