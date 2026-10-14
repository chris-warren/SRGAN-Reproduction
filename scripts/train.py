"""Train the GAN reproduction.

python scripts/train.py configs/local.yaml
python scripts/train.py configs/local.yaml --resume          # newest ckpt
python scripts/train.py configs/smoke.yaml --max-steps 4     # plumbing only
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from mp3ml.config import deterministic, fits_in, load_config, single_bitrate
from mp3ml.tracking import Tracker
from mp3ml.train import Trainer, latest_checkpoint


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("--run-dir", type=Path, help="default: outputs/<config name>")
    p.add_argument("--resume", action="store_true", help="continue from the newest checkpoint")
    p.add_argument("--max-steps", type=int, help="stop early; defaults to train.iterations")
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--checkpoint-every", type=int, default=1000)
    p.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="run a small validation pass every N steps and log it as phase=val. "
        "Training losses do not tell you whether the audio is improving; these do. "
        "0 disables.",
    )
    p.add_argument("--vram-gb", type=float, default=8.0)
    p.add_argument(
        "--bitrate",
        type=int,
        choices=(16, 32, 64),
        help="train on ONE bitrate, as the paper does (REPRODUCTION.md 1.3). "
        "Without this the run mixes every bitrate in the config, which is a "
        "different experiment and not comparable to Table 2.",
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="train WITHOUT noise input, for Table 2's det_* rows (REPRODUCTION.md 1.3). "
        "A separately trained network, not a stochastic model run at z=0.",
    )
    p.add_argument("--tracker", default="none", choices=("none", "wandb"))
    p.add_argument("--project", default="mp3-restoration", help="W&B project name")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)
    if args.bitrate is not None:
        cfg = single_bitrate(cfg, args.bitrate)
    if args.deterministic:
        cfg = deterministic(cfg)
    elif len(cfg.data.bitrates) > 1:
        print(
            f"NOTE: training on mixed bitrates {list(cfg.data.bitrates)}. The paper "
            "trains one model per bitrate (REPRODUCTION.md 1.3); pass --bitrate "
            "for a Table 2 comparison."
        )
    run_dir = args.run_dir or Path("outputs") / cfg.name

    # A lower bound only (see config.estimate_size) — it catches the obviously
    # oversized config, not the marginal one.
    ok, reason = fits_in(cfg, args.vram_gb)
    print(f"config {cfg.name}: {reason}")
    if not ok:
        print("  ^ over budget by the lower-bound estimate; a real run will be worse")

    tracker = Tracker(run_dir, cfg, backend=args.tracker, project=args.project)
    trainer = Trainer(cfg, run_dir, device=args.device, tracker=tracker)
    if args.resume:
        ckpt = latest_checkpoint(run_dir)
        if ckpt is None:
            print(f"no checkpoint under {run_dir}/checkpoints — starting fresh")
        else:
            trainer.load(ckpt)

    trainer.fit(
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        max_steps=args.max_steps,
        num_workers=args.workers,
        eval_every=args.eval_every,
    )
    print(f"done at step {trainer.step}; checkpoints in {run_dir / 'checkpoints'}")


if __name__ == "__main__":
    main()
