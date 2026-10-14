#!/usr/bin/env python
"""Report a config's size and whether it fits a VRAM budget, without needing torch.

    python scripts/check_config.py                        # every config
    python scripts/check_config.py configs/local.yaml --vram 8

Exists because the faithful architecture is ~649 M parameters, dominated by the
frequency-collapsing Conv4, and will not train on an 8 GB card. Checking a config
before committing to a run is cheaper than discovering it at allocation time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mp3ml.config import estimate_size, fits_in, load_config


def measure_peak(cfg) -> float | None:
    """Run one real critic + generator step and report peak GB, or None on OOM.

    The generator runs twice, as the paper does (section 3.2.1).
    """
    import gc

    import torch

    from mp3ml.config import target_crop
    from mp3ml.models import Critic, Generator

    if not torch.cuda.is_available():
        return None
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    try:
        g, d = Generator(cfg).cuda(), Critic(cfg).cuda()
        og = torch.optim.Adam(g.parameters(), lr=cfg.train.lr_g)
        od = torch.optim.Adam(d.parameters(), lr=cfg.train.lr_d)
        b, bins, frames = cfg.train.batch_size, cfg.stft.n_bins, cfg.data.segment_frames
        lo, hi = target_crop(cfg)
        y = torch.randn(b, 2, bins, frames, device="cuda")
        real = torch.randn(b, 2, bins, hi - lo, device="cuda")
        cond = y[..., lo:hi]
        loss_d = d(g(y).detach(), cond).mean() - d(real, cond).mean()
        od.zero_grad()
        loss_d.backward()
        od.step()
        loss_g = -(d(g(y), cond).mean() + d(g(y), cond).mean()) / 2
        og.zero_grad()
        loss_g.backward()
        og.step()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1e9
    except torch.OutOfMemoryError:
        return None
    finally:
        torch.cuda.empty_cache()
        gc.collect()


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("configs", nargs="*", type=Path, help="defaults to configs/*.yaml")
    p.add_argument("--vram", type=float, default=8.0, help="budget in GB")
    p.add_argument(
        "--measure",
        action="store_true",
        help="actually run a critic+generator step on the GPU and report "
        "real peak memory (the analytic figure is a lower bound)",
    )
    args = p.parse_args()

    paths = args.configs or sorted(Path("configs").glob("*.yaml"))
    if not paths:
        print("no configs found", file=sys.stderr)
        return 1

    failed = False
    for path in paths:
        cfg = load_config(path)
        est = estimate_size(cfg)
        ok, reason = fits_in(cfg, args.vram)
        print(f"\n{path}  [{cfg.name}]")
        print(
            f"  stft            n_fft {cfg.stft.n_fft}, {cfg.stft.n_bins} bins, "
            f"hop {cfg.stft.hop}, {cfg.data.segment_frames} frames"
        )
        print(f"  generator       {est['generator_params'] / 1e6:8.1f} M params")
        print(
            f"  critic          {est['critic_params'] / 1e6:8.1f} M params "
            f"(group factor {cfg.model.critic_group_factor})"
        )
        print(f"  total           {est['total_params'] / 1e6:8.1f} M params")
        print(
            f"    of which Conv4 {est['conv4_params'] / 1e6:7.1f} M "
            f"(collapses {cfg.stft.n_bins} bins -> 1)"
        )
        print(f"  weights fp32    {est['weights_gb']:8.2f} GB")
        print(f"  + grads + Adam  {est['adam_state_gb']:8.2f} GB")
        print(
            f"  activations     {est['activations_per_sample_mb']:8.1f} MB/sample, "
            f"{est['activations_batch_gb']:.2f} GB at batch {cfg.train.batch_size}"
        )
        print(f"  analytic lower bound: {'under' if ok else 'OVER'} {args.vram:g} GB — {reason}")
        print("    (lower bound only; under-reports ~5x. Use --measure to settle it.)")
        if not ok:
            failed = True
        if args.measure:
            peak = measure_peak(cfg)
            if peak is None:
                print(f"  MEASURED: OOM at batch {cfg.train.batch_size}")
                failed = True
            else:
                verdict = "FITS" if peak < args.vram * 0.9 else "TOO TIGHT"
                print(f"  MEASURED peak: {peak:.2f} GB at batch {cfg.train.batch_size} — {verdict}")
                failed = failed or peak >= args.vram * 0.9

    print("\nnote: estimates ignore cuDNN workspace and fragmentation — treat a")
    print("marginal pass as a fail. See configs/REPRODUCTION.md section 3.1.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
