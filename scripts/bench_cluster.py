"""Size the hardware needed for a faithful run. Run this FIRST on the cluster.

Answers three questions a 40,000-iteration commitment depends on, none of which
can be extrapolated from a development card:

1. **Does it fit, and with how much room?** Peak allocated memory for a real
   training step at the config's batch size, with `amp` and `grad_checkpoint`
   as configured.
2. **How fast is a step?** Median seconds per optimizer step, excluding warm-up,
   and the implied wall time for the full schedule.
3. **Is the GPU or the data pipeline the bottleneck?** With `data.source:
   apollo` the MP3 round-trip costs ~0.8 CPU-seconds per item, so at batch 12 a
   step needs ~9.5 CPU-seconds of work. Below roughly 12 workers the GPU waits,
   and no faster GPU helps. This prints the crossover.

    python scripts/bench_cluster.py configs/apollo64.yaml --workers 16

Report the output in REPRODUCTION.md rather than trusting it to memory: these
are measurements, and this project's history is that unmeasured performance
claims do not survive (REPRODUCTION.md §3.4).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mp3ml.config import load_config, single_bitrate
from mp3ml.data import build_train_dataset
from mp3ml.tracking import Tracker
from mp3ml.train import Trainer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("--steps", type=int, default=30, help="timed steps after warm-up")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--bitrate", type=int)
    p.add_argument(
        "--batch-size",
        type=int,
        help="override the config's batch size. The paper's 12 is [P], but a "
        "measurement that OOMs is no measurement -- halve it until it fits, then "
        "scale the memory figure back up. Activations dominate and scale ~linearly.",
    )
    p.add_argument(
        "--memory-breakdown",
        action="store_true",
        help="report where the memory goes (weights, grads, optimizer, peak) "
        "instead of only the peak",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        help="override the corpus path in the config (also settable with "
        "$APOLLO_DATA). The YAML points at a workstation drive that will not "
        "exist on a cluster.",
    )
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.batch_size is not None:
        import dataclasses

        cfg = dataclasses.replace(
            cfg, train=dataclasses.replace(cfg.train, batch_size=args.batch_size)
        )
    if args.data_root is not None:
        import dataclasses

        cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, root=args.data_root))
    if args.bitrate is not None:
        cfg = single_bitrate(cfg, args.bitrate)
    device = torch.device(args.device)

    print(f"config {cfg.name} | source {cfg.data.source} | batch {cfg.train.batch_size}")
    print(f"amp {cfg.train.amp} | grad_checkpoint {cfg.train.grad_checkpoint}")
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        total = torch.cuda.get_device_properties(device).total_memory / 1e9
        print(f"device {name} ({total:.0f} GB)")

    # --- 1. data pipeline alone, no GPU work ---------------------------------
    dataset = build_train_dataset(cfg)
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    # Two traps here, both of which produced absurd numbers (3,000 items/s
    # against a known ~1.3 items/s per worker) before being fixed:
    #
    # 1. DataLoader prefetches `prefetch_factor * workers` batches, so the first
    #    reads are instant and measure the queue draining, not production.
    # 2. Batches come back in strict index order, so with W workers the timings
    #    arrive in bursts of period W -- one long wait on worker 0 and then W-1
    #    batches that were already finished. A median over a window that short
    #    lands on the sub-millisecond entries and reports the queue, not the
    #    pipeline.
    #
    # So: warm up past the prefetch buffer, then take TOTAL elapsed over whole
    # worker cycles. Mean over cycles is the only statistic that survives the
    # burst pattern.
    it = iter(loader)
    workers = max(args.workers, 1)
    for _ in range(2 * workers + 2):
        next(it)
    cycles, batches = 4, 4 * workers
    t0 = time.perf_counter()
    for _ in range(batches):
        next(it)
    data_s = (time.perf_counter() - t0) / batches
    del cycles
    print(
        f"\ndata: {data_s:.3f} s/batch at {args.workers} workers "
        f"({cfg.train.batch_size / max(data_s, 1e-9):.1f} items/s)"
    )
    del it, loader

    # --- 2. a real training step --------------------------------------------
    trainer = Trainer(
        cfg,
        Path("outputs") / f"bench_{cfg.name}",
        device=device,
        tracker=Tracker(Path("outputs") / f"bench_{cfg.name}", cfg, backend="none"),
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    trainer.fit(
        log_every=10**9, checkpoint_every=10**9, max_steps=args.warmup, num_workers=args.workers
    )
    t0 = time.perf_counter()
    trainer.fit(
        log_every=10**9,
        checkpoint_every=10**9,
        max_steps=args.warmup + args.steps,
        num_workers=args.workers,
    )
    step_s = (time.perf_counter() - t0) / args.steps

    print(f"step: {step_s:.3f} s/step over {args.steps} steps")
    if args.memory_breakdown and device.type == "cuda":
        n_g = sum(p.numel() for p in trainer.generator.parameters())
        n_d = sum(p.numel() for p in trainer.critic.parameters())
        gb = 4 / 1e9
        print(
            f"  weights        {(n_g + n_d) * gb:6.2f} GB  "
            f"(G {n_g / 1e6:.0f} M + D {n_d / 1e6:.0f} M)"
        )
        print(f"  + grads        {(n_g + n_d) * gb:6.2f} GB")
        print(f"  + Adam moments {(n_g + n_d) * 2 * gb:6.2f} GB")
        fixed = (n_g + n_d) * 4 * gb
        print(f"  = fixed        {fixed:6.2f} GB")
        print(
            f"  activations etc{torch.cuda.max_memory_allocated() / 1e9 - fixed:6.2f}"
            " GB (peak minus fixed)"
        )
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(device).total_memory / 1e9
        print(f"peak memory: {peak:.1f} GB of {total:.0f} GB ({100 * peak / total:.0f}%)")

    # --- 3. verdict ----------------------------------------------------------
    hours = cfg.train.iterations * step_s / 3600
    print(f"\n{cfg.train.iterations:,} iterations -> {hours:.1f} h ({hours / 24:.1f} days)")
    if data_s > step_s * 0.9:
        need = int(args.workers * data_s / (step_s * 0.5)) + 1
        print(f"BOTTLENECK: data ({data_s:.3f} s) is not behind compute ({step_s:.3f} s).")
        print(f"  Try ~{need} workers, or pre-build the degraded audio so the")
        print("  MP3 round-trip is not paid per step. A faster GPU will not help.")
    else:
        print(f"GPU-bound ({step_s:.3f} s/step vs {data_s:.3f} s/batch of data) — as it should be.")
    print(
        f"\nSix models for Table 2 at this rate: {6 * hours:.0f} GPU-hours "
        f"({6 * hours / 24:.1f} days on one card, no DDP in this codebase)."
    )


if __name__ == "__main__":
    main()
