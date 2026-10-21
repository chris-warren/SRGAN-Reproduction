"""Is the §4.6 divergence an artefact of `local.yaml`'s reduced size?

**This is the open question blocking a real run.** Training diverges with
`recreate_optimizers: true` (the original implementation's behaviour) and is
stable with it off — see REPRODUCTION.md §1.2 for the measured matrix.

The leading untested hypothesis: a fresh Adam optimiser makes every update
sign-SGD with step size exactly `lr`, so a noisy gradient flips signs and the
weights random-walk at full stride. `local.yaml` uses **batch 4**; the authors'
coefficients were recovered from runs at the paper's **batch 12**. If that is
the mechanism, larger batches should be markedly more stable, and width should
matter little (or go the other way, since more parameters each move the full
`lr`).

Run with `recreate_optimizers: true` in the config — the point is to reproduce
the divergence and see what suppresses it::

    python scripts/diagnose_divergence.py

Kept in the repo rather than a scratch directory because the last set of
results for this question was lost to a /tmp clear.
"""

import copy
import statistics
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mp3ml.config import load_config, single_bitrate
from mp3ml.data import build_train_dataset
from mp3ml.train import Trainer

BASE = single_bitrate(load_config(Path("configs/local.yaml")), 16)
RUN = Path("outputs/experiments/_scratch_run")
STEPS = 40


def bench(label, mutate):
    cfg = copy.deepcopy(BASE)
    mutate(cfg)
    cfg.train.gp_divergence_patience = 10**9
    ds = build_train_dataset(cfg)
    loader = DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=6,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
    )

    def endless():
        while True:
            yield from loader

    torch.manual_seed(0)
    try:
        tr = Trainer(cfg, RUN, device="cuda")
        b, gps = endless(), []
        for _ in range(STEPS):
            gps.append(tr.train_step(b)["gradient_penalty"])
        print(
            f"{label:<38} peak={max(gps):11.4g}  "
            f"median_last10={statistics.median(gps[-10:]):11.4g}",
            flush=True,
        )
        del tr
    except torch.cuda.OutOfMemoryError:
        print(f"{label:<38} OOM", flush=True)
    torch.cuda.empty_cache()


def nop(c):
    pass


def b8(c):
    c.train.batch_size = 8


def b12(c):
    c.train.batch_size = 12


def wide(c):
    # halfway to faithful: full latent_freq/stack widths, still 512 bins
    c.model.latent_ch, c.model.latent_freq, c.model.conv4_out = 128, 32, 4096
    c.model.stack_ch, c.model.noise_ch = 256, 64
    c.train.batch_size = 4


def wide_b12(c):
    wide(c)
    c.train.batch_size = 12


bench("batch 4  (local default)", nop)
bench("batch 8", b8)
bench("batch 12 (paper's batch)", b12)
bench("faithful widths, batch 4", wide)
bench("faithful widths, batch 12", wide_b12)
