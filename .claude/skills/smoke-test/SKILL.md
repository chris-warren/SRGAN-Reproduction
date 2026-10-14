---
name: smoke-test
description: Run the fast sanity training run to catch shape/dtype/NaN bugs before a full training run. Use after changing data, model, or training loop code.
allowed-tools: Bash(python scripts/smoke_test.py *)
---

Run `python scripts/smoke_test.py` — a couple of steps on a tiny synthetic batch, CPU, should finish in well under a minute.

Report:
- Whether it completed without error
- Whether the loss is a finite number that changes between steps (not NaN/Inf, not frozen)
- The exact error and file/line if it failed

If `scripts/smoke_test.py` doesn't exist yet, say so and offer to write one: build the model, run 2-3 steps of the real train loop on a tiny synthetic batch, assert the loss is finite, and exit non-zero on failure.
