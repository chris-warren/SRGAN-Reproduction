---
paths:
  - "src/**/train*.py"
  - "scripts/train*.py"
---

# Training loop conventions

- Every run must be resumable from a checkpoint under `outputs/<run_name>/checkpoints/`.
- Log loss/metrics every N steps through the project logger, not ad hoc prints. Log the config and seed at run start.
- Guard against silent NaN/Inf loss: check and fail loudly rather than letting a run continue.
- Test training-loop changes against `configs/smoke.yaml` before a full run.
- Keep device placement (`.to(device)`) in one place — the training loop setup — not scattered across model/data code.
