---
paths:
  - "src/mp3ml/dataset.py"
  - "src/mp3ml/waveform.py"
  - "scripts/build_*.py"
---

# Data pipeline conventions

- Datasets must not do heavy work or network I/O at import time — do it in `__init__` or lazily in `__getitem__`.
- Never let anything here peek at the val/test split during train-time preprocessing (e.g. computing normalization stats on the full dataset). Fit stats on train only, and say so in a comment.
- Split **by track, never by chunk or segment**. Chunks from one song landing in both train and val is leakage, and musdb stems must stay in their own mixture's split — they are derived from it.
- New datasets return plain tensors/dicts, not framework-specific wrapper objects, unless the rest of the pipeline already expects one.
- Give every public `Dataset`/`DataLoader` factory a one-line docstring stating what split and shape it returns.
