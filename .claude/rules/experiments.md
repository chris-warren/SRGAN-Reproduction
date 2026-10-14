---
paths:
  - "configs/**"
  - "experiments/**"
  - "src/mp3ml/metrics.py"
  - "outputs/experiments/**"
---

# Experiment & config conventions

- One YAML per experiment in `configs/`. Inherit shared defaults rather than copy-pasting a full config.
- Config field names must match the dataclass field they populate — don't rename one without the other.
- New config fields need a sensible default so existing configs keep working.
- Record what changed and why in the run's `outputs/<run_name>/notes.md`, not only in the git commit message — different audiences read each.

## Validate the metric before you trust it on a model

Two days were lost to an `istft` bug that made every waveform metric wrong by up
to 46 dB, and a day before that to two metrics that saturate (REPRODUCTION.md
§4c, §5). Three cheap habits, each of which would have caught one of them:

- **Score the pass-through on every metric, in the same run, and print it
  first.** "The loss fell 84%" meant nothing against an input that was already
  9x better. A model number with no baseline beside it is not a result.
- **A baseline that shares the bug's blind spot proves nothing.** The MP3
  baseline's SNR matched the paper's to 1 dB *while the metric was broken*,
  because `clean` and `y` carry identical edge artefacts that cancel in their
  difference. Validate a metric with a **perturbed target** — error independent
  of both signals, several known magnitudes — and check the numbers move
  together. No model, no GPU.
- **Two metrics that contradict each other are a bug until proven otherwise, and
  the first suspect is the ruler.** Training MSE 39x better than the input while
  SI-SDR said 23 dB worse produced two confident, wrong, model-shaped
  explanations before anyone perturbed a known signal. A metric frozen to two
  significant figures while its neighbours move is broken, not stuck.

When a test works around a limitation — `test_stft.py` trimmed the bad edges
before comparing, from the day it was written — **the code must enforce it**, or
the workaround is the only place the knowledge lives.
