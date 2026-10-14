"""VGGish embeddings for Frechet Audio Distance. **Not part of the paper.**

FAD (Kilgour et al., Interspeech 2019) is the Frechet distance between VGGish
embeddings of a reference set and a generated set. `metrics.frechet_distance`
does the maths with no dependencies; this module supplies the embeddings, which
need a pretrained model.

VGGish loads through `torch.hub`, so there is no pip package to add — but the
first call **downloads weights**, which needs network access. `available()`
reports whether the model can be loaded, and the harness skips FAD with a
warning rather than failing, exactly as it does for PEAQ.

Two properties of FAD worth knowing before quoting one:

* **It is not a per-item metric.** It compares distributions, so it needs a set
  (a few hundred excerpts is usual) and produces a single number per set. It
  cannot go in the per-item best-of-20 table.
* **It is not comparable across embedders or set sizes.** A FAD from VGGish is
  not a FAD from PANN or CLAP, and small sets bias it upward. Report the
  embedder and the count alongside the value, always.

VGGish expects **16 kHz mono**; inputs are resampled, which means FAD sees none
of the >8 kHz content that MP3 restoration is largely about. That is a real
limitation of VGGish-FAD for this task, not a bug here — flagged so nobody
reads a good FAD as evidence about high frequencies.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

VGGISH_SAMPLERATE = 16_000
_MODEL = None


def _load():
    """Load VGGish once per process. Returns None if it cannot be loaded."""
    global _MODEL
    if _MODEL is None:
        try:
            model = torch.hub.load("harritaylor/torchvggish", "vggish", verbose=False)
            model.eval()
            model.postprocess = False  # want raw 128-d embeddings, not PCA+quantised
            _MODEL = model
        except Exception as exc:  # noqa: BLE001 - network, hub layout, torch version
            logger.warning("could not load VGGish for FAD: %s", exc)
            _MODEL = False
    return _MODEL or None


def available() -> bool:
    return _load() is not None


def embed(wavs: torch.Tensor, samplerate: int = 44100) -> torch.Tensor:
    """Embed `(n, samples)` mono waveforms into `(n * frames, 128)`.

    VGGish emits roughly one embedding per 0.96 s, so one excerpt yields several
    rows; they are all pooled into the set, which is what the reference FAD
    implementation does.
    """
    model = _load()
    if model is None:
        raise RuntimeError("VGGish unavailable; call fad.available() first")

    if wavs.ndim == 1:
        wavs = wavs.unsqueeze(0)
    wavs = wavs.detach().cpu().float()

    out = []
    for wav in wavs:
        resampled = _resample(wav, samplerate, VGGISH_SAMPLERATE)
        with torch.no_grad():
            emb = model.forward(resampled.numpy(), fs=VGGISH_SAMPLERATE)
        out.append(emb.cpu() if isinstance(emb, torch.Tensor) else torch.as_tensor(emb))
    return torch.cat([e.reshape(-1, e.shape[-1]) for e in out], dim=0)


def _resample(wav: torch.Tensor, src: int, dst: int) -> torch.Tensor:
    """Linear resample. Adequate here because VGGish is a coarse 64-mel front end;
    anything sharper would be wasted on it."""
    if src == dst:
        return wav
    n_out = int(round(wav.shape[-1] * dst / src))
    idx = torch.linspace(0, wav.shape[-1] - 1, n_out)
    lo = idx.floor().long().clamp(max=wav.shape[-1] - 1)
    hi = (lo + 1).clamp(max=wav.shape[-1] - 1)
    frac = idx - lo
    return wav[lo] * (1 - frac) + wav[hi] * frac
