"""Is the frequency-collapse convolution the pathology? Seconds of GPU time.

`faithful.yaml` measured **71.6 s/step** at batch 4 on an H100, where scaling
from `local.yaml` on a far weaker card predicts 3-7 s. Gradient checkpointing
was ruled out by direct A/B (3.49 vs 3.20 s/step, an 8% effect), which leaves
cuDNN algorithm selection as the leading suspect -- and `front.collapse` is the
layer to suspect.

It is `Conv2d(38, 4096, kernel=(1024, 1))` on a `(B, 38, 1024, T)` input, so its
output height is **1**. Mathematically that is a per-frame linear map from
38*1024 = 38,912 inputs to 4096 outputs, i.e. a matmul: 53.6 GFLOP/sample, or
~214 GFLOP at batch 4, which an H100 should finish in **milliseconds**. Written
as a convolution with a 1024-tall kernel, cuDNN picks an algorithm by heuristic,
and heuristics are least reliable on shapes like that.

This times the same arithmetic three ways -- the convolution with autotuning off
and on, and an explicit matmul -- and checks they agree numerically. If the
matmul is dramatically faster, the fix is a reformulation that leaves Table 1
and the maths untouched.

    python scripts/bench_collapse.py
"""

import time

import torch
import torch.nn as nn

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
B, CIN, F, T, COUT = 2, 38, 1024, 336, 4096
x = torch.randn(B, CIN, F, T, device=dev)
conv = nn.Conv2d(CIN, COUT, (F, 1)).to(dev)


def timed(fn, n=5, label=""):
    for _ in range(2):
        fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    print(f"  {label:<34} {(time.perf_counter() - t0) / n * 1000:9.1f} ms")


print(f"device {dev} | input {tuple(x.shape)} -> ({B}, {COUT}, 1, {T})")
for bench in (False, True):
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = bench
    timed(lambda: conv(x), label=f"Conv2d, cudnn.benchmark={bench}")

# Same arithmetic as a matmul: fold (CIN,F) into one axis, one GEMM per frame.
w = conv.weight.reshape(COUT, CIN * F)
b = conv.bias


def as_matmul():
    xf = x.permute(0, 3, 1, 2).reshape(B * T, CIN * F)
    return (xf @ w.T + b).reshape(B, T, COUT).permute(0, 2, 1).unsqueeze(2)


timed(as_matmul, label="equivalent matmul")

with torch.no_grad():
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
    err = (conv(x) - as_matmul()).abs().max().item()
print(f"  max abs difference: {err:.2e}  (same arithmetic, different kernel)")
