---
paths:
  - "src/mp3ml/models.py"
  - "src/mp3ml/stft.py"
  - "src/mp3ml/losses.py"
  - "src/mp3ml/train*.py"
  - "src/mp3ml/config.py"
  - "configs/**"
---

# Faithfulness check — run it on every addition, before calling it done

This repo reproduces a specific published model. Code that merely *runs* is not
the goal; code that matches the paper is. Every addition gets checked against
the source text **in the same turn it is written**.

## The check

1. **Find the paper's words for it.** Open the extracted text and locate the
   section, table row or equation the new code implements. Quote it in the
   docstring. If no passage covers it, it is **[A]** — say so explicitly.
2. **Compare line by line.** Not "does it look right" — read Table 1's columns
   one at a time: in maps, out maps, kernel, dilation, padding, **non-linearity**,
   output size. The non-linearity column is the easiest to skip and has already
   been missed once (`DeConv1`'s PReLU).
3. **Assert it in a test.** Shapes, channel counts, activation types, parameter
   counts. A faithfulness claim with no test is a claim that will silently rot.
4. **Update `configs/REPRODUCTION.md`** — the §1.2 cross-reference gets a row per
   new element with its source; new assumptions go in §3; new deviations in §4;
   add a changelog entry.
5. **Record what you got wrong.** If the check finds a mistake, put it in the
   changelog rather than quietly fixing it. Knowing which parts were hard to get
   right is information for whoever reads the results.

## Things already found by doing this

Each was caught by the check, not by the code failing:

- `nn.GLU` is **not** the paper's self-gating (eq. 2 uses PReLU; §3.1.2 says
  linear units were unstable). Shapes match either way — it would have trained.
- The critic had **one** stack, ignoring §3.1.1's two views. A parameter-count
  mismatch against the estimator exposed it.
- `DeConv1`'s **PReLU** was missing — Table 1's non-linearity column.
- The training setup *is* in §3.2; an earlier truncated search missed it and led
  to an unnecessary inference from a different paper's code.

## Do not

- Do not infer from other papers or repositories when this paper states it —
  check the paper first, exhaustively, before looking outward.
- Do not promote an inference to a fact. Markers **[P]**/**[I]**/**[A]**/**[D]**
  exist for this; changing one needs a changelog row.
- Do not trust an analytic estimate where a measurement is possible. The memory
  estimator under-reported by ~5x until `--measure` was added.
