---
paths:
  - "native/**"
  - "src/mp3ml/**"
  - "scripts/**"
  - "configs/**"
  - "tests/**"
---

# Keep the docs in step with the code, in the same change

Docs here carry **measured numbers and hard-won facts** — parameter counts,
memory figures, MP3 behaviour, what the reference paper does and does not say.
A stale figure is not untidy, it is a false claim, and the next reader has no
way to tell it from a current one. **Update the docs in the same turn as the
code, not afterwards.**

This rule previously missed `configs/**`, which is how `REPRODUCTION.md` ended
up asserting "the paper gives no training hyperparameters" after we had found
that it does. Hence the globs above, and the enforcement below.

## Enforcement — `tests/test_docs.py`

Config-derived figures quoted in `configs/REPRODUCTION.md` are **asserted
against the code**. Change a config and the doc fails the suite until updated.
Verified to work: raising `local.yaml`'s `stack_ch` fails with
`Obtained: 82.7, Expected: 84.85`.

If you add a new measured claim to a doc, add an assertion for it. If a claim
cannot be asserted cheaply (dataset-derived figures needing the corpus), say in
the doc how it was produced so it can be re-derived.

## What to update when

| you changed | also update |
|---|---|
| `configs/*.yaml` or `src/mp3ml/config.py` | `configs/REPRODUCTION.md` — the §1.2 cross-reference, §3 assumptions, §4.1 sizing table; re-run `scripts/check_config.py` and paste real numbers |
| anything about the paper's architecture or training | `configs/REPRODUCTION.md` §1 (with the section reference) and the changelog |
| `src/mp3ml/waveform.py` or the waveform builders | `DATASET.md` — especially the alignment section |
| `native/lame_dump/mp3dump.c` | `native/lame_dump/README.md` — traps, record layout, flags |
| the `record_t` struct | `RECORD_DTYPE` in `src/mp3ml/oracle.py` **and** the dumper README |
| `scripts/validate_oracle.py` | README's "What is verified", and `.claude/skills/validate-oracle/SKILL.md` |
| a CLI flag or command | README Commands block and CLAUDE.md Setup & commands |
| any quoted measurement | the number **everywhere it appears** — grep for the old value |

## Grep for the old value

The specific failure mode here is a number quoted in several files. After
changing one, **search for the previous value across the repo** rather than
relying on memory:

```sh
grep -rn "1\.45 GB" --include='*.md' .        # the stale figure
```

## Provenance must survive edits

`configs/REPRODUCTION.md` marks every claim **[P]** paper / **[I]** inferred /
**[A]** assumption / **[D]** deviation. When a claim's status changes — an
assumption confirmed by a closer reading, say — **change the marker and add a
changelog row**. Do not silently promote an inference to a fact; the record of
having been wrong is part of what makes the reproduction auditable.

## Discoveries about LAME go in the dumper README

`native/lame_dump/README.md` records non-obvious LAME behaviour: the
`double`-holding-`float` trap, the frame-lag join, the per-bitrate input scale,
the resampling boundary. When an experiment turns up another, write it there
**before moving on**, with the evidence. Each cost real debugging time to find.
