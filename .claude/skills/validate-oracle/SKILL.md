---
name: validate-oracle
description: Run the five-check validation on the LAME granule oracle. Use after changing anything in native/lame_dump/ or src/mp3ml/oracle.py, and before generating any dataset you intend to train on.
allowed-tools: Bash(bash native/lame_dump/build.sh), Bash(./.venv/bin/python scripts/validate_oracle.py *), Bash(./.venv/bin/python -m pytest tests/ *)
---

Rebuild the dumper if its source changed, then run
`./.venv/bin/python scripts/validate_oracle.py` (pass a wav path and `--seconds`
if the user named a file; otherwise the synthetic signal is fine and faster).

Report each check with its actual numbers, not just pass/fail:

1. **Encoder settings** — should be `identical` at every bitrate with analysis
   off. A DIFFER here means our LAME configuration has drifted from stock and
   nothing downstream is trustworthy. The reported "analysis perturbs N B" at
   320/192 kbps is expected and documented — do not report it as a failure.
2. **Decoder agreement** — normalized RMS should be ~1e-4 or below.
3. **Degradation trend** — must be monotonic; quote the 320k and 64k values so
   the difficulty range is visible.
4. **Granule alignment** — `block_type` and `global_gain` must both be exactly
   1.0000. Anything less means the frame join is misaligned and every pair is
   invalid; that is a hard stop, not a warning. Also quote the block-type
   histogram, since a corpus with no short blocks is not exercising the
   interesting path.
5. **xr across bitrates** — rel diff should be ~4e-7 where the lowpass matches.
   The 64 kbps row legitimately differs because its lowpass is 16500 Hz.

If check 4 fails, say so first and stop — the likely causes are the frame-lag
join or a `RECORD_DTYPE`/`record_t` mismatch, both described in
`native/lame_dump/README.md`.

If `scripts/validate_oracle.py` or the `mp3dump` binary is missing, say so and
offer to build it with `bash native/lame_dump/build.sh`.
