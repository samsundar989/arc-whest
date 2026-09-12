# Tiny oracle attempt 1 — fixture failure, 2026-09-12

After Samuel's committed predictions, ran
`uv run python local_sim/test_p2_d21_fusion_oracle.py --output experiments/2026-09-12_f21/tiny_oracle.json`
against probe source commit `cb940ee`. It exited 1 before writing output:
`AssertionError` at the `column_cancellation` fixture's zero-tensor assertion
(line 161). The independent exact tensor for that fixture was
`[[0,74,-82],[-12,0,-70],[28,64,0]]`, so the fixture did not satisfy its
claimed premise. No candidate source was imported, and no challenge seed or
truth was read.

Cause: paired opposite `A_L` columns with equal `w2b` produce equal-sign
two-leg contributions because `C=A_L*w2b`. The repair uses equal `A_L`
columns and opposite `w2b` within each pair, while preserving opposite
`A_D` and duplicate `M`; the cancellation is then algebraic in both blocks.
The failed attempt remains part of the record rather than being counted as a
passed candidate check.
