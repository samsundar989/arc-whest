# 2026-10-12 — exact prices of the pruning space (strategist session)

Every number is a real patched copy of `estimator_k3_v84_age6.py` metered by the shape-only interpreter (`fakefs.py`, from `../2026-10-11_p6price/`);
the base reproduces 381,613,795,475 FLOPs / 22,113 ops to the FLOP. Summary: `docs/research/2026-10-12_deep_review.md` §5.3.

- `x3_tables.txt` — all tables (singles S1–S3, combos C1–C4, ages x old tier x glue, last layer, rounding, per-network variability).
- `x3lib.py` — the patch builder (exact-anchor text patches on v84: separate D21 row/column sets, kink-at-birth hub columns, two-column-set,
  leg output-row sets, ages, old tier on/off/rank, last-layer K youngest). Shapes are right; numerics are placeholders.
- `x3run.py`, `x3price.py`, `x3grid.py`, `x3extra.py`, `x3report.py`, `x3fit.py`, `x3round.py`, `x3pernet.py` — runner, pricing, grids, reports.
- `sbmodel.py` — the Strassen cost recurrence with a `glue` parameter and `best_depth`.
- `alpha_stats.py`, `x3classes.py`, `alpha_stats.npz` (git-ignored) — per-net alpha for FIT-100 and the class-count tables.

Trap: reusing one variant filename lets Python load stale `__pycache__` bytecode; use unique names and `python -B`.
Glue model here is variant A (21 quarter-areas per level); it was priced before the real `out=` probe confirmed it.
