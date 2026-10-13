# 2026-10-12 — Strassen glue with `out=` views (strategist session probe)

Measured on the real flopscope 0.12.1 (package copied out of `.venv/Lib/site-packages/flopscope`, imported with the VM's numpy).
Summary and tables: `docs/research/2026-10-12_deep_review.md` §5.1 and `docs/HANDOFF_2026-10-12_LANE_S.md`.

- `probe_out.py` / `probe_out.log` — what each `out=` form bills (all bill numel(out); slice assignment still raises).
- `swproto.py` — prototypes: `sw_out` (A: stacked buffers written with `out=`, post-adds into output quadrants, 21 quarter-areas per level),
  `sw_out(fused=True)` (B: A + fused last level, 15 on that level), `sw_ks` (C: Winograd in a Karstadt–Schwartz alternative basis, in place, 12 per level);
  a verbatim copy of the estimator's `_sb`; exact cost models (equal the real meter to the FLOP and the op).
- `test_proto.py`, `test_exact.py` — meter vs model, bit-identity of A/B vs `_sb` (`np.array_equal`), error of C.
- `audit_out.py` — every `out=` write is billed numel(out) (issue #275 control); NaN-poisoned `empty` buffers leak nothing.
- `ks_search.py` — alternative-basis search (3 + 3 + 6 is the minimum found).
- `ledger_summary.py`, `estimate_v84.py`, `shape_table.py` — the v84 tables (need a per-call ledger from the shape-only interpreter, `../2026-10-11_p6price/`).
- `bench.py`, `leaf_time.py`, `*.log` — VM wall/RSS and real-meter runs.

Paths inside the scripts point at the session scratch dir; set them to this folder before rerunning.
