# 2026-10-12 — the gain mode on exact labels (strategist session experiment)

10 FIT nets (`cache/twin_labels/fit_000..009.npz`), all 16 layers; weights from `cache/twin/full_w.f32`.
Summary: `docs/research/2026-10-12_deep_review.md` §5.2 and `docs/HANDOFF_2026-10-12_LANE_K.md`.

Conventions found: `dG` in the label files is kappa4/2; the diagonals of C11/D21/D22/Q31 are stored as zero; D21[a,c] = kappa3(a,a,c);
W is (n_in, n_out), z_l = W_l^T h_{l-1}.

- `common.py` (loaders), `gain.py` (per-net analysis: v estimates, gain share by class, pruning test), `mc.py` (Monte-Carlo check of the
  two amplitudes p, q), `extra.py` (convention check, off-diagonal covariance error, combined pruning).
- `agg*.py` → `tables_agg1..6.txt`, `tables_mc.txt`, `tables_extra.txt` (full per-layer tables).
- Raw per-net results `out_*.json` are left on disk here but git-ignored (11 MB).

`run_gain.sh` and the loaders point at the session scratch dir; set the paths before rerunning.
