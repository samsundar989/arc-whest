# CLAUDE.md — arc-whest project memory

## What this repo is

Samuel's entry for the **ARC White-Box Estimation Challenge 2026** (AIcrowd). Task: given a random He-init ReLU MLP (Phase 1: 256 wide × 32 deep, no biases), predict every neuron's expected post-ReLU activation under x ~ N(0, I), returning a (depth, width) array. Score = mean over MLPs of `final_layer_MSE × max(0.1, C/B)`, lower is better; C = flopscope-counted FLOPs + 1e11 × uninstrumented wall-seconds; B = 2.72e11 (Phase 1). **Phase 2 deadline: Sep 19, 2026, 23:59 UTC.** Full context: `docs/COMPETITION_BRIEF.md` (explainer), `docs/GAME_PLAN.md` (strategy), `docs/RESEARCH_NOTES.md` (rules/intel/links), `docs/SETUP_WINDOWS.md` (pipeline commands), `experiments/LOG.md` (results so far). Read them before nontrivial work.

## Layout

- `estimator.py` — THE submission (self-contained single file; `whest submit` ships only this). v0 = covariance propagation + offline-fitted per-layer calibration (`_CAL_LAYER_SCALE`, fitted at width 256; provenance `experiments/calibration.json`).
- `local_sim/` — offline lab: numpy shims for `flopscope`/`whestbench` so the exact `estimator.py` runs without competition packages; He-MLP generator; MC ground truth; `fit_calibration.py`; `validate_estimator.py` (held-out check — v0 reference: raw 7.98e-5 → calibrated 2.67e-5 final-layer MSE).
- `experiments/` — LOG.md (append every experiment: idea → local MSE → graded score → verdict), calibration.json, run reports (save grader reports under `experiments/run_reports/`).

## Hard rules (grader + integrity)

1. `estimator.py` may import ONLY `flopscope`, `flopscope.numpy as fnp`, `whestbench`, and Python stdlib. No numpy/scipy/torch in that file — the grader sandbox doesn't have them. Grader arrays are immutable (no `a[i]=`, no `+=` on arrays); pass `str` not `Path`; single array ≤ 4 GiB.
2. Determinism: any randomness must derive from `mlp.seed` (predict) / `context.seed` (setup) via `fnp.random.default_rng(...)`. Never `fnp.random.seed(...)`, never time-based seeds. v0 is fully deterministic (no RNG).
3. **Never exploit the flopscope wall-time accounting loophole** (RESEARCH_NOTES §3, thread 18099) — routing math through raw uninstrumented arrays to dodge FLOP counting. Organizers call it out; we compete clean and build for the private re-grade.
4. Secrets: the AIcrowd API key lives only in `whest login`'s credential store. Never write it into tracked files, logs, or LOG.md.
5. Budget discipline: 50 submissions/day cap exists, but default to ≤2/day with purpose; validate → local mini-split run → submit, in that order. A failed MLP scores as zeros (catastrophic) — never submit something that hasn't passed `whest validate` + a clean mini run (`n_failed_mlps: 0`).
6. Keep `estimator.py` shape-generic (read `mlp.width`/`len(mlp.weights)`); calibration auto-disables off-width via `_calibration_column`. Env var `ARC_WHEST_DISABLE_CAL=1` disables calibration (used by fitting).

## Commands

```powershell
uv sync                                          # env (Python 3.10 + flopscope + whestbench)
uv add --dev numpy scipy                         # dev-only deps for local_sim tools
uv run whest validate --estimator estimator.py   # contract check (seconds)
uv run whest run --estimator estimator.py --dataset hf://aicrowd/arc-whestbench-public-2026@v1-phase1 --split mini --runner subprocess
uv run whest submit --estimator estimator.py --watch
uv run python local_sim/validate_estimator.py    # offline held-out MSE check
uv run python local_sim/fit_calibration.py       # refit calibration (writes experiments/calibration.json)
```

## Known unknowns / gotchas (as of 2026-08-04)

- **Phase 2 config unconfirmed** — assumed same as Phase 1 (256×32, B=2.72e11, floor 0.1, λ=1e11). The graded run report's `admin_config` is authoritative (also grab `wall_time_limit_s`). No `v1-phase2` HF dataset tag existed as of Aug 4 — if one appears, switch local runs to it and note in LOG.
- The shims in `local_sim/` approximate the real fnp API. If `whest validate` flags a name/signature (`fnp.diagonal`, `fnp.diag`, `fnp.outer`, `flops.stats.norm.*`, `flops.as_symmetric`), fix `estimator.py` minimally without changing the math, mirror the fix into the shim if needed, and log it.
- `whest login`/submit 302 or permission errors usually mean challenge rules need (re-)accepting in the browser for the current phase — stop and ask Samuel.
- Expected v0 grader result: adjusted ≈ 2–3e-6, `mean_compute_utilization` < 2% (multiplier at 0.1 floor). If >3× off from local expectation, investigate before resubmitting (flopscope pricing of einsum/norm.cdf, wall-time overhead ≈ 3.7% budget per 0.1 s are prime suspects).

## Working style

Samuel has a CS degree but is new to ML/probability — explain what you're doing in plain English as you go, in the final summaries. Claude (cloud session) drives strategy and heavy experiments; Claude Code handles local pipeline, submissions, and grader feedback. Roadmap and phase gates: `docs/GAME_PLAN.md` (A: RQMC → B: hybrid+control variates → C: learned corrector → D: mechanistic cumulant track for the $20k contribution prize, write-up due Sep 26).
