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

## Known unknowns / gotchas (updated 2026-08-04, after submission 323872)

- **We are in PHASE 1 until Aug 7 23:59 UTC.** Rounds (from the challenge page): 1425 warm-up (closed), **1428 "Phase 1" Jun 18 → Aug 7 23:59 — LIVE**, 1429 "Phase 2" Aug 8 → Sep 19 23:59 — UPCOMING (empty leaderboard, no `v1-phase2` HF tag). Earlier docs said Phase 2 opened Aug 1; wrong. Phase 1 was **extended one week** (Jul 31 → Aug 7) after a leaderboard re-score; write-up deadline moved to **Aug 17** (forum 18109). Phase 2 uses the same grader stack.
- **Live board (Aug 4)**: #1 4.87e-8, #10 9.85e-8, #200 6.40e-7. The old "Phase 1 #1 = 1.24e-8" note is superseded — see RESEARCH_NOTES §5b.
- **Never call `flops.budget_summary_dict()`/`budget_summary()` inside `predict`** — O(op history), measured 157× residual blowup (forum 18092). Size sample counts analytically. Freeing large temporaries also bills residual wall time (18101), so prefer **few large chunks over many small ops**. Full list: RESEARCH_NOTES §3c.
- **Phase 1 config confirmed** by the graded report: width 256 × depth 32, n_mlps 100, B = 2.72e11, λ = 1.000e11 (derived exactly from per-MLP telemetry), multiplier floor 0.1. **`admin_config` is admin-only** — participants never see it; `evaluation.config_snapshot` + per-MLP telemetry is what we get. `wall_time_limit_s` remains unknown (only bound: ≥0.6 s, our max was 0.592 s with no timeouts). **Phase 2's config stays unverified until a graded run in round 1429 (Aug 8+).**
- Real flopscope API matches `estimator.py` exactly — `whest validate` passed with zero changes (`fnp.diagonal/diag/outer/einsum`, `flops.stats.norm.cdf/pdf`, `flops.as_symmetric(data, *, symmetry=(0,1))` all correct, and `as_symmetric` accepts our width-256 covariance). If a future edit trips a name/signature, fix `estimator.py` minimally without changing the math, mirror into the shim, and log it.
- Grader reports are **not** returned by `whest submit --json` (it gives only id/score/score_secondary) nor by the `/api/v1/submissions/{id}` endpoint. The full report lives in the submission page's `data-react-props` payload → `props.data.submission.evaluation`.
- `whest login`/submit 302 or permission errors usually mean challenge rules need (re-)accepting in the browser for the current phase — stop and ask Samuel. (Did not occur for the Phase 1 round.)
- v0 grader baseline to beat: adjusted **2.4927e-6**, `mean_compute_utilization` 1.45%, 0 failures. Local mini-split predicts the grader to within ~7%. If a future run is >3× off local expectation, investigate before resubmitting.

## Working style

Samuel has a CS degree but is new to ML/probability — explain what you're doing in plain English as you go, in the final summaries. Claude (cloud session) drives strategy and heavy experiments; Claude Code handles local pipeline, submissions, and grader feedback. Roadmap and phase gates: `docs/GAME_PLAN.md` (A: RQMC → B: hybrid+control variates → C: learned corrector → D: mechanistic cumulant track for the $20k contribution prize, write-up due Sep 26).
