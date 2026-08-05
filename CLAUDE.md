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
uv run python local_sim/board_snapshot.py --top 30 --label <tag>   # archive a leaderboard (+ --diff a.json b.json)
uv run python local_sim/board_forensics.py                          # implied V*c, entry counts, score timeline
uv run python local_sim/twin_diff.py --all                          # suite identity across our graded reports
uv run python local_sim/phase2_checklist.py                         # rounds / dataset tags / boards
uv run python local_sim/export_split.py --split mini --n 100 --out <npz>   # dataset -> npz for the GPU venv
.venv-gpu\Scripts\python.exe local_sim/track_t_mint.py --data <npz> --out <npz> --draws 256
.venv-gpu\Scripts\python.exe local_sim/track_p_spectrum.py --data <npz> --mlps 100 --jensen --out <npz>
```

## Known unknowns / gotchas (updated 2026-08-05 night)

- **The public score is the mean over 50 FIXED MLPs (indices 0–49)**, and every submission returns those MLPs' exact per-MLP `final_layer_mse` + a 32-long `per_layer_mse` in `results.public_scores`. Verified bit-for-bit against the aggregates; the suite is identical across all four of our submissions. Consequences: (a) our graded A/B comparisons are **paired re-measurements**, not fresh samples — never read a graded improvement as population evidence; (b) the level carries a fixed 7–20% offset vs the local 100; (c) **decisions stay on the local 100-MLP mini split**, which is the honest proxy for the private re-run that sets the final ranking. Full record + the arithmetic: RESEARCH_NOTES §2g (**private — never the forum**). **We do not use that channel.**
- **The `sampling_mse` label-noise bound is WITHDRAWN.** 89 of 200 board entries show a final-layer MSE below 6.4695e-7, so it cannot be additive reference noise. Most likely a reference sampling baseline's MSE. Do not repeat the "nothing honest below 6.47e-8" claim.
- **Track T (shrinkage-moment tail) is CLOSED** — 0.9998× vs a 1.10× gate. Only `diag(WᵀΣW)` = `mean(z²)−mean(z)²` enters, so there is no covariance to estimate (no MP regime); v̂ is already 1.13% rms accurate and the exact-variance oracle still gives 0.611×. 41% of the analytic tail's error is irreducible model bias; ρ=1.000 with the sample mean in the α≥3 band that carries 76.5% of the MSE. Don't reopen without a new mechanism.
- **Track P (rank-k deviation propagation) is CLOSED** — 1.093× vs a 1.25× gate, even pricing an oracle Jensen correction at zero. Rank-k propagation *is* a rank-k weight factorisation; shape-based billing breaks even at k = width/2 = 128, which is also the rank truncation needs. (The spectrum result itself is real and write-up-worthy: effective rank 25.9→2.8 over depth, and the closure's own covariance eigenvectors find the subspace for free.)
- **`fnp.linalg.eigh` on JJᵀ is only well-determined in its leading directions** — LAPACK and cuSOLVER disagree on 106 of 256 columns beyond sign. The rotation's *distribution* is unaffected but the realised per-MLP error is BLAS-dependent, so expect ~1% of unexplained local-vs-grader difference from this alone.
- **RQMC helps the first moment only**: measured effective N 32,870 for `mean(z)` (2.01×) but 18,719 for the sample variance (1.14×). Any scheme leaning on sample *variances* gets no QMC discount.

## Older gotchas (from 2026-08-04, after submission 323872)

- **We are in PHASE 1 until Aug 7 23:59 UTC.** Rounds (from the challenge page): 1425 warm-up (closed), **1428 "Phase 1" Jun 18 → Aug 7 23:59 — LIVE**, 1429 "Phase 2" Aug 8 → Sep 19 23:59 — UPCOMING (empty leaderboard, no `v1-phase2` HF tag). Earlier docs said Phase 2 opened Aug 1; wrong. Phase 1 was **extended one week** (Jul 31 → Aug 7) after a leaderboard re-score; write-up deadline moved to **Aug 17** (forum 18109). Phase 2 uses the same grader stack.
- **Live board (Aug 4)**: #1 4.87e-8, #10 9.85e-8, #200 6.40e-7. The old "Phase 1 #1 = 1.24e-8" note is superseded — see RESEARCH_NOTES §5b.
- **Never call `flops.budget_summary_dict()`/`budget_summary()` inside `predict`** — O(op history), measured 157× residual blowup (forum 18092). Size sample counts analytically. Freeing large temporaries also bills residual wall time (18101), so prefer **few large chunks over many small ops**. Full list: RESEARCH_NOTES §3c.
- **The score is `V·c/B`** — per-sample variance × per-sample cost; N cancels above the 0.1 floor. Cheapening a sample is worth exactly as much as de-noising it. Pricing that matters: **f16/int8/bool are rate 1.0, same as f32 — there is no dtype trick**; matmul bills `(2K−1)·M·N` **by shapes only**, so pruning pays only via smaller *dense* matrices; gather/where 4/elt; transcendentals 16/elt (`norm.ppf` 168/elt); thin SVD 26n³ ≈ 0.16% B, eigh 9n³. Full table: RESEARCH_NOTES §3d.
- **α = 1.00 is theory, not a bug** — for discontinuous integrands the RQMC rate bonus is 1/(4d−2) ≈ 0.001 at d=256 (He & Wang 2015). Sobol buys constants only; rate recovery needs kinks concentrated into few coordinates.
- **Decision policy: nothing is decided on fewer than the full 100-MLP mini split.** Three "wins" evaporated on re-measurement (LMS scrambling, C/B=0.5, a +33% rotation).
- **The analytic family is retired** (GAME_PLAN rev 3): B1 fusion net ~0, B2 tail −67%…−214%, B3 learned residual R²=0.012.
- **Phase 1 config confirmed** by the graded report: width 256 × depth 32, n_mlps 100, B = 2.72e11, λ = 1.000e11 (derived exactly from per-MLP telemetry), multiplier floor 0.1. **`admin_config` is admin-only** — participants never see it; `evaluation.config_snapshot` + per-MLP telemetry is what we get. `wall_time_limit_s` remains unknown (only bound: ≥0.6 s, our max was 0.592 s with no timeouts). **Phase 2's config stays unverified until a graded run in round 1429 (Aug 8+).**
- Real flopscope API matches `estimator.py` exactly — `whest validate` passed with zero changes (`fnp.diagonal/diag/outer/einsum`, `flops.stats.norm.cdf/pdf`, `flops.as_symmetric(data, *, symmetry=(0,1))` all correct, and `as_symmetric` accepts our width-256 covariance). If a future edit trips a name/signature, fix `estimator.py` minimally without changing the math, mirror into the shim, and log it.
- Grader reports are **not** returned by `whest submit --json` (it gives only id/score/score_secondary) nor by the `/api/v1/submissions/{id}` endpoint. The full report lives in the submission page's `data-react-props` payload → `props.data.submission.evaluation`.
- `whest login`/submit 302 or permission errors usually mean challenge rules need (re-)accepting in the browser for the current phase — stop and ask Samuel. (Did not occur for the Phase 1 round.)
- v0 grader baseline to beat: adjusted **2.4927e-6**, `mean_compute_utilization` 1.45%, 0 failures. Local mini-split predicts the grader to within ~7%. If a future run is >3× off local expectation, investigate before resubmitting.

## Working style

Samuel has a CS degree but is new to ML/probability — explain what you're doing in plain English as you go, in the final summaries. Claude (cloud session) drives strategy and heavy experiments; Claude Code handles local pipeline, submissions, and grader feedback. Roadmap and phase gates: `docs/GAME_PLAN.md` (A: RQMC → B: hybrid+control variates → C: learned corrector → D: mechanistic cumulant track for the $20k contribution prize, write-up due Sep 26).
