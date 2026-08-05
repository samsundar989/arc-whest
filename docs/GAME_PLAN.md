# Game plan — ARC White-Box Estimation Challenge 2026

*Goal: win Phase 2 (Best Score, $50k) and contend for Best Algorithmic Contribution ($20k). Written Aug 4, 2026; **round dates corrected Aug 5** after reading them off the challenge page.*

## Round calendar (measured, authoritative)

| round | window (UTC) | state as of Aug 5 |
|---|---|---|
| warm-up (1425) | — | closed |
| **Phase 1 (1428)** | **Jun 18 → Aug 7, 23:59** | **LIVE — closes in days** |
| Phase 2 (1429) | **Aug 8 → Sep 19, 23:59** | upcoming, empty leaderboard, same grader stack |

Earlier drafts assumed Phase 2 opened Aug 1. Wrong — Phase 1 is still live and closes **Aug 7 23:59 UTC**, so near-term work targets the Phase 1 board; Phase 2 opens Aug 8 and re-verification of its config happens with our first submission there.

## The strategic picture

- **The bar** (live Phase-1 board, measured Aug 4): **#1 = 4.87e-8**, #10 = 9.85e-8, #200 = 6.40e-7. The warm-up round topped out at 3.91e-8. ~~Phase 1's leader sat at 1.24e-8~~ — that late-July figure is **superseded**; it does not match any live round and should not be used for target-setting. Assume winning Phase 2 means **≲4e-8**. Ground-truth noise (~2e-10) says there's real room below the current leader.
- **The two currencies**: leaderboard score (engineering + statistics) and the contribution prize (mechanistic insight + a clean PDF write-up). These pull in the same direction until late September, when we split effort.
- **Our structural advantages**: (1) the data distribution is *fully known* — He-init MLPs — so we can mint unlimited labeled data offline; offline compute is unmetered and we have a recent NVIDIA GPU for it; (2) the cloud workspace runs the exact submission file against self-generated ground truth (local_sim/), so iteration never waits on the grader; (3) 50 submissions/day is effectively unlimited — the grader is our oracle for anything ambiguous (flopscope pricing, wall-time caps, Phase 2 config).
- **Fair play**: the flopscope wall-time loophole (RESEARCH_NOTES §5) is off-limits. It's explicitly against organizer statements, poisons the contribution prize, and a patch + re-grade could vaporize scores built on it. We build methods that win a *fair* re-run. If organizers patch it mid-phase, our relative position improves for free.
- **Unknown to resolve immediately**: Phase 2's exact config (same 256×32? new budget? floor still 0.1? λ still 1e11?). No new dataset revision exists yet and the kit still pins `v1-phase1`, so same-config is the working assumption — but our first graded run report (`admin_config`) + the forum will confirm within days.

## Phase 0 — pipeline online (this week, target: first submission within ~48h)

1. ✅ Cloud repo built: submission-ready `estimator.py` (covariance propagation + offline-fitted per-layer calibration), local simulation harness, docs.
2. Windows setup (SETUP_WINDOWS.md): install `uv`, `uv sync`, `whest login` with Samuel's AIcrowd API key.
3. `whest validate` → fix any API-surface mismatches (shim vs real flopscope) → `whest run` on the public mini split (real flopscope FLOP counts, real scoring) → **submit v0**.
4. Read the graded run report: `admin_config.wall_time_limit_s`, actual budget B, λ, floor, dataset shape → update RESEARCH_NOTES and this plan if Phase 2 differs from Phase 1.
5. Join the Discord / watch the forum for the Phase 2 announcements we haven't seen yet.

*Expected v0 landing: adjusted ~2–3e-6 (calibrated cov-prop at the 0.1 floor). Not competitive — but the pipeline, the score reports, and the calibration machinery all get proven, and we beat the published baseline from day one.*

## Phase A — the honest sampling floor (week 1–2)

Build the tuned RQMC estimator: scrambled Sobol or rank-1 lattice points (implemented in pure `fnp` — Sobol direction numbers ship as shipped weights), antithetic pairs, exact closed-form layer 1, dead/saturated-neuron pruning via propagated firing probabilities. All randomness seeded from `mlp.seed`.

*Target: ~1.5–4e-7 adjusted (validated publicly by two Phase-1 write-ups). Decision data: measure error-vs-samples slope to choose the C/B operating point (for 1/N methods, anywhere between 10% and 100% budget scores the same — spend where the curve says).*

## Phase B — the hybrid (weeks 2–4; the expected winner)

Layer the classic variance-reduction stack, each piece validated in local_sim before grafting on:

1. **Control variate**: run covariance propagation analytically; on the *same* sampled inputs, evaluate a coupled linearized/frozen-mask surrogate whose exact mean we know; sample only the residual. Fit per-neuron CV coefficients offline (they're properties of the distribution, not of the specific MLP — shippable).
2. **Smoothed tail (Rao-Blackwell)**: at layers L−1/L, replace sampled ReLUs with conditional Gaussian expectations given the sampled layer L−2 state — closed forms exist; kills most of the last-layer variance at zero extra samples.
3. **Stratify/condition on ‖x‖** (the input radius drives a big shared variance component; 1-D quadrature over it is nearly free).
4. Budget tuning: measure marginal MSE per FLOP for each component; allocate.

*Target: ≤5e-8 adjusted, i.e. top-3 territory on Phase-1 numbers. Milestone gate: if we're not ≤2e-7 by Aug 21, simplify and consolidate instead of adding pieces.*

## Phase C — learned residual corrector (weeks 3–5, GPU)

Mint 10⁵–10⁶ He MLPs on the RTX GPU with high-precision MC labels (torch, float32, batched — each 256×32 MLP takes ~seconds at 1e6+ samples). Train a *small* per-neuron corrector (permutation-equivariant features: analytic μ̂/σ̂, firing probability, row norms, Gram-overlap stats, depth index) to predict the hybrid's residual. Distill to pure-`fnp` inference (a few matmuls, ~1e7 FLOPs) + shipped `.npz` weights (≤50 MiB).

*Target: 1.5–3× on top of Phase B → ~1–2e-8. Risk: publicly-reported failures of naive correctors near layer 32 (residual ≈ GT noise). Mitigation: train to predict the *bias* component only (averaged over input noise), validated on held-out self-generated MLPs against 1e9-sample local truth for a handful of networks.*

## Phase D — the mechanistic bet (parallel, best-effort; the $20k track)

Depth-robust cumulant propagation: start from ARC's public `mlp_cumulant_propagation` code; add (a) low-rank K=3/K=4 cumulant tracking in the top-k covariance eigendirections, (b) per-layer renormalization fitted offline (the generalization of our per-layer calibration — same trick, deeper object). This is ARC's own stated wishlist (townhall). Even a partial result + clean analysis = strong contribution-prize entry. Phase 1's write-up deadline showed the format: one PDF + one submission ID; Phase 2 deadline **Sep 26**.

## Operating cadence

- **Cloud workspace** = development + local_sim validation (exact submission file, self-generated truth). **Windows PC** = official `whest run` + submissions + GPU offline training. Repo lives in `Documents\arc-whest`; sync via this Claude session.
- Every experiment gets a row in `experiments/LOG.md` (idea → local MSE → graded score → keep/kill).
- Submit *at least* daily once the pipeline works — the grader is free information (real FLOP pricing, wall-time headroom, private-split variance). Keep the best-scoring submission clearly tagged; Phase 2 requires designating one.
- Weekly: re-crawl forum/Discord for rule changes (esp. flopscope patches), re-run the leaderboard snapshot, update RESEARCH_NOTES.
- Writing: keep a running `docs/writeup-draft.md` from week 2 — the contribution prize rewards analysis quality, and it's brutal to reconstruct in the last week.

## Risk register

| Risk | Odds | Mitigation |
|---|---|---|
| Phase 2 changes architecture/budget | Medium | Estimator is shape-generic; calibration keyed by (width, depth) with safe fallback; refit within a day of any announcement (GPU makes refits cheap) |
| Wall-time loophole patched → leaderboard reshuffles | Medium-high | We never depend on it; pure upside for us |
| Loophole NOT patched → exploiters keep inflated scores | Medium | Final ranking is a private re-run; organizers signaled enforcement ("must not exploit"); contribution prize unaffected; if it stands, honest ceiling is still top-tier |
| flopscope API drift breaks submission (fnp surface mismatches) | Low-medium | `whest validate` catches locally before submitting; keep estimator's API footprint minimal and documented |
| Sobol/QMC implementation cost under flopscope pricing | Medium | Ship precomputed point sets as `.npz` (loading is ~free); measure real pricing in run reports before committing |
| GT noise floor limits gains below ~1e-8 | Low | That's *good* for us — compresses the top; win on consistency across the private set |
| Time (Samuel learning + building from zero) | — | Claude drives implementation; Samuel runs pipeline + learns via BRIEF curriculum; scope gates at each phase |

## Success criteria checkpoints

- **Aug 6**: v0 submitted, graded, run report parsed. Phase 2 config confirmed.
- **Aug 14**: Phase A live, ≤4e-7. | **Aug 28**: Phase B live, ≤1e-7. | **Sep 11**: Phase C live, ≤3e-8, freeze-candidate chosen. | **Sep 16–19**: hardening only — determinism re-runs, failure-mode audit (the private re-run punishes fragility, not slowness). | **Sep 26**: contribution write-up submitted.
