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

## Phase B — FUSION (revised Aug 5 after Phase A; the expected winner)

Phase A's measurements rewrote this section. The original plan (control variates, radius stratification, Rao-Blackwell tail) is **dead on arrival**: measured against real ground truth, the layer-1 control variate's ceiling is 1.80× and it adds *nothing* on top of QMC (both remove the same variance); radius stratification is 1.20×; antithetic is 1.03×. See `experiments/LOG.md` 2026-08-05.

What the data *did* show is that our two estimators fail in orthogonal ways:

| | error character | size |
|---|---|---|
| v0 covariance propagation | **pure smooth bias**, per-neuron, ~zero common mode | MSE 2.5e-5 |
| v1.1 Sobol RQMC | **pure variance**, flat in N (α=1.00), **42% in one common scale mode** | MSE 1.49e-6 |

Two estimators with uncorrelated errors are worth more together than apart. Hence:

### B1 — precision-weighted fusion (`estimator_v2.py`)
Run both (covprop costs 3.3e9 FLOPs ≈ 4.7% on top of sampling's 7.0e10), then combine in closed form, O(n) online:
1. **Scale-mode kill**: estimate the sampling run's common scale fluctuation ĝ by regressing s on p across the 256 neurons, weighted by 1/Var(ε_i); shrink ĝ by its own estimation noise; correct s′ = s/(1+γĝ). This attacks the 42% directly, using covprop as the reference direction.
2. **Per-neuron GLS blend**: t̂_i = w_i p_i + (1−w_i) s′_i with w_i = Var(s′_i)/(Var(s′_i)+Var(covprop residual_i)).

Error models are fitted offline on the **full 1000-MLP split** (1e9-sample truth = zero label noise) and ship as literals: sampling noise from covprop's own in-flight σ̂² (rescaled — covprop underestimates post-ReLU variance ~30%), covprop residual variance as a low-dimensional function of (σ̂_i, α̂_i).

*Target: graded ≤2.4e-7. Gates: mini ≤3.2e-7, 0 failures, C/B 0.26–0.30, determinism.*

### B2 — push the last k layers analytically
The fusion-hurt map (neurons where the covprop error model is wrong) tells us where the analytic side is untrustworthy. B2 replaces the last k layers' sampled ReLUs with a better analytic treatment, k chosen from that map.

### B3 — GPU learnability probe (2-day box, parallel)
Mint 50k He MLPs on the RTX 5090 (38.3 TFLOP/s measured), label with 64k–256k-sample MC, train a small per-neuron regressor to predict the *calibrated covprop residual*. **R² > 0.5 → Phase C is alive** (a learned error model replaces B1's static one); **R² < 0.15 → drop Phase C**, GPU pivots to B2 calibration data.

*Overall target: ≤5e-8 adjusted. Milestone gate: if not ≤2e-7 by Aug 21, consolidate instead of adding pieces.*

## REV 7 (2026-08-06) — endgame doctrine: write-up primary, governance, stewardship, opportunistic

*Repo docs are canon; strategy verdicts arrive in prompts and get folded in here.
Rev 6 below is the Claude Code rev — the strategy session's own rev 6 and §2h never
synced, so both were merged into this file rather than either overwriting the
other.*

**Score work is closed** (rev 6) and stays closed unless *external* evidence
reopens it. The four workstreams from here, in priority order:

1. **Write-up — primary.** `docs/writeup-draft.md`, full draft by **Sep 5**, due
   Sep 26. Now has the cost-ceiling table and DP result as §4b's centrepiece, §8's
   R-bootstrap and metric-tail decomposition, three rendered figures in
   `docs/figures/`, and a stub at §7b for the NNGP correlation-collapse theory with
   the measured numbers it has to explain. Appendix F stays private.
2. **Governance.** The organizer email is final and waits on Samuel. Note that two
   of its five paragraphs were *retired by evidence* during the Aug-6 crawl — the
   organizers ruled on residual billing in 18122/18132 — which is the model for how
   this workstream should behave: crawl, then ask only what is actually open.
3. **Stewardship.** The estimator is a designation candidate, not a work in
   progress. v2.4.1 hardened the one real defect (an irreproducible rotation).
   Remaining stewardship: keep `edge_cases.py` at 24/24, keep the fresh-MLP
   distribution current if the shape changes, and re-validate pricing on any
   flopscope/whestbench release (done for 0.10.0 — all our graded numbers are
   already post-change).
4. **Opportunistic.** Phase-2 opening (event-triggered), the Phase-1 write-up
   harvest through Aug 17, and anything the organizers answer.

**Ranking-noise nuance, corrected.** The fresh-MLP score distribution is heavy
(max/median 12.9×, p90/median 1.50× at R=10), but the re-run suite is **common
across teams**, so hardness is a common-mode term that inflates or deflates
everyone together and largely cancels in the *ranking*. What survives is the
interaction between draw and method. Ranking jitter between two similarly-placed
teams at R=10 is therefore **±10–20%, not ±50%**. Both numbers go in the write-up;
conflating them overstates the case.

## REV 6 (2026-08-05, late) — the honest sampling frontier is declared at ~2.7e-7

**Sprint 3 (structure-aware FLOP allocation) came back at 1.0246× against a 1.20×
gate, and the ceiling is arithmetic.** Details in LOG. The short form: for a
confidently-on neuron ReLU is the identity, so at the scored layer it needs no
per-sample value and at a feed-forward layer it is linear and composes — both
verified, the second to within 0.07% MSE over eight layers. But under `(2K−1)MN`
billing the multiply is the cost, not the nonlinearity, so the reduction is worth
the share of the cost model it touches: **the scored layer is 3.67% of the
per-sample budget, so freeing it entirely is 1.038×.** Clearing 1.20× would need
six of the last layers to cost nothing.

**Therefore: we stop optimising score.** Every remaining known lever is small and
additive — a hand-rolled normal quantile in place of `stats.norm.ppf` (~1.4%), the
Sprint-3 scheme (~2.5%) — perhaps 1.04× in total, for new code paths on an
estimator we are scored on **once**, on MLPs we have never seen. That is a bad
trade. v2.4 stands as the designated submission unless Phase 2 changes shape.

**What the effort goes to instead, in priority order:**

1. **Robustness on fresh MLPs** (`local_sim/robustness.py`, `edge_cases.py`). The
   private re-run scores one designated submission on unseen draws, so the
   distribution's tail is the risk, not its mean. Edge-case suite on the shipped
   file: **24/24 pass**, including degenerate pruning, a identically-zero layer,
   1e4 dynamic range, seed 2⁶³−1, widths 64/128, depths 2–64, budget 0, and input
   immutability.
2. **The write-up** (`docs/writeup-draft.md`), due Sep 26. This is now where the
   remaining expected value is. The negative results carry mechanisms *and* oracle
   bounds, and the on-neuron reduction gives it a theorem with a price tag.
3. **Phase 2 config verification** (Aug 8). If the shape changes, re-measure before
   optimising: α-histogram → error-vs-N → τ → stack. v1.1 stays the shape-generic
   fallback.
4. **Governance**: the organizer email is drafted and waits on Samuel. Nothing on
   the forum.

**Scoring policy, restated because it now drives everything**: judge on the local
100-MLP mini split and on ≥1000 fresh minted MLPs. Public-board rank is not a
target — §2g establishes the board is a fixed-suite reconstruction game — and our
own graded runs are *paired re-measurements* on 50 fixed MLPs, useful only against
our own prior submissions.

## REV 5 (2026-08-05, night) — both analytic revivals are bounded out; the board is signal-poor; optimise for FRESH MLPs

Rev 4 retired variance reduction as a growth area. Rev 5 closes the two tracks that
were meant to reopen it, and changes what we optimise *for*.

**Track T (shrinkage-moment tail) — dead, 0.9998× against a 1.10× gate.** Three
independent bounds, any one of which is sufficient (LOG 2026-08-05 night):
1. Only `diag(WᵀΣW)` enters the final-layer means, and that equals
   `mean(z²) − mean(z)²` exactly — so there is no covariance matrix to estimate,
   no Marchenko-Pastur regime, and v̂ is already accurate to **1.13% rms**.
   Replacing it with the *exact* variance — the oracle for every denoising scheme,
   linear or not — still gives only **0.611×** of v2.4.
2. The analytic tail's residual is **41% irreducible model bias**; with exact
   moments it beats sampling **1.474×** (independently reproducing 18097), but the
   exact moments are the very thing sampling estimates, and at N=16384 the
   mean-estimation noise alone is 57.9% of the error.
3. Blending is capped by ρ = **0.77** between the two errors. In the α ≥ 3 band —
   **76.5% of the MSE** — ρ = **1.000**, because there Φ=1 and the "analytic tail"
   *is* the sample mean. Oracle per-band blending: 1.0003×.

**Track P (rank-k deviation propagation) — bounded out at 1.093× against 1.25×.**
Stage 1's energy gate passes handsomely (top-64 holds 96.2% at layer 16, and the
closure's own covariance eigenvectors find the subspace for free), but propagating
in a rank-k subspace *is* a rank-k factorisation of each weight matrix, and under
shape-based billing two factors only beat a dense 256×256 matmul for **k < 128** —
which is exactly the rank at which truncation stops costing accuracy. Even pricing
an oracle-accurate Jensen correction at **zero**, the best cell in the whole grid is
1.093×. No prototype: the stage-1 data already bounds one.

**The board is signal-poor, so stop steering by it.** Measured Aug 5: the public
score is the mean over **50 fixed MLPs whose exact per-MLP MSEs are returned with
every submission**, the suite is identical across all four of our submissions, and
257 submissions suffice to solve for those references outright (RESEARCH_NOTES §2g).
The Phase-1 leader has 917. Consequences for planning:

* **Optimise expected score on FRESH MLPs.** The final ranking is a private re-run
  on MLPs nobody has seen; that is also what our local 100-MLP split proxies.
  Rank on the public board is now a weak target and a misleading one.
* **Our own graded runs are paired re-measurements**, not fresh samples. Useful for
  A/B against our own prior submission, useless as evidence about the population.
* **We do not touch that channel** (§2g rule 1). It is not the wall-time grey area.
* The email to organizers states the observation neutrally and asks whether it is
  intended. Nothing goes on the forum.

**What is left that could still move the score**, in order of expected value:
1. **Cost (c).** Still the only channel with headroom, and the one the field ignores.
   v2.4 spends 2.86e6 FLOPs/sample; the irreducible dense-matmul floor at the
   measured 80% alive fraction is ~2.8e6, so *this* design is nearly optimal — the
   next win has to change the design, not tune it. Pruning τ is at a sharp optimum,
   the op-count work is banked, and residual wall time is 1.83% on the grader.
2. **The write-up** (`docs/writeup-draft.md`). Our negative results now come with
   mechanisms *and* oracle bounds, which is the strongest form of the genre. Due
   Sep 26. This is where the remaining expected value actually is.
3. **Phase 2 config verification** (Aug 8) and a shape-generic fallback. If the shape
   changes, re-measure before optimising: α-histogram → error-vs-N → τ → stack.

**Standing measurement policy is unchanged and was vindicated twice more**: the
Track T and Track P verdicts both came from the full 100-MLP split with oracle arms
that bound the entire family, not from a promising subset.

## REV 4 (2026-08-05, evening) — variance is at its floor; we work cost, governance, and the write-up

Rev 3 retired the analytic family. Rev 4 retires **variance reduction itself** as a growth area, on evidence:

- **The common mode is irreducible variance, not correctable bias.** Five independent attempts to predict the per-MLP scale fluctuation g — closure reference at the final layer (R² 0.094) and mid-depth (0.155), point-set radial imbalance (−0.010), the exact closed-form layer 1 (−0.037), and a learned regressor over 3000 MLPs × 8 draws (out-of-fold corr 0.012). The official reason is **projection geometry**: g = ⟨ŝ−t, t⟩/⟨t,t⟩ projects onto the unknown truth direction, and every proxy for that direction carries more error along it than g itself. This predicted R² ≈ 0.09 before measurement. **No further prediction attempts.**
- **Point sets are bounded by theory.** He & Wang 2015: the RQMC rate bonus for discontinuous integrands is 1/(4d−2) = 1/1022 at d=256. Measured α = 1.00. Every variant lands within ±15% of plain Sobol, including direct gradient optimisation (G2 plateaued at 1.06×).

**Operating frame: adjusted ≈ V·c/B, V is at floor, so we work c.** Three tracks:

1. **Cost (c).** Dead-neuron pruning was the largest single win of the month (1.47×). Op-count engineering is the untouched channel — residual wall time was 10.8% of billed compute at 1794 ops/MLP.
2. **Governance.** The 18099 wall-time bypass has no organizer reply. A neutral, technical clarification email is drafted at `docs/email-draft-organizers.md` for Samuel to send from his own address. Never posted to the forum.
3. **The write-up** (`docs/writeup-draft.md`, private). Our negative results have mechanisms attached, which is what the contribution prize rewards. Due Sep 26.

## REV 3 (2026-08-05) — the analytic family is retired

Everything on the analytic side has now been measured against 1e9-sample truth and killed: B1 fusion (net ~0), B2 analytic tail (−67% to −214% at every k), B3 learned covprop residual (held-out R² = 0.012 on 50k MLPs). Covariance propagation is 18× worse than sampling at layer 32 and its residual is not a learnable bias. **We stop trying to make the analytic estimator contribute.**

What replaces it follows directly from the scoring identity `adjusted = V·c/B` (RESEARCH_NOTES §3d): the only two levers are per-sample **variance** and per-sample **cost**.

### Sprint 1 — cheap wins stacked on v1.1 (before Phase 1 closes Aug 7 23:59 UTC)
1. **Dead-neuron pruning** (cost lever). Predict dead neurons from covprop's α, slice both sides of each matmul to smaller dense shapes, compensate with the exact constant offset, report pruned neurons with their closed-form rectified-Gaussian mean.
2. **Rotation + radius** (variance/rate lever). Fold an alignment basis Q into W1 (zero marginal per-sample cost); put an exactly stratified chi₂₅₆ radius on one Sobol coordinate.
3. **Offline base-set choice** (free). Compare the shipped Sobol set against orthogonal-block + antithetic variants and Owen scrambling.

*Gate: combined ≥1.2× over v1.1 (mini adjusted ≤3.4e-7), n_failed=0, determinism → spend ONE reserve on Phase 1 before close. Otherwise submit nothing.*

### Track G — GPU, through ~Aug 12
- **G1 amortized common-mode regressor.** 48% of the error is one per-MLP scalar worth 1.9× if removed exactly. Train offline over (MLP × K sample-set draws) to predict it from cheap online observables; cross-fitted and James-Stein shrunk by out-of-fold R² so it cannot add net bias. Ship as inline float literals. *Gate: ≥1.3× held-out, confirmed on real mini.*
- **G2 task-optimized point set.** Optimise the base point matrix directly against minted-MLP final-layer error; regularise with exact moment matching and drift penalties. *Gate: ≥1.3× held-out + mini.* G1 and G2 overlap on the mode — report the **combined** gain, never the product of solo gains.

**Standing measurement policy**: no operating-point or variant decision on fewer than the full 100-MLP mini split. Three separate "wins" (LMS scrambling, C/B=0.5, a +33% rotation) evaporated when re-measured.

## Phase C — learned residual corrector — RETIRED (see REV 3; B3 measured R² = 0.012)

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
