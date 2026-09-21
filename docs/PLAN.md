# PLAN — ARC White-Box Estimation Challenge, Phase 2 (rev 12, 2026-09-21)

**This is the only plan.** It supersedes `game-plan.md` rev ≤11, every `STRATEGIST_*`, `*_HANDOFF_*`, `STRATEGY_REVIEW_*`, the Sep 18 packet and its "2× release gate". It answers `docs/STRATEGIST_PACKET_2026-09-18_COMPLETE_SLICE.md`. Labels: **[M]** measured / read from a primary source today, **[I]** inference.

Samuel's decisions today (Sep 21): (1) **standing gated approval** for non-designated canary submissions, up to 5/day, gates in §5; designating finals and any external message stay his alone. (2) **Organizer public labels are unlocked** for development and fitting: 200 fit/dev nets + 800 holdout. (3) **Process cut to the minimum** (§4). (4) Strategist writes this plan; the worker archives old repo docs as Deliverable 0.

---

## 1. Where we stand [M]

Board, Sep 21 20:30 UTC, 184 teams: #1 J2W 2.10e-9 (raw 1.64e-8, 0.131 B, 303 entries); #2 marius_binner 2.40e-9 (1.68e-8, 0.146 B, 151 entries); #3 suliman_tadros / a_s6 2.50e-9; #5 Puffi / reds 2.80e-9; #10 mliston 3.50e-9; ranks 13–41 are trimmed derivatives of 504aldo's public code at 4.2–5.3e-9; ranks 42–57 are sixteen exact clones at 5.40e-9. **We are #71 at 8.75e-9** (v31 #331317, raw 2.91e-8, 0.301 B, wall 62–68 s, 15 entries total, none since Sep 16). Close Oct 17 23:59 UTC, team freeze Oct 2, write-up Oct 24. Phase-1 winners' code is still unpublished.

Best local candidate: **C10** (`estimator_k3_v43_complete_c10.py`), 100-net organizer mini = 6.767e-9 → **≈6.5e-9 graded** (raw 3.1e-8, 0.2187 B, 23,421 ops, 5.8 GiB). It has been ready and unsubmitted for three days.

Two facts that set the size of the job. Even at the 0.1 cost floor C10's predictions would score 2.97e-9, so **accuracy is mandatory, not optional**. And the public MIT baseline run natively on our own nets (raw 2.38e-8 at 0.253 B) still beats our week-long re-derivation of it (C10), so **we are currently behind a free download**.

## 2. What we have been doing wrong

1. **Not submitting.** 15 entries in 25 days against ~250 available; the leaders have 150–300 each. The public board (50 nets, 1e9-sample truth) is the best instrument in the competition and we do not use it.
2. **Process ate the project.** 291 commits since Sep 14; >50% instrument/protocol work, 75% on Sep 18 (143 commits, 0 submissions). The last experiment (P15) took 5 h elapsed for 36 min + 4×20 min of compute, died once on a missing sidecar, and its decision reader was never run. `experiments/LOG.md` is 55k lines; there are 109 root estimators and 376 files in `local_sim/`.
3. **Self-imposed data starvation.** The repo forbids choosing anything on the organizers' 1000-net public split (1e9-sample labels, noise ≈9e-11) and instead mints 24-net blocks with truth noise 2.25e-9 — as large as the leader's whole score — at 36+ min each. Every paired gate has been fought against that noise.
4. **A 2× release gate.** Nothing ships unless it doubles v31 on a single-use block. Leaders climb in 3–8% steps.
5. **Searching for "one causal intervention".** The Sep 18 packet asks for a single mechanism worth 31.7%. The evidence says the frontier is reached by many small fitted and engineering gains on a correct chain (§3), iterated fast.

Measurement itself is **sound** and needs no more work: mints are bit-identical to the organizers', FLOPs match the grader exactly (grader-twin f64 read-only arm; subprocess runner reads 2^25 low), and local 100-net mini → graded has been 1.0386 / 1.0385 / 1.0339 on three submissions (±0.3%); `predict_public.py` missed v31 by 0.45%. **Predicted public = mini score / 1.037.** We can predict our score offline. The problem is throughput only.

## 3. Architecture verdict

**The family is right and is the leaders' family** [M/I]: deterministic cumulant propagation — mean + full covariance, factorized third cumulant with a young exact tier and a compressed old tier, regenerated κ4 (exact diagonal + λ·C_off), Strassen products, f32, single file. Evidence: every top-12 entry has chain-like telemetry (deterministic, 77–117 s wall, 0.13–0.19 B); the leader's all-layers/final error ratio (0.41–0.45) equals ours (0.41), i.e. depth-accumulated truncation error; sampling, QMC, control variates, mixtures and generic learned correctors are each measured dead by us or by others (residual-sampling arithmetic: a 0.05 B sidecar needs a 1600× variance reduction; best measured 1.5×). Do not change family.

**What we had not identified** is what separates 2.1e-9 from 5.4e-9. My reading of the evidence:

- *Accuracy (raw 2.13 → 1.64e-8) is mostly end-to-end fitted coefficients on structurally correct terms.* 504aldo's single fitted scalar per layer on the κ4 form was worth ~1.75×; our own four coarse-gridded controls gave 1.145× held-out; Hilton says the best warm-up entries "combined [K3] with learned networks that consume the cumulant estimates"; a leaked grader message from a top team mentioned "ridge regression and gradient boosting"; J2W moved raw 1.87 → 1.69 → 1.64e-8 in small steps over ~300 submissions, which is what constant-fitting looks like, not new algebra [I]; and #330001 showed the dominant error is *propagated drift* (5.2× the local one-step residual, correlation −0.05 with local features) — which is exactly why post-hoc correctors give 1.01× and why the fit must run *through* the chain. We fit 4 numbers on 24 noisy nets. The data to fit ~30–130 numbers on 200 noise-free nets has been sitting on disk.
- *A second, principled accuracy source exists* [M]: ARC's own repo (`TODO.md`, `kprop_harmonic.factored_keeps_term`, Aug 2026) states that factored AUGMENT **drops terms of the same Θ(n^-K) size as the ones it keeps** (non-hypertree diagrams such as covariance cycles; products with the all-distinct K3 block). J2W, marius, ben3 and reds are all below 504aldo's measured f64 ceiling for "K3 + memoryless κ4" (1.94e-8), so somebody carries content beyond it.
- *Cost (0.25 → 0.13 B) is the old tier.* 504aldo's bill minus his old-source tier is 153 units = 0.150 B — the leaders' bill. Our P15 block says C10's rank-256 shared basis costs 1.23× raw (24/24 nets); his 384/224 two-tier costs ≤1.08× raw but 107 units. Nobody has disclosed how the leaders carry old content for ~free. Note also that **only the final layer is scored**: lode_dockx runs 0.116 B predicting only the last layer, and a_s6 (#3, 0.126 B) has a flat per-layer error profile — they are not paying for intermediate-layer accuracy.

**Target architecture** = C10's chain (kept) + (a) a *fitted coefficient layer*: smooth per-layer profiles for λ, closure scale, feed and feedback gains and per-term-class multipliers, fitted end-to-end on organizer labels, with per-network adaptation from label-free statistics; (b) rank-16 D21 feedback; (c) an old tier replaced or cheapened by a fitted surrogate / final-layer-aware schedule (W4); (d) the restored same-order diagrams if they price in (W3); (e) billing polish. Are we sticking to the architecture? Yes on the chain, no on the method of improving it: the last week was spent on exact-algebra archaeology against 504aldo's code under a noisy instrument. That stops.

## 4. The new loop (infra) — Deliverable 1, due Sep 23

One command, no queue, no contracts:

`python tools/ev.py <candidate.py> [--panel dev|fit|holdout] [--n 100]` →
raw MSE, paired ratio vs cached best (mean, t-95%, wins/100), projected public score (= mean(MSE·max(0.1,F/B)) / 1.037), FLOPs, ops, wall, peak RSS — appended as one row to `RESULTS.md`.

- **Accuracy path:** non-counting shim (`local_sim/flopscope`) with `WHEST_BACKEND=cupy` on the 5090; acceptance: 100 nets < 6 min, predictions equal numpy to 1e-6 relative on 2 nets. If cupy on Blackwell/Windows is not working within 2 hours, use a 6-process pool on CPU (≈13 min/100 nets) and move on. Predictions cached by sha256(source) — controls are never recomputed. Run directly; **do not use workerq/gpuq for evals**.
- **Cost path:** real flopscope 0.12.1, f64 read-only twin, **one net** (FLOPs are net-independent) → FLOPs (+2^25), op calls, scalar materializations. Two nets under WSL `RLIMIT_AS` 8 GiB → peak RSS and wall. Projected server wall = local wall × (v31 graded 68 s / v31 local wall). Never take cost from the shim (reads ~18% low).
- **Panels** (organizer `v2-phase2`): DEV = the 100 mini nets; FIT = 100 nets from the full split disjoint from mini (verify by weight hash); HOLDOUT = 800 remaining, read at most twice a week, only to check DEV/HOLDOUT gain ratio stays within 2%.
- **Retired:** fresh mints and seed blocks, SINGLE_USE/DECIDE blocks, P13, pre-registered contracts and predictions, dispatch-plan commits, producer sidecars, instrument preflight, per-experiment Skeptic, the 2× gate, the 22k-op screen (replaced by 24.5k, §5), `competition_score.decide` t-gates beyond the paired ratio above. Skeptic review is kept only for the two designated finals and write-up numbers.

**Decision rule:** paired DEV gain ≥3% with ≥70/100 wins → keep and canary. Otherwise drop, one line in `RESULTS.md`, next idea. Target cadence: ≥8 candidate evaluations and 2–5 submissions per day.

## 5. Standing submission gates (canaries, non-designated, ≤5/day)

DEV projected score ≥3% better than the best graded; `whest validate` passes; WSL 8 GiB gate passes; projected server wall < 100 s; ops ≤ 24,500 (leaders run 0.34–0.40 s residual; tighten or loosen after the first graded residual); setup < 2 s; no bare `except` that returns a constant. Log every graded row (score, raw, util, wall, residual, failures incl. sealed half) in `RESULTS.md`. If graded/projected deviates >2%, stop and recalibrate before the next slot.

## 6. Deliverables

**D0 — tonight (Sep 21): submit C10** under §5 (it passes; 23,421 ops). Expected ≈6.5e-9, rank ≈59. It also buys the first server residual measurement at 23k ops. Then archive (§8).

**D1 — Sep 23: the loop in §4**, with v31 and C10 rows reproduced (C10 DEV must read 6.77e-9 ± 1%).

**D2 — Sep 23–26: fitted coefficient layer (the main accuracy bet).** On C10 + P3 at rank 16 (504aldo: no loss 32→16; halves P3's +59 G):
 a. Parametrize six per-layer profiles (λ, closure θ, κ4→κ3 feed gain, D21-feedback gain, (2,1)-consumer gain, old-tier readout gain) as 4-knot splines → 24 parameters, all defaulting to current values.
 b. Derivative-free fit (CMA-ES or coordinate descent) of final-layer MSE on FIT, 25-net minibatches, on the GPU evaluator, run overnight; report DEV and one HOLDOUT read.
 c. Port v29's per-network adaptive λ rule (`λ_l ← λ_l · clip(ratio/REF_l, 0.5, 2)^β`, own code, own REF table from FIT) — never tried on our chain.
 d. *Residual anatomy* (free by-product): finite-difference sensitivity vectors ∂μ_L/∂p_j; report the share of each DEV net's final residual inside their span with global vs per-net-oracle coefficients. If per-net-oracle ≫ global, add label-free per-net predictors (as in c) for those parameters. The share outside the span is what only new diagrams (D4) can fix.
 Gate: ≥10% raw on DEV with HOLDOUT agreeing → canary. Kill: <5% after 300 evaluations. Evidence range for the gain is 1.15× (ours, 4 parameters, noisy) to 1.75× (504aldo's λ); I do not know where in that range it lands.

**D3 — Sep 24–26 (parallel, mechanical): billing polish**, each an exact transform checked by bit-parity on 2 nets then one cost run: Strassen for the plain-`@` join and P2 products (C10 lines ~698–702, 1116–1121); true 15-addition Winograd form and pooled `out=` buffers instead of per-level `concatenate` (also cuts ops/residual; 504aldo ships `out=` and is graded); remove recomputed `A_L*M` / `C_L`; aliasing / `as_symmetric` for genuinely symmetric products. Target −6–10% FLOPs and −15% ops.

**D4 — Sep 22 (one day, no labels needed): price the missing diagrams.** Run ARC's repo at matched L/n = 1/64 — (n,L) = (128,2), (256,4), and (512,8) if it fits — for factored-SIMPLE, factored-AUGMENT, unfactored-AUGMENT K3 and factored K4. If unfactored/factored AUGMENT ≥1.3× in MSE at two shapes, derive the covariance-cycle and all-distinct-product terms from `get_all_terms_iso` minus `factored_keeps_term`, implement with Σ_off ≈ U Uᵀ (r = 64; our measured effective rank ≈60) at +2–3 units/layer, give each a fitted gain (D2 machinery), and test on DEV. If <1.15×, close it in one line.

**D5 — Sep 27–Oct 3: the old tier (the cost bet).** On the unshared chain, log per layer the age>4 tier's aggregate contributions to D3 (n) and D21 (n×n). Kill tests, each ≤1 day on the fast loop, in this order:
 H1 *regenerated old tier* — the κ4 trick applied to old K3: replace the old sources by a carried aggregate (one n-vector + one n×n matrix transported per layer, ≈2 units/layer instead of ≈7) with fitted per-layer gains. Gate: raw within +8% of unshared at ≤0.16 B total.
 H2 *refit under aggressive truncation* — age 3 / rank 192 lost 1.76× unfitted; rerun D2's fit on it (cost 0.19 B).
 H3 *final-layer-aware schedule* — only the last layer is scored: taper old-tier rank and switch off D21/old readouts over the last 2–4 layers, births at layers ≥ L−2 keep only the (3)-diagonal readout. Expected −5–10% FLOPs.
 H4 *linear-response hand-off* — after age 4 read each source out once to the final mean through J = Π D_k W_k, keep its intermediate effect only through the rank-16 feedback.
 Ship whichever pass; if none does, adopt the two-tier 384/224 schedule (own code) and stop spending on cost.

**D6 — Oct 4–10:** second fitting round on the final structure with more parameters (per-term-class × layer, ~130) and per-net adaptation; DEV/HOLDOUT check; canaries daily.

**D7 — Oct 11–16: harden and designate.** Wall ≤ 95 s, residual ≤ 0.33 s, RSS ≤ 7 GiB, three clean graded runs each (0 failures on all 100, including the sealed half — visible in the public telemetry). Designate A = best score, B = same method with more wall/residual slack. **This is a real edge:** the leaders run 100–117 s against a 120 s cap and 0.34–0.40 s against 0.4 s; J2W's board entry has three `TIME_EXHAUSTED` sealed nets and marius is resubmitting identical bytes to find a safe pair. A failed net scores ~0.9 vs 2e-9 — one failure in the private rerun ends their run. Never trade our margin for the last 3%.

**D8 — Oct 17–24: write-up**: "end-to-end fitted coefficients on a factorized cumulant chain" + the term-by-term κ4 anatomy + whichever of D4/D5 landed + negative results. One hour a week until Oct 10, not more.

## 7. Expected ladder and honest odds

C10 tonight ≈6.5e-9 (#59) → D2+D3 by Sep 27: raw ≈2.3–2.6e-8 at ≈0.20 B ≈ 4.5–5.2e-9 (#14–30) → D5 by Oct 3: ≈0.15–0.17 B ≈ 3.3–4e-9 (#8–12) → D4/D6: raw ≤1.9e-8 ≈ 2.6–3e-9 (#4–6). Probabilities as I see them: top-25 80%, top-10 35–40%, top-3 10%, #1 5%, plus some uplift in the private rerun from competitors' timeouts. The first rung is engineering; every rung after it depends on D2 or D5 paying, and I do not know that they will. What I am confident of is that the current process could not have found out.

## 8. Deliverable 0b — archive (worker, one commit, `git mv`, nothing deleted)

Move to `archive/2026-09-21/`: all `docs/STRATEGIST_PACKET_*`, `STRATEGIST_RESPONSE_*`, `CLAUDE_CODE_HANDOFF*`, `CODEX_EXECUTION_HANDOFF*`, `STRATEGY_REVIEW_*`, `E14_*`, `IMPLEMENTATION_PLAN_*`, `RESEARCH_STRATEGY_*`, `PHASE2_*`, `WORKER_REVIEW_*`, `CANDIDATE_PROVENANCE_*`, `ELIGIBILITY_REVIEW_*`, `RESEARCH_NOTES.md`, `SSC_PUFFI.md`, `KERDOCK_MUB_SPEC.md`, `MOONSHOT.md`, `FRONTIER_ARCHITECTURE_PROPOSAL.md`, `GAME_PLAN.md`, `writeup-*draft.md`, `email-draft-organizers.md`, `SKEPTIC.md`, `HUMAN_MODEL.md`, `experiments/LOG.md`, root `AI_DEVELOPMENT_OPERATING_SYSTEM.md`, `AI_PROJECT_STACK_SELECTOR.md`, and every root `estimator*.py` except `estimator.py`, `estimator_k3_strassen_L4_f21.py`, `estimator_k3_v31.py`, `estimator_k3_v43_complete_c10.py`. Keep: `docs/PLAN.md` (this file), `RESULTS.md` (new), `VALIDATION.md`, `GOTCHAS.md`, `PROVENANCE_504aldo.md`, `COMPETITION_BRIEF.md`, `SETUP_WINDOWS.md`, `CANDIDATE_MANIFEST.md`. Replace `CURRENT_STATE.md` with ≤100 lines. Replace the rules section of `AGENTS.md`/`CLAUDE.md` with: *"Read docs/PLAN.md. Evaluate with tools/ev.py. Record one row per candidate in RESULTS.md. Submit under PLAN §5. Own code only — 504aldo's files are a local oracle, never shipped; record provenance. Never designate finals or send external messages."* Retire `scripts/preflight.ps1` from the default path.

## 9. Closed — do not reopen

Sampling/QMC/control-variate sidecars; Gaussian-mixture closures; generic post-hoc correctors; exact augmented κ4 (−9% for ≥0.5 B); the standalone (3,1)/q31 consumer; rank-32 old-query readout (v44); qEc/qEdc bases; CP/Tucker leg compression; f64; E14; further "why does v29's chain differ from ours" archaeology (D1–D3 of Sep 16 — the fit in D2 replaces it); fresh-mint blocks; the 2× gate.
