# Strategist response — 2026-09-24: review, the leaders' picture, and the last two bets

*Answers `docs/STRATEGIST_PACKET_2026-09-23.md`. Board re-read Sep 23 20:00 UTC. Labels: **[M]** measured, **[I]** inference, **[?]** unknown. This is PLAN rev 12.3 for Sep 24 – Oct 2; §7 is the executable slice.*

## 0. Review of the work so far

The last four days are the best work in this project. The probe framework (100 nets in minutes, panel mean at 3e-5 of the grader, every graded row within 1% of projection) turned every one of my structural hypotheses into a same-day number, and the numbers were right: the single-leg tower, N1, tiered ranks, P3, adaptive λ, the closure tail. Each of my predictions this week was wrong in the optimistic direction and the worker's were right; that is on record and I have set my priors accordingly below. The wall finding (leaf 32 runs at ~1.9× the ps/FLOP of leaf 64) is the single most useful *server* fact anyone here has measured, and it produced a designable entry at 60.7 s.

Where we stand [M]: v55 5.320e-9 (raw 2.461e-8, 0.216 B), **rank 43 of 152**, 23 entries. Top of board Sep 23: J2W 2.1e-9 (raw 1.64e-8, 0.131, 313 entries, 3 sealed-net TIME_EXHAUSTED failures, walls 105–117 s), suliman_tadros 2.2e-9 (raw 1.63e-8, 0.138, **49 entries**, walls 84–108 s, 0 failures, all-layers 6.4e-9), marius 2.3e-9 (raw 1.56e-8, 0.147). Both top-2 have per-net FLOPs varying ±6%; ours are constant. No organizer news; the nomination deadline is still unpublished.

**Verdict on architecture:** the family is right and the chain is now at the local optimum of 504aldo's design. Everything on the "richer/cheaper/re-ranked/refit" list is measured and closed. That is not failure; it is the map. What it says is that the leaders' 2.5× is not inside this design's knobs, and the two remaining lanes (§3, §4) are the ones this project has never actually built.

## 1. The leaders, from the telemetry [M → I]

Three numbers pin them down. (a) Kernel rate: J2W (110 s − ~13 s overhead) / 2.87e11 ≈ 340 ps/FLOP; suliman ≈ 270 ps/FLOP; ours 165 (leaf 32) and 92 (leaf 64). Their arithmetic runs in kernels 1.6–2× slower per FLOP than our slowest, i.e. **leaf-16-class Strassen (depth 6–7) or einsum-heavy contractions**, and they sit at the 120 s cliff for it. (b) Nominal chain: 0.131 B at depth-7 Strassen (×0.39) is ~0.33 B nominal; ours is 0.216/0.51 ≈ 0.42 B nominal. So their **un-Strassened chain is ~20% leaner than ours** — age 4, no P2/hubs, or a smaller term table — not a different representation. (c) Raw 1.56–1.64e-8 with a leaner chain than ours (our exact unshared T0 is 2.27e-8 board-scale; ARC's unfactored AUGMENT K3 in torch was 1.91e-8). **Their accuracy is not bought with FLOPs.** It is 1.2–1.5× beyond the best analytic chain anyone has published, on a lean chain, with 313 (J2W) and 49 (suliman) submissions — suliman's count says it is not brute-force server tuning.

Put together: **lean chain × deep Strassen at the wall cliff × a non-FLOP accuracy layer.** The deep-Strassen half is closed to us by wall (leaf 32 is already 114 s worst on a noisy day; leaf 16 would be ~180 s) — and it is why they lose sealed nets to timeouts. The accuracy layer is the part we have never built, and it is the one Hilton named ("combined with learned networks that consume the cumulant estimates as features") and the leaked grader message named ("ridge regression and gradient boosting"). Our project's "learned correctors cap at 1.02×" comes from two post-hoc fits on noisy own-mint labels with a handful of features. That is not the same experiment.

## 2. Answers

**Q1 (what else could they compress adaptively?)** Only one candidate survives the evidence: **per-lineage adaptive age** — fold a lineage into `Q` when its energy (or its D3 contribution) falls under a tolerance, rather than at a fixed age. It produces per-net cost variation, it is on the only axis that has paid (age), and it costs a day. Prediction: −2–5% score [I, low confidence]. Covariance low-rank+diag: no — the pre-activation covariance is Wishart-like (flat MP spectrum) and only 14 units. κ4 channel: nothing to compress.

**Q2 (make age cheaper?)** No. Within the leg representation the 4 nominal units per source-layer (2 transports + 2 wide D21 products) are irreducible, and every cheaper representation of old content is now measured dead. The only cost left is glue: concatenates (2–3% of FLOPs) and Strassen additions — a block-permuted layout without `concatenate` is worth −3–5% and nothing on wall. Do it only if idle.

**Q3 (does the wall finding point at a mechanism?)** Yes, at deep Strassen and at the cliff (§1). It also means: do not chase depth 6. It confirms the robustness edge is real — J2W's board entry would lose ~3% of nets in a private rerun on a noisy day (each ≈0.9 → that entry is dead), and marius is resubmitting identical bytes to probe wall variance. Our designate-B at 60 s is worth something only if A is also safe; see Q5.

**Q4 (stop rule).** §3 is the last accuracy bet, with a hard kill by Sep 28. If it fails: from Sep 29 the write-up gets ≥70% of worker time; score work is limited to A/B maintenance and one canary/day for measured ≥3% gains. If it passes: score work continues to Oct 10 and the write-up starts Oct 6 with the corrector as its centrepiece.

**Q5 (designation).** Agree: **B = v56b now.** A = the best config with worst wall ≤ 105 s on a noisy day; `tr64` is the candidate and needs its one server reading — I recommend Samuel approve that canary (a designation measurement, not a score chase). Standing rule going forward: one slot per day may be spent on an A/B safety measurement that does not beat the best score.

## 3. The last accuracy bet: an in-loop learned mean corrector, trained on the all-layer labels

The organizer labels give **E[h_l] for all 16 layers**, noise-free, on 1000 nets. We have used them only to score layer 16. They are exact supervision for the per-layer mean update, which is the one channel where our chain's error is *observable*: the final-layer residual is a random projection of layer-15 residuals (which is why post-hoc correctors see correlation −0.05 and cap at 1.02×), but the **one-step residual at layer l given corrected inputs is local** (#330001 measured 0.975 correlation with Edgeworth features at one step). The chain is deterministic and cheap, so the corrector can be trained teacher-forced, layer by layer, and applied in the loop so its corrections propagate.

Design (build on v55; cost is elementwise FLOPs and ~30 op calls per layer):
- Per neuron, per layer, permutation-equivariant features from the chain state at layer l: μ_pre, σ_pre, μ_pre/σ_pre, D3/σ³, D4/σ⁴, the young-tier and old-tier shares of D3, row energy of D21 (Σ_k D21_ik²)/σ⁴, ‖W_l[i,:]‖, the chain's μ_l, Δμ from the previous layer, and the previous layer's applied correction (transported: (W_l c_{l−1})_i). ~15 features + squares and pairwise products → ~50.
- Model 1: per-layer ridge (16 × 50 weights). Model 2: a 2-layer MLP, 32 hidden, shared across layers with a layer embedding (~2k weights). Both embedded as constants in the file; inference is a few (n × 50) products.
- Training: FIT (100 nets) + 300 more from HOLDOUT (leaves 600 untouched); sequential by layer: run the chain with the corrector active for layers < l, fit corrector_l on target = true μ_l − chain μ_l; then next layer. Loss = final-layer MSE for model selection; report DEV and HOLDOUT-600.
- Also apply the correction to the *inputs of the next layer's closure* (μ_l feeds W μ_l), not only to the output array.
- Kill criteria: ridge < 1% on DEV → try the MLP once; MLP < 3% on DEV or HOLDOUT disagreeing by > 1 pp → lane closed, Sep 28. Predictions to record: strategist — ridge 3–6%, MLP 8–15% [?; my priors are miscalibrated this week, discount accordingly]; worker — record own.
- A free by-product first (half a day, do it before the corrector): **residual anatomy with all-layer labels** — chain μ_l vs true μ_l per layer on DEV: MSE by layer, and the residual binned by μ_pre/σ_pre and by D3/σ³. If one bucket or one layer band carries most of the error, that is a closure defect with a formula fix and it goes ahead of the corrector.

## 4. The cost bet: per-lineage adaptive age (one day, after §3's anatomy step)

In the join, replace the fixed `_SHARED_AGE` with a tolerance: a lineage is folded into `Q` when its carried Gram energy (already computed for the range finder) is below τ × the youngest lineage's, with age ≥ 3 as a floor and age 6 as a ceiling. Sweep τ so that the mean age is ~4.5–5; measure DEV raw, FLOPs (now per net — record min/mean/max) and wall. Prediction: −2–5% score at equal wall [I]. Kill: < 2%.

## 5. Not doing (closed by measurement this week; do not reopen)

Deeper Strassen (wall), tiered ranks, sensitivity bases, P3 at any rank, per-net λ, exact Gaussian tails, joint refit of the existing profiles (D6: 0.00%), κ4-on-M beyond the one run already in flight (report it, then stop), K4, richer term tables, single-leg, memoryless κ3, old-tier rank < 256.

## 6. Odds and what "winning" now means

Score: top-25 95%, top-15 50% (§3 or §4 lands), top-10 20% (§3 lands ≥10%), top-5 5%. Private rerun: if J2W and marius designate their board entries, timeouts move them down; a clean 4.5e-9 could finish 8–12. **Contribution prize:** this project now holds the only complete measured anatomy of a factorized K3 chain at 1024×16 — an exact unshared reference, the elasticity table, the age/rank law, the single-leg/regeneration/diagonal negatives, N1's spectra, and the leaf-size wall law — plus, if §3 works, the learned-corrector result with all-layer supervision. That is a stronger write-up than most top-10 teams will produce, and it is the outcome I would now optimize for alongside the score.

## 7. Slice contract (Sep 24 – Oct 2)

- **Sep 24:** report κ4-on-M; residual anatomy with all-layer labels (§3, by-product); `tr64` canary on Samuel's approval; commit this file and the packet.
- **Sep 25–26:** corrector ridge → DEV/HOLDOUT; if ≥1%, MLP; canary anything ≥3% as an (A, B) pair.
- **Sep 27:** per-lineage adaptive age (§4).
- **Sep 28:** decision on §3/§4; write `docs/DESIGNATION.md` with A/B and both noisy-day readings.
- **Sep 29 – Oct 2:** per Q4. Claims ledger `docs/WRITEUP_CLAIMS.md` starts Sep 29 regardless.
- Standing rules, §5 gates (as amended), and probe-before-inference stand. Every prediction written before the run.
