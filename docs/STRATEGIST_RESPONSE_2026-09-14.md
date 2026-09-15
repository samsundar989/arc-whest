# Strategist response — 2026-09-14 (evening): architecture verdict, the frontier, and the five-week plan

*Supersedes `phase2-strategy-2026-09-05.md` rev 10, the Sep 10-v2 response, and the earlier (morning) Sep-14 response that recommended measuring the E14-f32 κ4 core first — that measurement was done tonight in a cheaper form (§3) and the recommendation is withdrawn. Inputs: repo snapshot (code, docs, LOG through line 49,559), live AIcrowd pages read through Samuel's Chrome at ~21:30 UTC Sep 14, 504aldo's MIT repo (cloned and read), ARC paper/repo/blog, a web literature sweep, and one new numerical experiment run tonight (§3). Epistemic labels: **[M]** measured by us or read from a primary source tonight; **[R]** reported by a competitor/organizer; **[I]** inference.*

---

## 0. Verdict in six sentences

1. **The family is right.** Every top-10 entry is a factorized third-cumulant (K3) propagation chain with some fourth-cumulant (κ4) content; sampling, learned correctors and pure covariance propagation are all dead at this shape. Our repo picked this family on Aug 26 and that decision is validated **[M: board + 504aldo + our own AUGMENT measurement]**.
2. **The product is wrong.** F21 (#330950, adjusted 1.596e-8, rank 62/158) is an *exact, uncompressed, scalar-κ4* K3 tower. A public MIT baseline (504aldo v29) sits at 5.4e-9 — **3× better than us** — using the same κ3 content plus three things we lack: a fitted off-diagonal κ4 regeneration, a rank-16 D21→birth feedback, and old-source compression; and its clones have already been trimmed to 4.2e-9 **[M: board, repo]**.
3. **E14 (the K4 "tree") is not the missing piece.** Tonight's experiment shows E14's label-free λ is a different object from the λ that produces the 2× κ4 gain: opposite sign from layer 4 on, ~10× smaller in magnitude, 1.7× too large at layer 0, R² ≤ 0.2 **[M, §3]**. The κ4 gain comes from κ3-*sourced* κ4 production, which a Gaussian-tree diagram does not contain.
4. **The leaders (2.5–2.8e-9) sit at raw 1.7–1.9e-8 and 0.13–0.16 B.** Their FLOP bill equals 504aldo's *without* the old-source tier to three digits, their per-MLP FLOPs vary by ±6% (adaptive), and their wall is 71–94 s at 0.13 B **[M: submission #330876 ledger]**. Nobody public knows how they carry old sources that cheaply; the best-supported hypothesis is an old-source representation that is flat in source count (a merged low-rank core in a sensitivity-weighted basis), not a compressed version of the leg representation.
5. **The bottleneck since Sep 7 has been process, not physics.** Two fresh-truth measurements in five days, ~750 queued preflight jobs, 33 documents, 11k LOG lines, and a candidate held by an absolute 2e-5 tolerance at n=3 and a local residual gate the repo itself documents as 12–53× wrong **[M: LOG/doc census]**. No submitted candidate has changed raw MSE since Sep 7.
6. **A credible route to the top-8 exists in three weeks; a route to #1 exists but is not demonstrated.** The plan below reaches ~5.4e-9 in week 1 by re-implementing published mechanisms, ~3.6–4.3e-9 in week 2 with one novel lever (sensitivity-weighted old-source merging), and needs either κ4 memory or the leaders' unknown old-source trick to go below 3e-9.

---

## 1. Where we are — the numbers that matter

### 1.1 Live board (Sep 14, ~21:30 UTC) **[M]**

| # | team | adjusted | raw MSE | C/B | note |
|---|---|---|---|---|---|
| 1 | suliman_tadros | 2.50e-9 (detail 2.540e-9) | 1.89e-8 | 0.134 | all-layers MSE 7.82e-9; per-MLP FLOPs 2.77–3.11e11; wall 71–94 s |
| 2 | marius_binner | 2.60e-9 | 1.70e-8 | 0.155 | 101 entries |
| 2 | J2W | 2.60e-9 | 1.87e-8 | ~0.14 | 250 entries |
| 4 | Puffi | 2.80e-9 | 1.94e-8 | 0.146 | Phase-1 SSC author |
| 5–10 | oqaris, Camaro, ben3, thegamers, hyojun_kwon, a_s6 | 3.1–3.9e-9 | 1.8–2.7e-8 | 0.14–0.22 | |
| 28 | 504aldo (+13 clones) | 5.4e-9 | 2.13e-8 | 0.253 | MIT code public since Sep 10 |
| 62 | **neopok919 (us)** | **1.60e-8** | **3.78e-8** | **0.422** | F21 #330950; all-layers MSE 1.56e-8 |

Best raw MSE anyone has posted in Phase 2: **1.50e-8** (Insights page). The whole field's accuracy is bunched at 1.5–2.2e-8; the race is now cost at fixed accuracy.

Arithmetic to #1 from 2.54e-9: raw 1.9e-8 needs C/B ≤ 0.134; raw 1.7e-8 needs ≤ 0.149; raw 2.13e-8 (504aldo's) needs ≤ 0.119. The 0.1 floor is not in play for anyone with competitive accuracy.

### 1.2 Our stack, honestly

| | F21 (shipped) | 504aldo v29 (public) | gap |
|---|---|---|---|
| κ3 content | exact all-age tower (rank 2n/layer, never compressed) | same blocks; M-leg residual rank 16; ages >4 in rank-384 basis, >7 in rank-224 | v29 loses ~5% raw to compression |
| κ4 | one isotropic scalar c4 per layer | exact diagonal g + G_off = λ_l·C_off (16 fitted λ) + (3,1) slice + κ4→κ3 feed | **the 1.8× raw gap** |
| D21→birth feedback | none | rank 16 (−10% raw) | |
| Strassen | depth 4, min block 64, concat glue | depth 5, min block 32, pooled `out=` buffers, batched families | |
| cost | 432 units = 0.422 B | 260 units = 0.253 B | 1.67× |
| raw | 3.78e-8 | 2.13e-8 | 1.77× |
| adjusted | 1.60e-8 | 5.4e-9 | 3.0× |

(1 unit = 2n³ = 2³¹ FLOPs; B = 1024 units.) F21 is, to within riders, exactly 504aldo's V16b stage of Sep 3 **[M: code comparison]**.

### 1.3 Validation infrastructure — what is trustworthy **[M: audit]**

Trustworthy and exact: MLP mint (bit-exact vs `whestbench.sample_mlp`), counted FLOPs (928,457,637,478 local = graded, on every graded row; the +2²⁴/+2²⁵ cast offsets are modelled and predicted out of sample), the score identity, and the paired arithmetic (`competition_score.decide`). Local MSE runs a stable +3–5% pessimistic against the public 50 (the public 50 are a slightly easy draw; this will not transfer to the private rerun); paired ratios between two candidates transfer to <1.5%. Truth: torch f32 GPU, 2×2²⁴ replicas, floor ≈2.2e-9, correctly noise-corrected.

Broken or ceremonial: the **local-seconds residual gate** (local µs/op grows with op count; server is flat at 10–13 µs/op; the gate refused three submissions that graded 0/100 failures and forced three human waivers); single-use DECIDE truth blocks spent on exact FLOP transformations (P09 burned by a scorer crash on a 3e-13 raw delta); the Phase-1 vault and its preflight steps; `test_enforcement_labels`; the aborted-read-counts-as-spent rule. There is no single command that emits "predicted public score ± CI, wall/residual/memory go-no-go."

---

## 2. What the frontier actually is

### 2.1 What is public and verified **[M/R]**

- **ARC** (paper v2, repo): SIMPLE vs AUGMENT (κ4's non-traceless part in flight), FACTORIZED (κ3 at size n²·L). Reported only at width 256; the TODO says factorized-AUGMENT is now "the augmented term set minus what it cannot afford" and the paper "needs updating". ARC's public hints: "cumulant methods will beat QMC in the long run", "effective rank of the activation covariance decreases with depth" (low-rank refinement), Borel resummation of the divergent Edgeworth series, offline training on external MC data is legal.
- **504aldo** (rank 28, MIT): the full recipe above, a 25-item dead-end list, and the key negative results: memoryless closures 45× worse; any Hadamard product of factored legs has Khatri–Rao rank r² ("the wall"); old-source Tucker core in the shared basis costs r³/n² units and only wins below r≈150, under the accuracy cliff; adjoint (#20) no cost drop; learned corrections 1.01×; float64 +0.004%.
- **Puffi** (rank 4, Phase-1 write-up): "signed spectral closure" — κ4 regenerated from ≤4 covariance-response matrices with a 4×4 ridge-fitted Λ; Phase-2 numbers 1.94e-8 at 0.146 B; unshipped extra: refit Λ to the *consumed mean correction* (−13.6%). 504aldo could not reproduce SSC at 1024×16 (9.8e-7).
- **Our own torch measurement** (Sep 9): ARC `kind=AUGMENT` 1.912e-8 vs SIMPLE 4.005e-8 on 24 nets, 24/24 — the leaders' raw floor is exactly AUGMENT-class κ4. The factored AUGMENT port is 3.6–5.8× over budget. **[M]**
- **Leaders' telemetry** (#330876): 137 units; 50/50 clean; wall 71–94 s at 0.134 B (i.e. ~4× the wall per FLOP of F21 — many small ops or deep Strassen glue); FLOPs vary per MLP.

### 2.2 The two unexplained facts and the best hypotheses

**Fact A — accuracy 1.7–1.9e-8.** 504aldo's f64 K3+memoryless-κ4 chain ceils at ~1.94e-8; exact per-source κ4 buys −9% more (≈1.78e-8). marius_binner's 1.70e-8 is therefore either κ4 with memory (thin legs), a better λ target (Puffi's consumed-mean refit), or D21 feedback at higher effective rank. All three are testable on 504aldo's lean chain in a day each.

**Fact B — cost 0.13–0.16 B while carrying old sources.** 504aldo's bill without the old tier is 0.150 B — exactly the leaders'. So the leaders carry old-source κ3 information at near-zero marginal cost per source. A leg representation cannot do that (the Hadamard wall). What can: **a single merged κ3 object for all old sources, held in a fixed low-rank basis, contracted once per layer** — cost r³/n² units per layer regardless of how many sources it contains. 504aldo priced this at r=384 (54u, too dear) and r=224 (10.7u) because his basis is chosen by *forward energy* of the legs. Two published facts say a much smaller r should suffice if the basis is chosen by *what the output can see*: our Sep-14 memo measured the effective rank of the layer-16 correlation at ≈60 and the readout's backward sensitivity as rank-1 per neuron (cubes of rows of the mean-gate-linearised product Π D W); ARC's own "mechanistic L² sketching / relevance" principle says exactly this. The per-MLP FLOP spread of the leader (±6%) is what an adaptive rank from a spectrum looks like. **[I — the single most valuable hypothesis to test; §5, lever N1]**

### 2.3 What the literature adds (full sweep: `lit-sweep-deep-web-2026-09-14.md`)

Nothing in the annealed finite-width literature (Yaida, PDLT, Hanin, Nica–Ortmann, Celli 2026, Fischer, Segadlo, Bordelon) supplies κ3: E_W[κ3]=0, so it is purely quenched and must be computed per network — 504aldo's triage rule and our memo agree. Transferable items are few and concrete: (i) closed-form *universal* κ4 (≈4.1·l/n mean profile) as a zero-FLOP prior for the diagonal channel; (ii) exact non-zero-mean bivariate ReLU moments (Kuang–Lin 2026, Kan–Robotti) to replace degree-2 Hermite truncation of the off-diagonal covariance in high-|ρ| pairs; (iii) blocked Strassen–Winograd and upper-triangular-only symmetric products (billed (b+1)/2b) — 504aldo already uses both; (iv) deflated randomized estimators (Hutch++/XDiag) are only viable inside the rank-60 subspace — a second-order lever; (v) learned correctors have never exceeded ~1.5× on a Gaussian base and 1.01–1.02× on a K3 base in three independent hands (ours, 504aldo, yujigon). The literature does not contain the leaders' trick; it will have to be measured into existence.

---

## 3. Tonight's experiment: E14's derived λ vs the fitted λ that works **[M]**

*Files: scratchpad `exp_lambda/` (scripts, CSVs, logs); summary in `exp-lambda-e14-vs-504aldo-2026-09-14.md`.*

Question: E14's `_tree_core` produces the same retained core as 504aldo's V17 (G = diag(g) + λ·C_off, same 6/(n+4) constant, same consumers q4/q31/q22 with metric 2.0), but with λ *derived* from a connected-tree Gaussian diagram instead of fitted. If the two λ agree, E14 is 504aldo's 2× gain with a label-free derivation — a genuine algorithmic-contribution item. If not, E14 will not deliver the gain.

Method: 8 He-init 1024×16 MLPs (seeds 0–7; HF and PyPI are blocked from this container, so same distribution, not the same realisations — the chain's per-layer mean variance matches 504aldo's public-MLP telemetry to 1–3%), float64 Gaussian covariance chain with E14's own frozen Wick table and degree-2 off-diagonal closure, `_tree_core` ported line by line and checked against `local_sim/e14_tree_oracle.py::fast_h`, λ computed both as coded (post-ReLU C_out) and with 504aldo's definition (LS on transported pre-activation C_pre_off(l+1)). Index alignment: E14's λ from the call for weight l ↔ `LAM[l]` ↔ `lam_prefit_perdump[:, l]`.

Result:

| layer l | 504aldo λ (8-dump mean) | E14 λ, 504aldo's definition | ratio |
|---|---|---|---|
| 0 | 4.99e-3 | 9.14e-3 | 1.83 |
| 1 | 8.09e-3 | 3.08e-3 | 0.38 |
| 2 | 9.83e-3 | 1.35e-3 | 0.14 |
| 3 | 1.055e-2 | 4.2e-4 | 0.04 |
| 4 | 1.085e-2 | −7e-5 | −0.006 |
| 5–14 | 1.08e-2 → 7.26e-3 (smooth, all positive, per-MLP sd 2–10%) | −4.8e-4 … −8.0e-4 (negative plateau) | −0.045 … −0.10 |

Depth-profile correlation −0.01; best single scale factor 0.02 with 49% residual; E14's own model explains ≤21% of its off-diagonal core beyond layer 1 (504aldo reports R² 0.66–0.97 for the object he fits). A Monte-Carlo check at n=12 shows E14's H_off points the right way (cosine 0.96–0.9996 with the true Gaussian-order partial trace) but overshoots by 1.34–1.71× — the (3,1) repeated-index pattern is evaluated with the 4-vertex tree formula instead of the exact Wick expansion — which is the 1.83× excess at layer 0, the one layer where both objects are pure Gaussian-order production.

Reading: the λ that halves the error is the LS coefficient of the *augmented chain's* off-diagonal κ4 core, which is dominated by κ3-sourced κ4 production (ARC's k211/hub channel). E14 regenerates only the Gaussian-order tree from C_pre and derivative Wick coefficients; that piece is small, decays, and turns negative as |α| grows with depth. **E14 as specified should be stopped.** Its consumers and feed plumbing are correct and reusable; its λ must come from a κ3-sourced κ4 object.

The clean, label-free way to get that λ is already in our repo: run our torch `kind=AUGMENT` chain (`p2_augment.py`, 24 nets, measured 1.912e-8) on **our own minted networks**, record G_pre_off per layer, and fit λ_l = ⟨G_off, C_off⟩/⟨C_off, C_off⟩. No competition labels are involved, so there is no public-set overfitting question, and it is a derived-from-first-principles-chain quantity rather than a label fit. Predicted values: within ~10% of 504aldo's table (his was fit the same way on the public dumps' augmented chain). Cost: one GPU evening.

---

## 4. Do we have the right architecture? Point by point

| claim in our canon | status tonight |
|---|---|
| "K3 cumulant propagation is the family" (rev 8, Aug 26) | **Confirmed** by the entire top-40. |
| "Exact factorized K3 tower is the primary chassis" (rev 10) | **Correct as an oracle, wrong as a product.** At 1024×16 the exact tower's marginal κ3 content over a rank-16 M-residual + age-gated old tier is worth ~5% raw and costs 1.7× FLOPs. |
| "AUGMENT's gain dies under truncation; κ4 pays only in an exact tower" (Sep 10 theorem) | **Falsified in the form stated.** 504aldo's memoryless λ·C_off regeneration retains ~90% of AUGMENT's gain (4.46→2.26e-8) at zero n³ cost. What died in our T1 was *deleting* κ4 births without replacing them by a fitted regeneration; the "theorem" generalised from one ablation. Retract it from the write-up. |
| "Learned correctors: 1.016×, dead" (T2) | **Confirmed** independently (504aldo 1.01×, yujigon). Keep dead. |
| "Adjoint κ3 is the key idea" (Sep 14 memo, lever 2) | **Dead for cost at this shape.** D21(l) is consumed by the closure at every layer (the memo's own prediction that the (2,1) slice is load-bearing is right, and it kills the memo's adjoint): the backward object for intermediate layers is rank-n per readout neuron. 504aldo #20 and our agent's derivation agree. |
| "Per-layer mean correction, not readout, is where 95% of the error is" (memo lever 1) | **Confirmed and already exploited** by every K3 chain; not a new lever. |
| "Universal κ4 mean profile as a free prior" (memo lever 3) | **Plausible, cheap, untested** — but it addresses the *diagonal* channel; the off-diagonal λ·C_off channel is where the 2× is. Worth a 3-hour test after the λ table lands. |
| "Rank-60 effective correlation; triangles not free" (memo lever 4) | **Reframed**: the rank structure is the lever for *cost* (old-source basis), not for triangle accuracy. |
| E14 tree-projected scalar-metric K4 | **Stop** (§3). Keep the consumers; replace λ. |
| Compact full-rank bridge B0–B4 (own compression lane) | **Superseded** by the published age-gated shared basis; the f32-vs-f64 conformance gates that stalled it were testing rounding, not algebra (their own f64 run passes 841/841). |
| Strassen depth 4, wall-safe depth 2 | **Keep** and go to depth 5 with pooled `out=` buffers (504aldo: residual 0.29 s steady, 100/100 graded). |

**What the winning architecture looks like, as far as the evidence reaches:** a factorized K3 chain with (a) young sources as dense legs under the 4-unit contraction identity, (b) old sources merged into one fixed-basis object whose rank is set by output sensitivity, not forward energy, (c) κ4 as exact diagonal + λ·C_off regeneration with λ fitted to a κ3-sourced κ4 object, plus whatever κ4 memory can ride on thin legs, (d) rank-16 D21 feedback into births, (e) depth-5 batched Strassen–Winograd with pooled buffers, block-symmetric covariance products, exact final-layer trim, (f) f32 throughout. Items (a), (c-first-half), (d), (e), (f) are public and reproducible in a week; (b) and the κ4-memory half of (c) are the open research and are what separates rank 28 from rank 1. **We have not identified (b); nobody public has.**

---

## 5. Levers, ranked by expected score gain per day, with kill tests

Units: 1u = 2n³. Baseline after week 1 assumed at 504aldo parity: raw 2.13e-8, 260u.

| # | lever | class | expected effect | kill test (≤1 day each) |
|---|---|---|---|---|
| P1 | κ4 regeneration G_off = λ_l·C_off + exact diag + (3,1) slice + κ4→κ3 feed, λ fitted from **our torch AUGMENT chain** on own mints (control: 504aldo's table) | public mechanism, own λ | F21 raw 3.78→~2.1e-8 (×0.55) at +0 n³ | λ within 15% of 504aldo's per layer; F21+P1 on 24 P01 nets ≤2.3e-8 |
| P2 | 7→4 contraction identity (S_sep on A leg, M-residual rank 16 thin leg) | public | 432u→~300u; +1.6% raw | tiny-shape parity vs F21 to f32; 1024×16 ledger |
| P3 | D21→birth feedback rank 16 | public | −8–10% raw | ablation on 24 nets |
| P4 | Strassen L5, min block 32, pooled `out=` buffers, block-sym C_pre, final-layer trim | public | →~280u; residual ~0.3 s | ledger; op count ≤22k; graded canary |
| P5 | age-gated shared basis (384@age>4, 224@age>7, randomized range finder) | public | →~260u, +4.6% raw | rank sweep on 8 nets reproduces 504aldo's age law |
| **N1** | **sensitivity-weighted old-source basis + merged core.** Weight the join Gram by JᵀJ, J = Π_{k>l} D_k W_k with D = diag(Φ(α)) from the K2 pre-pass (0.03 B, already computed as the chain runs — use the transposed transport); merge all old sources into one Tucker core in that basis; rank chosen per MLP from the core spectrum | novel (ARC's "relevance"/L² sketching made concrete) | if r_eff ≈ 128–160 at equal error: old tier 107u→~20–30u → total ~170–185u = 0.17–0.18 B → 3.6–3.9e-9 (top 8) | on 504aldo's lean chain: plot raw vs r for forward-energy vs sensitivity-weighted bases at age 5–8; kill if sensitivity weighting does not halve r at +2% raw |
| N2 | κ4 memory on thin legs: per source, transported rank-16/32 κ4 leg (Zf-style) instead of memoryless regeneration | semi-novel (504aldo did not ladder it) | −5–9% raw (toward 1.94→1.8e-8) at ~15u | ablation on lean chain vs exact per-source κ4 (which gives −9%) |
| N3 | λ refit target = consumed mean correction (Puffi's unshipped −13.6% at P1 shape) | published idea, unshipped | −5–13% raw, free | refit on own mints; held-out 24 nets |
| N4 | exact κ4 transport metric WᵀW instead of 2I | ours, measured 1.07× raw | −7% raw, cost to be metered (diag channel only → O(n²)) | meter it; keep if ≤2u |
| N5 | universal κ4 diagonal prior (4.1·l/n·σ⁴) added where the chain has no κ4 memory | memo lever 3 | unknown, likely ≤3% | 3-hour injection test |
| N6 | exact non-zero-mean bivariate ReLU covariance (Kuang–Lin) for top-rank-64 pairs | literature | ≤5% raw, n² cost | swap-in on one net |
| N7 | per-MLP adaptive rank/age gates from the join core spectrum (free) | matches leader telemetry | ≤3% cost | measure spread |
| N8 | Strassen L6 (FLOP optimum 0.530 vs 0.558 at L5) | arithmetic | −4% cost, +ops | residual-gated |

Realistic composite: P1–P5 → ~5.4e-9 (rank ~25–30, matching the clones at 4.2e-9 if their trim is found). + N1 (if it works) → ~3.6–3.9e-9 (top 8). + N2/N3/N4 (raw → ~1.8e-8) → ~3.1–3.3e-9 (top 5). Sub-3e-9 needs N1 to deliver r ≲ 100 *and* raw ≤1.8e-8, or the leaders' unknown trick. State this plainly in every packet: **top-8 is the evidence-backed target; #1 is a research bet on N1.**

---

## 6. Five-week plan (Sep 15 → Oct 17; freeze Oct 2; write-up Oct 24)

**Operating rules for the whole period (replace the current protocol):**
- One fresh-truth score per candidate per day is the *minimum* cadence, on a standing 24-net P01-class panel plus the mini-100 paired predictor; single-use DECIDE blocks only for candidates with fitted constants.
- The local-seconds residual gate is deleted from every decision path; the op-count model (≤22,000 ops; 13.4 µs/op) is the gate, local seconds are printed as advisory.
- Skeptic review only for (i) numbers that go in the write-up and (ii) the two designated files. No Skeptic on infrastructure.
- Tiny-shape conformance uses *relative* tolerances at f32 (1e-4 rel) against an f64 oracle; absolute tolerances are banned.
- Submissions: 10/day are cheap; one graded canary per material change; never designate a file without 3 clean graded runs.
- A single command `predict_public.py --candidate X --control 330950` prints: paired ratio ± t-CI on mini-100, projected public score, FLOP multiplier {F, F+2²⁴}, op count → residual forecast, wall projection ×1.6, memory-gate verdict. Build it on day 1 (all pieces exist).

**Week 1 (Sep 15–21) — reach the public frontier with our own code.** Day 1: `predict_public.py`; retire the residual gate; post the rules question on derived code (below). Days 1–2: P1 λ fit from our torch AUGMENT chain (GPU evening) + F21+P1 scored on 24 nets (prediction: ≤2.3e-8) → graded canary at 0.42 B (prediction ≈ 9e-9, rank ~45). Days 2–5: P2, P3, P4 into a new file `estimator_k3_v30.py` (v29 as parity oracle at tiny shapes; our F21 as the exact oracle); graded canary (prediction ≈ 5.5–6e-9 at ~0.27 B). Days 5–7: P5 age-gated basis; canary (≈5.2e-9 at 0.25 B). Deliverable: a file at 504aldo parity that is ours line by line, with provenance.

**Week 2 (Sep 22–28) — N1, the research bet.** Run the N1 kill test on the lean chain first (2 days). If it passes: implement the merged sensitivity-weighted old core in v30, rank sweep, per-MLP adaptive rank (N7). Canary target ≤4.0e-9. If it fails: fall back to the clones' known trims (7–22% cost, 504aldo's own reply says they exist: candidates are dense-leg count, regeneration schedule, join overhead) and to N8; target ~4.5e-9.

**Week 3 (Sep 29–Oct 5) — raw levers.** N3 (λ refit to consumed mean, free), N4 (exact metric), N2 (thin κ4 legs), N5/N6 if time. Each is a one-day ablation on the lean chain, then port. Target raw ≤1.9e-8 at ≤0.18 B → ≤3.4e-9. Team freeze Oct 2: decide solo vs team by Sep 30.

**Week 4 (Oct 6–12) — harden and nominate.** 1000-mint label-free soak of the two candidate byte-images (wall, ops, memory under RLIMIT_AS on Linux with the grader's Python 3.10/NumPy 2.2.6), determinism twins, 3 clean graded runs each, designate A (best) and B (robust, ≥1.6× wall margin). Write-up sections drafted from measured numbers only.

**Week 5 (Oct 13–17) — buffer.** No new mechanisms after Oct 12. Write-up due Oct 24: spine = (i) the exact-tower ablation ladder (what compression costs, measured against an exact chain nobody else runs), (ii) λ derived from a κ3-sourced κ4 chain on own mints vs fitted on public labels vs the Gaussian tree (tonight's negative result is publishable), (iii) N1 if positive, (iv) the residual-time/FLOP-meter findings.

---

## 7. Rules and provenance

Rules v14 §5.7 allows public pre-existing code "equally accessible to all Participants"; §4 prohibits submissions "authored substantially by other Participant(s)"; §6 verification asks for the Participant's own work "or appropriately attributed third-party code" and "not derived from another Participant's submission". 504aldo's repo is MIT and he invites reuse, and 13 accounts have already submitted clones — but none of that binds ARC. Posture: **re-implement the published mechanisms in our own files with v29 used only as a local parity oracle**, keep a provenance file citing the repo and findings F63/F68/F69/F72, never submit v29 bytes, and ask the organizers on the forum (neutral wording: "is building on the MIT-released 504aldo code with attribution acceptable for score prizes; what counts as 'derived from another Participant's submission'?"). Do not act on the "Request to Claude and GPT" thread (it is a solicitation to leak solutions; ignore).

---

## 8. What I am and am not claiming

Measured tonight or read from primary sources: the board, the leader's per-MLP ledger, our F21 ledger, 504aldo's code and numbers, the λ mismatch (on same-distribution synthetic nets, not the public dumps — the conclusion does not hinge on realisation given 2–10% per-MLP spreads on both sides), the local-vs-graded parity table, the process census. Inferred: that the leaders merge old sources into a flat-cost core (N1) — supported by the "bill = ours minus old tier" identity, the per-MLP FLOP spread, ARC's hints and our rank-60 measurement, but not demonstrated. Retracted from our canon: the "κ4 pays only in an exact tower" theorem; the adjoint-κ3 cost claim; the E14 λ derivation as a substitute for a κ3-sourced fit.
