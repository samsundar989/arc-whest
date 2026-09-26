# Strategist Packet — 2026-09-26: the plan is executed; −4.7% graded (−5.3% projected), the 10% goal is not in the plan's levers

**Self-contained. Assumes no repository access and no shared chat history.**
Answers `docs/PLAN.md` rev 13 and `docs/HANDOFF_2026-09-25.md`. Brought forward from the
scheduled Sep 28 date because every concrete lever in the plan has now been executed or closed.
Every number is measured unless marked **[I]**. Paths are given so a Skeptic can check them.

**The short version.** Steps 0–3 are done and Step 4's three arms are done. The differentiable
twin — PLAN's main bet — works and produced **six straight graded improvements in two days,
4.9756e-09 → 4.7404e-09 (−4.73% score)**; a seventh (v68, Strassen-Winograd) is grading at a
projected 4.7103e-09 (−5.3%). The slice goal was **≥ 10% (≤ 4.5e-09)**. The twin has plateaued
(last stage −0.28%, inside noise), Step 4(b)/(c) are closed in both directions, and leaf 16
stays closed. **What remains in the plan is Step 5 (harden, designate, write-up, from Oct 9).**
The decision needed: how to spend Sep 27 – Oct 8.

---

## 1. Where we stand — graded rows since the Sep 25 packet

| sub | candidate | what changed | graded score | raw MSE | F/B | worst (median) wall | day |
|---|---|---|---:|---:|---:|---|---|
| 332232 | v59_corr400 | 400-net corrector, leaf 32 | 4.9756e-09 | 2.3008e-08 | 0.21626 | 94.42 (89.84) | quiet |
| 332313 | v62_s1 | twin S1: corrector end to end | 4.9347e-09 | 2.2819e-08 | 0.21626 | 102.61 (90.27) | moderate |
| 332350 | v64_s2 | + λ/θ/feed per-layer gains | 4.9124e-09 | 2.2716e-08 | 0.21626 | 94.47 (91.11) | quiet |
| 332393 | v65_s2t | + per-term closure gains (lr 3e-4) | 4.8134e-09 | 2.2249e-08 | 0.21634 | 94.34 (89.27) | quiet |
| 332409 | v66_s2t2 | S2t trained further | 4.7854e-09 | 2.2119e-08 | 0.21634 | 101.10 (89.24) | moderate |
| **332440** | **v67_s3** | **+ multi-mode κ4_iiij** | **4.7404e-09** | **2.1903e-08** | 0.21643 | 94.44 (90.74) | quiet |
| 332364 | v64_ws (B) | v64_s2 at leaf 64 | 5.2699e-09 | 2.2715e-08 | 0.23200 | **63.55 (55.27)** | **noisy (1.150)** |
| 332478 | v68_wino | v67 + Strassen-Winograd | PENDING (proj 4.7103e-09) | ≈ v67 | 0.21473 | — | — |

All 100/100, 0 failed. Every graded row landed within 0.15% of its projection (DEV/graded
factor observations now 1.0366–1.0396; constant 1.038 stands). Designation (Samuel's call):
A = v67_s3 (implied noisy worst 112.5 s), B = v64_ws (measured 63.6 s on a noisy day); v68 and
its leaf-64 twin v68_ws (gates pass) replace them if v68 grades as projected.

## 2. What was learned (details in `RESULTS.md`)

**Anatomy 2 (Step 1)** — v59 against keenanpepper's exact N = 1e9 pre-activation cumulants,
20 FIT nets. Rolled substitution (object exact at every layer, final-layer MSE removed):
**covariance 42%, κ3 51% (D3 22%, D21 12.5%), κ4 15% (all via Q31; exact D4/D22 make it
worse)**, everything but μ 80%. Exact *var* recovers 46–48% of each layer's one-step error.
D21's error is **birth/closure, not compression** (unshared T0: 99% of v59's D21 error).
κ4_iijj's off-diagonal structure is unexplained by every family tested (R² ≤ 0.19). Label noise
is layer-dependent: 7.7e-11 at the scored layer (not 6.8e-10 — AGENTS.md corrected).

**The twin (Step 2)** — the shipped file run on a torch `flopscope` shim, so gradients flow
through exactly the arithmetic that ships; accepted at DEV raw +0.006%.

| stage | trainable | HOLDOUT-100 | DEV (paired) |
|---|---|---:|---:|
| S1 | corrector pages | −0.76% | −0.62%, 65/100 |
| S4 | σ page on var_pre | −0.10% | closed (no learnable signal) |
| S2 | + λ/θ/feed per-layer gains | −0.42% | −0.48%, 83/100 |
| S2t | + 60 per-term closure gains/layer, lr 3e-4 | −2.39% | −1.96%, 90/100 |
| S2t2 | continued | −0.63% | −0.70%, 95/100 |
| S3 | + 5-mode κ4_iiij correction | −0.88% | −0.86%, 89/100 |
| S3b | continued, lr 1.5e-4 | −0.28% | plateau |

Two lessons: the handoff's absolute lr 1e-3 is catastrophic (relative steps required), and lr
2e-3 was too coarse all along — dropping to 3e-4 is what made S2t the largest stage. The learned
κ4_iiij correction *shrinks* the slice though the truth is larger: end to end, channels
compensate each other rather than become individually accurate.

**Step 3 / Step 4 arms, all priced (real f64 meter + twin DEV):**

| arm | ΔFLOPs | Δraw | Δscore | verdict |
|---|---:|---:|---:|---|
| cov_out from corrected pK1 | 0 | −0.15% | −0.15% | killed (< 2%) |
| early-born lineages fold at age 4 / 3 / 3 (≤5) | −2.7 / −5.4 / −8.3% | +7.6 / +31.6 / +50.9% | +4.8 / +24.5 / +38.4% | killed |
| shared rank 192 / 224 | −5.8 / −1.8% | +21.0 / +6.8% | +14.0 / +4.8% | closed |
| shared rank 288 / 320 | +3.9 / +7.9% | −3.4 / −5.5% | +0.3 / +2.0% | closed |
| **Strassen-Winograd (15 adds)** | **−0.79%**, −1,947 ops | 0.00% | **−0.79%** | **shipped (v68)** |
| leaf 16 / depth 6 | dearer in our spelling (Sep 23) | — | — | stays closed |

**The old tier is valuable at exactly its current price**: every tier size or age change moves
raw by several times its FLOP change, in the direction that leaves the score flat or worse.

## 3. The gap, restated

Board (Sep 25): #1 2.00e-9, #5 2.60e-9, #10 ≈ 2.8e-9. **Us 4.74e-09** (≈ #30 [I]). Accuracy gap
≈ 1.35× (our raw 2.19e-8 vs 1.56–1.67e-8); cost gap ≈ 1.4–1.7× (0.215 vs 0.127–0.155 B). At the
0.1 floor with today's accuracy: 2.2e-9. With the leaders' accuracy at our cost: 3.4e-9.

## 4. Decisions needed

1. **Direction for Sep 27 – Oct 8.** Every lever in rev 13 is executed. Candidates, with what the
   evidence says:
   - **(a) A cheaper representation of the old tier.** It is 50% of the drift (κ3) and priced at
     exactly its value; only a *representation* change (not a size change) can move the cost gap.
     The leaders run ~40% fewer FLOPs at better accuracy (Sep 25 packet: J2W ≈ 0.127 B). No
     concrete design is in hand [I]; this is research, 1–2 weeks, uncertain.
   - **(b) More twin capacity** — per-layer term gains for the (2,1) birth path, a larger corrector
     trained end to end, a learned D3 correction (D3 owns 22% of drift and is only indirectly
     touched today). Evidence: twin stages still paid 0.3–2% each at the right lr; expected
     another 1–3% total [I], 2–4 days.
   - **(c) Stop development early and move to Step 5** (hardening, noisy-day re-runs, write-up).
     The contribution-prize write-up has strong material now (Anatomy 2, the twin method,
     the old-tier price curve).
   Worker's recommendation: **(b) for 3–4 days, then (c)**; (a) only if the strategist has a
   concrete representation to test — the worker does not.
2. **Stop rule.** PLAN: "If by Oct 3 no graded row is ≥ 10% better than v59, Steps 3–4 continue
   only as one canary/day and ≥ 60% of time goes to Step 5." At −4.7% (−5.3% projected) this will
   trigger unless (a) lands. Confirm, or re-set it against a realistic target.
3. **Team freeze Oct 2** — nothing needed from the worker; noted.

## 5. Flags

- **Machine contention.** Twin S1 ran 18 s/net/step (vs 3.4 s) while another project's direct-
  launched eval held all 16 CPU threads at normal priority; queued jobs run BelowNormal.
- **Memory margins.** The first Winograd build peaked at 7,727 MiB (8 GiB limit) because block
  temporaries lived across the recursion; fixed (6,736 MiB). A later v68_ws gate run failed the
  two-sided "settled VM did not change" check on allocator shrinkage (−404 MB), then passed on a
  re-run (+20 MB). Both recorded.
- **Label panels.** The dense exact labels exist only for `full` (FIT/HOLDOUT), not DEV.
