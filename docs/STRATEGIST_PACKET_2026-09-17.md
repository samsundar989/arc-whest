# Strategist packet — 17 September 2026

> **UPDATED after the independent Skeptic (A PROVISIONAL, B REJECT, C PROVISIONAL) and its three follow-ups (2026-09-17):**
> (1) the (i)/(ii) route split is exact at non-unit variance — v32_i's paired lead over v32 (t95 [+2.2e-11, +8.4e-11]) is error cancellation, not a derivation error; the v32 vs v32_i choice needs its own contract.
> (2) The N1 "0.17 B" gate was post-data and is retracted; N1 is now parked on MEASURED accuracy: continuous merged tier at n=96 loses +98% (join_b a=2 r=n/8), +86% (fixed basis a=3) — only r=n/4 (0.5–1.0 B at 1024) is within 5%.
> (3) v17's joint κ4 recipe on our base hurts (0.785× with v17 λ, 0.887× with ours; 0/24). The κ4-channel conclusions in §4 therefore STAND; §3's "rejected at n=40 and n=64" for the JᵀJ proxy is corrected: informative at n=64 (3/3), did not pay at r=n/4.


*Self-contained: assumes no repository access and no shared chat history.
Answers the 16-September audit and its execution handoff
(`CLAUDE_CODE_HANDOFF_NEXT_SUBMISSION_2026-09-16.md`). Evidence labels:
**[M]** measured in this session with raw artifacts, **[R]** reported by others,
**[I]** inference. Independent Skeptic verdict on the promotion and N1 claims:
see the end of this file when appended; until then, treat the promotion as
provisional.*

## 1. The situation in one paragraph

ARC White-Box Estimation Challenge, Phase 2: predict every neuron's post-ReLU mean
for a random He-init 1024-wide × 16-deep ReLU MLP under Gaussian input. Score per
network = final-layer MSE × max(0.1, counted FLOPs / 2^41); lower is better; mean
over networks. Our latest graded file is **v31: adjusted 8.752e-9, raw 2.907e-8,
C/B 0.3010**, 0/100 failures. Public leader ≈2.45e-9. The public, MIT-licensed
chain of team 504aldo (**v29**) scores **5.40e-9** public (raw 2.13e-8, C/B 0.2526).
The requested next release target was ≈4.0–4.4e-9. **This session did not find a
supported path to that target.** The best supported bundle is ≈7.9e-9 (planning
arithmetic). No submission was made.

## 2. What was tested, and what it did (all on the same 24 fresh DESIGN networks, "P11")

Truth: two independent 2^24-sample replicas per network, label-noise 4.76e-9.
Statistic: paired arithmetic score difference, studentised-t 95%. All caps off
(accuracy only). Every contract and prediction was committed before its data.

| # | intervention (vs its control) | raw MSE | C/B | score gain | t95 on score diff | verdict |
|---|---|---:|---:|---:|---|---|
| 1 | **v32**: add the all-distinct κ4→κ3 birth feed, routes (i)+(ii), vs v31 | 3.3249e-8 vs 3.4455e-8 | 0.3023 vs 0.3010 | **1.032×** (23/24) | [+2.36e-10, +4.04e-10] | slice gate 1.5× **FAIL**; supported improvement (secondary) [M] |
| 1d | route (i) only (504aldo's feed form) vs v31 | 3.3073e-8 | 0.3023 | 1.037× | [+2.75e-10, +4.71e-10] | diagnostic [M] |
| 2 | D1: v32 with 504aldo's λ table | 3.2469e-8 vs 3.3249e-8 | = | 1.024× (20/24) | [+9.5e-11, +3.8e-10] | λ profile is not the gap [M] |
| 3 | **v34**: add exact first-order κ4(×C) bivariate slices ((3,1) + link terms) vs v32 | 3.5945e-8 | 0.3027 | **0.924×** (0/24) | [−9.3e-10, −7.3e-10] | **FALSIFIED** [M] |
| 3d | (3,1) rows only (= 504aldo `wk431`, term-for-term) vs v32 | 3.6939e-8 | 0.3025 | 0.900× | [−1.32e-9, −9.3e-10] | [M] |
| 4 | v34 and (3,1)-only, both with 504aldo λ, vs v32 with 504aldo λ | 3.877e-8 / 4.166e-8 | = | 0.836× / 0.779× | both < 0 | interaction FALSIFIED [M] |
| R1 | public **v17** (unmodified, local oracle only) vs v32 | **2.7301e-8** | 0.4922 | 0.748× (cost) | — | raw 1.22× better [M] |
| R2 | public **v29** vs v32 | **2.5701e-8** | **0.2532** | **1.545×** (24/24) | [+3.16e-9, +3.93e-9] | reproduces public ordering [M] |
| A1–3 | v17 ablations: no (3,1) / no feed / no κ4 regeneration | 3.163e-8 / 2.984e-8 / 5.738e-8 | 0.49 | — | — | (3,1) +15.9%, feed +9.3%, regen 2.1× in THEIR chain [M] |

Per-layer trace, v17 vs v32, 3 networks [M]: layer MSEs match at L0–L3
(ratio 0.99–1.03); the gap grows L4–L10 to 1.13–1.29× and is then flat. The κ4
state (transported core dG, regenerated core g) tracks within a few percent early
and 5–20% late; formulas for transport and consumers are identical in both files.

## 3. Label-free findings that are solid

- **[M] Exact oracles.** An exact non-Gaussian reference (Gaussian scale mixture
  for κ4; Gaussian-plus-quadratic for κ3; quadrature to ~1e-13) confirms the
  estimator's κ4 model and table conventions and derives, with exact numerical
  agreement: (a) two κ4 routes into all-distinct births (route (i) alone overshoots
  53%); (b) three κ3×C routes into all-distinct births (504aldo's F69 form closes
  80%; the third is a Hadamard-with-C correction with no cheap factored form);
  (c) the complete κ4(×C) bivariate slices ((3,1) alone overshoots). **None of the
  exactness transferred to score except (a), at +3%, and (c) made score worse.**
- **[M] N1 old-history compression is parked on economics.** (i) The mean-path
  JᵀJ relevance weighting (as a valid PSD whitened objective) is no better than
  forward energy at n=40/64. (ii) One-shot energy compression of the age≥3 tier
  needs r/n ≈ 1/8 (≈90% of deletion damage removed) to 1/4 (≈99%), flat in width
  24–96. (iii) A fully metered merged Tucker core with exact per-lineage joins
  costs **0.2035 B at r=128, age gate 3** even with the cheapest exact spellings;
  the floor is the O(n(r+s)³) projection of each lineage's n source columns.
  r=256 is 0.5–1.0 B. Symmetry declarations that halve queries can raise
  `SymmetryError` on float32 (a failed MLP on the grader).
- **[M] Exact cost savings.** Strassen depth 5 (min block 32): −6.6%; final-layer
  variance-only readout + block-symmetric covariance: −0.7%; best feasible
  **C/B 0.2802, 17,483 ops, 6.5 GB Windows commit**. Depth 6: −1.8% more but
  10.4 GB peak (8 GiB cap) — infeasible. The P4 target ≤0.27 B is not reachable
  with exact transformations alone.

## 4. The unresolved bottleneck, stated precisely

Public v29 is **1.29× better in raw MSE at 0.84× the cost** of our best chain on
the same networks. v17 (their no-compression ancestor) is already 1.22× better
in raw. The gap accumulates between layers 4 and 10. It is **not** explained by:
the κ4→κ3 feed (+3% for us vs +9% for them), the λ depth profile (+2.4%), or the
(3,1) consumer — whose pK rows are term-for-term identical in both files yet worth
+15.9% in v17 and −8% (−22% with their λ) in ours. Their chain without κ4
regeneration is **much worse** (5.74e-8) than ours, and regeneration then gives
them 2.1×, versus ~1.24× for us. **[I]** Their κ4 terms appear to partly cancel
errors of their own K3 base chain (v16b), so those terms are not transferable
physics for our F21-lineage base; conversely our base is better without κ4 and
worse with it. The quantity that differs is therefore most likely in the **K3
source representation/closure** (how sources are born, compressed, and consumed),
not in the κ4 channel.

## 5. The exact next discriminating question

> On one P11 network, at IDENTICAL incoming state (mean, covariance, full κ3
> tensor, κ4 core) at a layer in 4–8, do v17's and v32's single-layer outputs
> (mean, covariance, repeated slices D3/D21 of the newborn source, regenerated g)
> agree? If they disagree, which output differs first — and which of the two is
> closer to an exact reference for that one layer?

This separates "their closure is more accurate per layer" from "their errors
cancel". It needs an adapter that loads v17's state representation (A/P legs,
thin residual, feed columns) into ours or both into a dense tensor at n=1024 —
dense κ3 at 1024 is 8.6 GB float64, so the practical form is the same experiment
at n=128–256 with the dense-tower harness this session built
(`local_sim/dense_tower.py`), after confirming the v17-vs-v32 gap reproduces at
that width (small-shape proxies have misled this project before).

Alternative, cheaper, and less informative: run v29's age-gated basis and D21
feedback on top of **our** v32 base as diagnostic arms to see whether their
cost-side machinery transfers — but the accuracy gap exists already in v17, so
it would not answer §4.

## 6. Decisions requested

1. **Release.** The supported bundle (v32 + Strassen depth 5 + two exact trims)
   projects to ≈7.9e-9 (≈10% better than graded v31). It still needs a raw-parity
   DESIGN run for depth 5, full-mini confirmation, a Linux 8-GiB enforcement run
   at the 6.5 GB footprint, and a Skeptic. Is a ~10% release worth a submission
   slot, or should it wait for a larger change? (The handoff advised against
   uploading merely to finish.)
2. **Direction.** Fund §5 (a representation-level comparison against v17 at
   identical state), or pivot to building on the published v29 method under the
   provenance policy (own re-implementation; no imported constants; the rules
   question was withdrawn by Samuel), or accept the current frontier and put the
   remaining time into the contribution write-up (the exact oracles and the
   negative transfer results in §2–3 are new, measured, and publishable).
3. **N1.** Parked. Re-open only with an inexact lineage-column compression
   contract, priced end-to-end before any truth.

## 7. Artifacts (repository-relative)

- Contracts, predictions, results: `experiments/LOG.md` (2026-09-16 entries from
  "EXECUTION CONTRACT AMENDMENT" onward).
- Stage 1: `experiments/2026-09-16_stage1_k4feed/`; generator `local_sim/s1_build_v32.py`.
- Stage 1b: `experiments/2026-09-16_stage1b_rep/`; `local_sim/s5_build_rep.py`.
- Diagnostics: `experiments/2026-09-16_diag_gap/`, `experiments/2026-09-16_diag_v17abl/`.
- P3 derivation (built, not screened; cost at rank 32 ×1.069, rank 4 ×1.011 by the
  guard): `experiments/2026-09-16_stage2_p3/`, `local_sim/s2_build_p3.py`.
- Cost: `experiments/2026-09-16_stage2_p4/`, `local_sim/s4_build_p4.py`.
- N1: `experiments/2026-09-16_stage3_n1/`.
- Stage-0 instrument repairs: `experiments/2026-09-16_stage0/` and LOG entry.
