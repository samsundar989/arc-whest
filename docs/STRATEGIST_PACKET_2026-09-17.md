# Strategist packet — 17 September 2026

> **REVIEWED AND CORRECTED, September 17:** the raw live-shape numbers stand, but the recipe comparison did not match the nonlinear closure or residual-compression boundary. Our table has 25 terms; public v17 has 62+4, including live mean terms absent here. The inference excluding the κ4/closure interaction is withdrawn. N1's latest tiny-width screen is exploratory and registration-deficient, not a live-shape closure. Stage 0 has two remaining integration defects. See `STRATEGIST_RESPONSE_2026-09-17.md` and its linked independent source/validation/Skeptic reviews; execute `CLAUDE_CODE_HANDOFF_2026-09-17.md` next.


*Self-contained: assumes no repository access and no shared chat history.
Answers the 16-September audit and its execution handoff
(`CLAUDE_CODE_HANDOFF_NEXT_SUBMISSION_2026-09-16.md`). Evidence labels:
**[M]** measured in this session with raw artifacts, **[R]** reported by others,
**[I]** inference. The September 17 source, validation and independent Skeptic
reviews linked from `STRATEGIST_RESPONSE_2026-09-17.md` govern interpretation of
these records. No new candidate has been promoted for release.*

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

Truth: two independent 2^24-sample replicas per network, per-replica noise 4.7634e-9;
noise of the scored two-replica average is 2.3817e-9.
Statistic: paired arithmetic score difference, studentised-t 95%. All caps off
(accuracy only). The reviewed primary stage-1 and stage-1b contracts preceded their
reads. Do not extend that provenance claim to the N1 tiny-width screen below.

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

## 3. Algebra, cost and exploratory compression findings

- **[M] Exact oracles.** An exact non-Gaussian reference (Gaussian scale mixture
  for κ4; Gaussian-plus-quadratic for κ3; quadrature to ~1e-13) confirms the
  estimator's κ4 model and table conventions and derives, with exact numerical
  agreement: (a) two κ4 routes into all-distinct births (route (i) alone overshoots
  53%); (b) three κ3×C routes into all-distinct births (504aldo's F69 form closes
  80%; the third is a Hadamard-with-C correction with no cheap factored form);
  (c) the complete κ4(×C) bivariate slices ((3,1) alone overshoots). **None of the
  exactness transferred to score except (a), at +3%, and (c) made score worse.**
- **[M] N1's tested merged policies are deprioritized.** The mean-path JᵀJ proxy
  was informative at n=64, but did not pay at the tested rank. The later continuous
  cycle's four networks per width are exploratory: its literal seeds were not
  registered and its contract commit followed the start of truth generation.
  Results are unfavorable at n=96, with broad paired uncertainty; they do not
  establish a width1024 compression limit. One-shot energy compression of the age≥3 tier
  needs r/n ≈ 1/8 (≈90% of deletion damage removed) to 1/4 (≈99%), flat in width
  24–96. (iii) A fully metered merged Tucker core with exact per-lineage joins
  costs **0.2035 B at r=128, age gate 3** even with the cheapest exact spellings;
  a major cost is the O(n(r+s)³) projection of each lineage's n source columns.
  r=256 is 0.5–1.0 B. Symmetry declarations that halve queries can raise
  `SymmetryError` on float32 (a failed MLP on the grader).
- **[M] Exact cost savings.** Strassen depth 5 (min block 32): −6.6%; final-layer
  variance-only readout + block-symmetric covariance: −0.7%; best feasible
  **C/B 0.2802, 17,483 ops, 6.0838 GiB Windows commit**. Depth 6: −1.8% more but
  9.7197 GiB peak and 20,519 ops — a memory warning and over the operation target.
  None of the tested spellings meets the full P4 gate. This is not a universal
  exact-transformation limit; paired accuracy parity and enforced Linux memory
  remain owed for the depth5 candidate.

## 4. The unresolved bottleneck, stated precisely

Public v29 is **1.29× better in raw MSE at 0.84× the cost** of our best chain on
the same networks. v17, before old-source basis confinement but still with
rank-16 newborn-residual compression, is already 1.22× better in raw. The gap
accumulates between layers 4 and 10. The tested feed, lambda-profile and q31
transplants do not close it. However the nonlinear programs are not identical:
v17's live mean includes D3²*w6/72 and D3*D4*w7/144 absent here, and its mixed
outputs contain many additional cumulant products. Its exact feed pieces also
sit outside the residual compressor, unlike their repeated-support equivalents
in our recipe arm. Thus the prior inference excluding the κ4 channel and
calling the terms “not transferable physics” is withdrawn. Closure content,
compression placement and their trajectory interactions remain discriminators.

## 5. The exact next discriminating question

> At the same preactivation mean, covariance, D3, D21, dG and lambda, which
> connected nonlinear terms and centered outputs differ? After matching the
> selected closure, which full-rank births agree and which contracted outputs
> change only because of rank truncation?

This requires O(n²) inputs, not a dense 1024³ tensor. Compare compact births
after one common dense transport, using tiny dense fixtures to validate the
adapter. An unmodified tiny-width v17 run disables regeneration; preserve and
record the intended diagnostic mode. Equal finite moments do not specify an
exact ReLU expectation, so an accuracy oracle needs a separately specified
realizable law. Implementation equality and physical accuracy are different
checks. The new response and handoff provide the ordered protocol.

## 6. Next direction after review

Keep the approximately 7.9e-9 bundle as provisional fallback arithmetic; no
release package exists and it does not meet Samuel's requested major jump.
Execute the independently derived closure, protected-feed and shared-factor
slice in `CLAUDE_CODE_HANDOFF_2026-09-17.md`. Internal mechanism integration is
already authorized under the recorded provenance policy; do not reopen the
withdrawn organizer question. No whole-effort write-up pivot is justified yet.
N1's tested merged-core configurations remain deprioritized, with the corrected
exploratory scope; any new representation requires its own prospective contract.

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
