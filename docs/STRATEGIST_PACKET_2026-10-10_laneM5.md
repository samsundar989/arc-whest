# Strategist packet — Lane M5, 2026-10-10 (daily paragraph + the Oct 11 κ₄ read-out)

Raw tables: `experiments/2026-10-10_m5/RESULTS_M5.md`. Hand-overs: `HANDOVER_M5.1_fold.md`,
`HANDOVER_M5.4_Hlit.md` (same folder). All numbers are twin v81d, VAL-100, refits 1 epoch from
`refit_starall2` against a same-recipe control (1.941552e-08).

## Oct 10 paragraph

M5.1 lands: the masked star-Q31 fold over the **recent** lineages is −1.23 % unrefit and **−2.58 % after
refit vs the matched control**, at +0.25 % F and +243 ops with zero extra products (interpreter);
built-file parity 4.8e-5. Folding the old tier too is worse (−0.95 %) and would cost an extra product, so
`rec` is the build. M5.4's hub move is closed: the transported hub matrix matches g₁C̃ + g₂Wᵀdiag(v)W only
at age 1 (R² 0.95–0.99); each older age misses one more transported renewal (age 2 0.5–0.8, then falling),
so the gate fails. But Hk2.5 vs Hkh2.5 shows the always-on hubs are worth only ~1.3 %, and the **literal
kink-active H costs +1.35 % after refit** (P1's +3.41 % was on v73): with −6.03 % F that is ≈ −4.8 % score
with no hub move — handed to P with the caveat that the count-table version is unmeasured. M5.2's radial
terms are physically right in the labels (D4 coefficient = physical v; Q31 ≈ ½; D22 ≈ 0.4×) but the
refit-tuned chain rejects them unrefit at every scale (D22 +6 % even at 0.1×); the first refit was
under-powered (relative steps froze the gains) and is being re-run with ×300 steps.

## The Oct 11 decision read-out (M5.3 regression)

FIT nets 0–9 fit, 10–19 test; target the true pre-activation Q31 (normalised, off-diagonal) residual over
the chain's own Q31, pairs weighted by (w3_a w1_b σ_a³σ_b)². Gaussian label-noise ceiling ≈ 1.0, so the
gaps are model error.

| | l 6–9 | l 10–14 |
|---|---:|---:|
| R² of the chain's Q31 (S0) | 0.43–0.49 | 0.36–0.39 |
| ΔR² star-Q31, all ages (free, = M5.1) | 0.027–0.031 | 0.029–0.056 |
| ΔR² radial Sym(μ⊗κ₃) (free) | 0.008–0.028 | 0.03–0.06 |
| ΔR² path-Q31 age 1 (paid) | ≤ 0.02 | ≤ 0.02 |
| ΔR² (2,2)-block (paid) | ≤ 0.02 | ≤ 0.02 |
| greedy 6-piece set | 0.09–0.14 | 0.18–0.21 |

Using the handoff's rule Δraw ≈ −0.155·ΔR²/(1−R²_S0): the free pieces' ~0.1 at depth is worth ≈ −2.5 %
unrefit, and the whole greedy set ≈ −5 % unrefit (≈ −10 to −15 % at the 3.7× refit multiplier, an upper
guide). Measured so far: the star fold delivered −2.6 % after refit; the radial and extra modes are being
re-tested. **No paid piece passes the build rule** (each ≤ 0.02, ~1–2k ops per product).

**Reading:** κ₄ through D31 can add a few percent, not the −20 % a top-10 push needs. The rest of the true
Q31 (R² ceiling 1.0 vs ≈ 0.6 reachable) is not in any O(n³) readout we tested. D4 is the surprise: the
chain's D4 has NEGATIVE R² against the truth at l 2–6 (the refit has made D4 a compensator), and a
per-layer scale/offset alone would explain 25–86 % there — but the mean-only readout is already refit
on, so this is likely compensation, not headroom.

## Late Oct 10 addendum
- Fold + H2.5 literal joint refit: 1.916160e-08, −1.31 % vs control; the fold's gain survives the column
  drop in full. Stacked ≈ −7 % score.
- Radial terms re-run with ×300 steps (gains learned to ≈ 0.5–1× physical): +0.03 % vs fold-only → killed.
- The regression's other free Q31 modes (D4n·Cn, the radial halves), zero-init with real steps: −0.07 %
  → not built. **The κ₄ D31 channel in our chain is worth the fold's ≈ 2.6 % and no more on this
  evidence; it cannot carry a top-10 push.**
