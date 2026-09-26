# Step 1 (HANDOFF 09-27) — harmonic old tier: predictions before any DEV run (2026-09-26 ~19:45 UTC)

Arms on v68_wino (`tools/build_harm.py`): the lineage that ages out is reduced to its radial part
Sym(u ⊗ C) and u joins one carried vector U; the shared rank-256 basis is off. H<age><metric>:
age = exact lineages kept (5 = v68's exact set); metric c = cov_out, d = diag(var_out).
Algebra verified before running (`tools/harm_check.py`: ARC constant 1.5e-8 = ARC's f32 coefficient;
factored trace vs dense 1e-15; consumer slices vs dense transported radial tensor 7e-15).
Note: the handoff's written D21 consumer is the transpose of the file's convention
(`D21[a,i] = T_aai = (2 U_a C_ai + U_i C_aa)/3`); the build uses the verified form.

**Strategist (HANDOFF 09-27):** H5c within +10% raw of v68 at −20% FLOPs [35%]; H5d worse than H5c
by ≥ 2× (in excess error); the age-6 arm with the free tier beats v68 on score.

**Mine:**
- FLOPs: H5c −18 to −24% vs v68 (old tier ~25% of FLOPs, minus ~1.7 units/layer for inv + the
  (n, 2n) product at each fold); H5d 1–2 pp cheaper than H5c.
- Raw (DEV 100, twin, gains 1, unrefit): H5c **+40% to +150%** vs v68 — the radial part is one
  vector per layer standing in for a rank-256 subspace, and every earlier measurement of the old
  tier (D5 age 4→5 −11% raw; fold-age A3U3 +32%; T1 drop-old-(2,1) ×8.85) says its content is
  large and not isotropic. P(H5c ≤ +10%) = 10%.
- H5d vs H5c: excess error within ±30% of H5c's (at width 1024 the off-diagonal metric is
  O(1/√n) per entry); NOT ≥ 2× worse. 60% confidence.
- H4c / H3c: excess raw roughly additive in the extra folded lineage: H4c ≈ H5c + (A4U3-like
  +8%… +25%), H3c ≈ H4c + 30–60%.
- Versus the reference point T1 (drop the old (2,1) at age > 4, ×8.85): every H arm is far better
  than ×8.85 (the radial part recovers most of the old tier's MEAN effect): H5c < ×2.5 at 80%.

## Addendum (2026-09-26 ~20:40 UTC) — metric C is ill-posed; three well-posed metrics added BEFORE any accuracy read

The first real-meter run (H3c, DEV net 0) died with `LinAlgError: Singular matrix`. Probe of `cov_out`
at every fold on DEV net 0: condition 1.3e6 at layer 3, 3.7e8 at 5, 2e12 at 8; from layer 9 on the
chain's own covariance is INDEFINITE (λ_min −0.26 … −0.28, one neuron variance −4.6e-6), and the H3c
chain diverged (min diag −399 at layer 13, 490 zero rows at 14). The harmonic projection in the
metric C weights directions by C⁻¹, i.e. by the inverse of near-dead neurons' variance (1e-11), so
u is dominated by directions that carry no value. This is inherent to square random W (σ_min → 0),
not a bug. H5c/H4c/H3c are kept only as the record of that failure.

General rule now used (`tools/build_harm.py`): u makes T − Sym(u⊗C) traceless under a weight K,
u = 3(τI + 2CK)⁻¹(T:K), τ = tr(CK), applied to first order. Arms:
- **H5d**: K = diag(1/var) (the handoff's cheap arm).
- **H5r**: K = (C + mean(var)·I)⁻¹ (ridge; one inverse per fold layer).
- **H5e**: K = C — the Euclidean least-squares fit of Sym(u⊗C) to T; no inverse.

Predictions (mine, before any read): all three finite on DEV. Raw vs v68: H5e best of the three
(it minimises the plain tensor error, which is closest to what the slices read), +40% to +120%;
H5d +60% to +200%; H5r between them. P(any ≤ +10%) = 10%. FLOPs −19 to −23%.
Strategist's H5c prediction is unscorable as written (the arm cannot run); scored against the
best well-posed arm instead.

## Addendum 2 (~21:00 UTC) — first read, and the zero control added before it runs
Probe, DEV net 0 only (real f32 chain, CPU): **H5e final MSE 1.662e-07 vs v68 ≈ 2.1e-08 — ×7.9**,
finite. The first-order solve is off by 2% (layer 5) … 63% (layer 13) at depth because C gains
large outliers (2λ²/τ not small); that alone cannot make ×7.9. Control **H5z**: fold = drop the aged
lineage and carry NOTHING (U ≡ 0). Mine: H5z ×8–11 on net 0 (≈ T1's ×8.85, which dropped only the
old (2,1)); if H5e is within 20% of H5z the radial part carries essentially none of the old tier's value.

## Addendum 3 (~21:45 UTC) — results, DEV net 0 (real f32 chain, CPU), and a CORRECTION
**Correction to Addendum 1:** plain v68's `cov_out` on net 0 is positive definite (λ_min 1e-3 at
layer 1 → 4e-11 at layer 6 → 1e-14 at layer 12; −2.1e-6 only at layer 13). The "indefinite from
layer 9" reading was taken inside the H3c chain and is that fold's own corruption. The ill-
conditioning (cond up to ~1e15, from near-dead neurons) is real and is why metric C is ill-posed.

| arm | final MSE, net 0 | × v68 |
|---|---:|---:|
| v68 | (rerun pending) ≈ 2.1e-08 | 1 |
| H5z (drop, U ≡ 0) | 4.085e-07 | ≈ ×19.5 |
| H5e (euc LS) | 1.662e-07 | ≈ ×7.9 |
| H5d (diag) | 1.451e-07 | ≈ ×6.9 |
| H5r (ridge) | 1.392e-07 | ≈ ×6.6 |

The radial part recovers ~70% of the dropped tier's excess error; the traceless remainder carries
the rest, and that remainder alone is ~6× v68's whole error. Handoff decision rule: > +50% ⇒ closed.
