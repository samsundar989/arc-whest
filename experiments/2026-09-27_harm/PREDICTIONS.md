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
