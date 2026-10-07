# STRATEGIST PACKET — 2026-10-07 (Lane P2, session 1)

**P2.0 done.** Rev 19 docs committed unchanged (93913f5); `lane-m` merged into main (1a84c67),
FRESH-A is the acceptance panel. **v78_kashws graded #334545 4.2496e-9** (projected 4.2260e-9,
+0.56 %): new wall-safe B, −5.5 % vs v76. v77 on FRESH-A: 4.0319e-9 (strategist 4.03e-9;
projected public 3.9567e-9 vs graded 3.9596e-9). flopscope #275 exposure: none (AST scan, v77/v78/v79).
**v77 census** (real meter, reconciles to the FLOP): recent tier **58.1 %** (D21 readout 30.3,
transport 27.8), old tier 28.0, covariance 9.2, closure 1.4; `experiments/2026-10-07_p2/census_v77/`.

**P2.1 dressed legs — PASS.** Twin, VAL-100, v77-equivalent chain (v62_gb + v77's mask, G1bg params):

| arm | VAL raw | vs control | strategist | worker |
|---|---:|---:|---|---|
| refit control (S2t, lr 1e-4, 1 ep) | 2.26911e-8 | — | | |
| full dressing, unrefit | 2.23405e-8 | −1.58 % (57/100) | not a verdict | −1 to +3 % |
| {C²} / {D21,D21ᵀ} / {D22} alone, unrefit | | +0.51 / −1.12 / +1.28 % | subsets can be worse | worse than set |
| full dressing, gate on both parts, unrefit | | −1.77 % | | |
| **full dressing, refit** | **2.11466e-8** | **−6.81 %** | −4 to −12 % [60 % ≥ 4 %] | −1 to −5 % |

Strategist right, me too pessimistic. Built **v79_dress** (v77 + dressing + refit baked;
`tools/build_dress.py`, selftest rebuilds v77 byte-exact) and its block-40 B twin v79_dressws.
**FRESH-A vs v77: gain 1.0784× (t95 [+2.48e-10, +3.58e-10]), 85/100, raw −7.4 %, fresh 3.729e-9 →
projected public 3.6595e-9**; F +0.06 %, ops 24,852, residual 0.276 s. Lane M2's finding (ARC chain,
no learned gains): symmetric dressing keeps only 15–25 % of AUGMENT's heavy-leg gain; the exact
asymmetric form X⊗(X+2H) equals AUGMENT; the spurious term can be subtracted as a −w2 family
(extra birth columns in v77's hcols layout, ≈ +0.7 % FLOPs per 1 % of hubs), but its energy is
not concentrated (top 5 % of hubs = 10–16 %). Candidate follow-up after v79 ships.

**P2.2 gain-mode κ4 — in refit.** Unrefit with β 0.7 blew the chain up (×7.7): the chain's κ4
slices are 0.31 (L1) → 0.56 (L14) of the β=1 template, with correlation 0.86–0.97, so β 0.7 doubled
κ4. Refits (arms a/b/c) start from chain-matched per-layer β and learn it.
