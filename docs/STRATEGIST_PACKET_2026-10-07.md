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

## Lane M2 (session 2, branch `lane-m`; pasted from `docs/STRATEGIST_PACKET_2026-10-07_laneM2.md` @ 3532090)

**M5 (ARC-b ablation, 16 DEV nets, n = 1024): AUGMENT is not separable.** ARC's factored AUGMENT
beats v73 −22.6 % ± 4.9 (16/16). Removing any single ingredient makes it 4–300× *worse than
SIMPLE*: no heavy legs 31×, memoryless κ4 core 17×, gain-mode core 9×, no 211→111 314×, no K211
252×, no loop at all 239× (once an r = 1 κ4 core is carried, the K211 terms keep it bounded).
All strategist and worker predictions of partial losses were wrong in kind. **Symmetric dressed
legs (R1 §5.1) give back 59 % of AUGMENT's gain** (+70 % ± 5 vs AUGMENT, 16/16, compounding
with depth); shrinking the dressing is worse (s = 0.5: 9.7×). **Exact dressing exists at no extra
leg:** Sym(X ⊗ e ⊗ (X + 2H)) ≡ AUGMENT (exact on 16/16), equivalently symmetric dressing minus a
symmetric −w2 lineage with leg H, which fits v77's one-leg lineages. Subtracting it only on the
top-f hubs closes ≈ the captured spurious energy: f = .05/.15/.40 → 18/43/83 % of the gap
(Lane P: +0.7 % FLOPs per 1 % of hubs). Lane P shipped v79_dress (symmetric + refit), graded
3.7025e-9 (−6.5 % vs v77), and has the f-correction built for a twin screen.

**M6 (κ3×edge family): GO for gain-mode E21b.** One-step oracle (dense κ3 at n = 1024, 6 nets): the
family moves D21 by only 0.2–0.9 % (the handoff's 2 % size gate fails as written; R1's toy said
2.4–3.2 %) but moves the off-diagonal κ4 core by 13–25 %. End-to-end in ARC's chain (16 nets),
adding E21b's gain-mode closed form (O(n²) + one n³ product C diag(w22) C per layer, zero fitted
parameters) cuts final MSE **−11.6 % ± 1.4, 16/16** (exact E21b −10.1 %; from layer 6 only
−10.5 %; E21a dilutes it, E3 has no cheap form, ridge proxies R² ≈ 0). Price ≈ +8 % FLOPs (+4 %
from layer 6) plus an off-diagonal κ4-core transport if the host carries one (v77 does not;
Lane P's gain-mode κ4 is the natural host). Handed to Lane P with the formula.

**M7 (coherent early errors, v77 vs N = 1e9 MC, 8 nets): early layers are not the lever.** At
layers 2–4 the coherent mean-scale error (6–9e-6) is ~8× under budget and the variance level
(1.6–8.8e-5) 1.8–7× under; projected cost ≤ 1.4e-9 summed over layers 1–6 (≤ 6 % of raw, an upper
bound). No prototype. One item recorded: v77's variance level runs low by a same-sign −3e-5 per
layer (−1.8e-4 at layer 6, 75 % of budget) — a cheap per-layer gain if it keeps growing deeper.

Infrastructure: worker-q admitted Lane M's low-priority jobs ahead of Lane P's twice; Lane M now
submits through `experiments/2026-10-07_laneM2/gatekeeper.py` (only when Lane P has nothing
waiting). RunPod was not usable from this session (pod SSH refused by its permission check).
