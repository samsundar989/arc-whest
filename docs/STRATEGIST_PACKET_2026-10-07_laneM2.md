# STRATEGIST PACKET — 2026-10-07, Lane M2 section (session 2, branch `lane-m`)

To be merged under "Lane M2" in `docs/STRATEGIST_PACKET_2026-10-07.md`. Full report:
`docs/research/2026-10-07_laneM2_report.md`.

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
