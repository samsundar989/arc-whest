# HANDOFF — Lane M3 (session 2, branch lane-m), 2026-10-08: is κ4 a memory, how much is fourth order worth, and is there a cheap piece of it

Worktree `../arc-whest-lane-m` (branch `lane-m`; rebase on main first — it has today's rev 20 docs once Lane P3 commits them; until then read them by absolute path in the main checkout `C:\Users\samsu\Documents\arc-whest\docs\`). No estimator edits on main, no submissions. **GPU:** run M9/M10 on the RunPod box (recipe on lane-m, commit 35ddf36; RTX PRO 5000); run M8 wherever it starts first (it needs ≈ 28 GiB host RAM at n = 1024 and gates Lane P's Oct 10 decision); local 5090 only through `gatekeeper.py` when Lane P has nothing waiting.

Read first: `docs/PLAN_rev20_2026-10-08.md`, `docs/research/2026-10-08_deep_review_strategist.md` (§3 the cumulant ladder, §6 the two hypotheses), `docs/research/2026-10-08_R3_arc_k4_inventory.md` (ARC code inventory: K=3/K=4 blocks, costs, star/path/3+2 births and readouts, sweep config), your `docs/research/2026-10-07_laneM2_report.md` (tools to reuse: `m5_ablate.py`, `m6_e2e.py`, `m6_terms.py`, `m2_variants.py`).

**Why.** Eight of the top fifteen teams sit at raw 1.46–1.52e-8 — ARC's complete third-order (unfactored AUGMENT) level — and compete on cost; marius (1.14e-8), oliver_chen (1.27e-8) and Luna (1.40e-8) are below it. ARC's conjecture MSE ≈ c_K (L/n)^K (your Oct 6 width data: K=3 slope ≈ −3.2) says the next order could be worth a lot at 1024×16 but costs ~n× more; how much it is really worth at depth 16 is unknown. Two questions decide our last week: (1) can the κ4 core stand in for old κ3 lineages — that would explain how complete-K3 teams afford 0.10–0.15 B and would let us drop most of our 28 % old tier, which is the only way the κ4-core loop pays in our chain (its own price is ≈ +30–45 % F); (2) how much is fourth order worth at n = 1024, and does a piece of it ride on the κ3 factors we already carry.

Record your prediction for each item before running it (the strategist's are below).

## M8 — Is κ4 a memory? (Oct 8, first; n = 1024, 16 organizer DEV nets, organizer N = 1e9 labels, floor-subtracted)
Chain: ARC factored AUGMENT (b) **plus E21b gain-mode** (your `m6_e2e.py` arm) = **b+E** (reference 1.583e-8); control: factored SIMPLE (a).
- **How to drop correctly.** Tag every κ3 factor column by its birth layer. Drop by building a new `FactoredTensor` from the kept columns (editing `_factors` in place leaves the cached (2,1)/(3,) slices stale). Each `from_dstensor` correction block cancels the repeated-index slices of *every* block present at its layer, old ones included — so dropping an old block while a younger correction survives leaves "−Rep ghosts". `from_dstensor` is linear: split each correction block by the tag of the block it corrects, so corrections die with their block. ("Never drop the corrections" keeps every ghost — don't use it as an arm.) **Gate: A = ∞ must reproduce b+E to float rounding (1.583e-8).**
- Keep the κ4 core, its transport, its r = 1 projection of the dense d = 4 slices, K211 and 211→111 intact. Note what they read (R3 §1.3 / factor_k3.py:470-521): K4.3 (−24/(2n+8), vector pK1 = E[φ]) reads pK_111 — every κ3 block after the nonlinear step (old blocks ×w1, this layer's births, the 211→111 blocks); K4.4 (+12/(2n+8), vector w(1,2) = E[∂(φ²)], w1 on the other legs) reads the incoming WK[3] — the old blocks; both remove the repeated part first. With old blocks dropped they read only what remains — that is part of the test.
- **Drop arms:** A ∈ {1, 2, 3, 5, 8, ∞}, for b+E and for a.
- **Compression arm (if time):** instead of dropping, fold blocks older than A into a shared rank-r basis (SVD of the stacked legs, re-based every layer, energy-ordered — our old tier's rule), A ∈ {3, 5}, r ∈ {128, 256}.
- Report final MSE and the per-layer MSE ratio to the exact chain, for b+E and a.
- **Reading:** our chain (no κ4 memory) loses ×19 when its old tier is dropped (RESULTS H5z). If b+E with A = 3 dropped stays ≤ 1.3× exact, the core carries old κ3 content ⇒ a complete-K3 chain can be cheap (the GO signal Lane P needs). If ≥ 3×, old κ3 content is irreplaceable even with the core.
- Predictions (strategist): b+E, A = 5 dropped: ×2–5 [60 %]; a, A = 5 dropped: ×10–20 [70 %] — the core reduces old κ3's value but does not remove it.

## M9 — How much is fourth order worth? (Oct 8–9; RunPod)
Two pipelines — pick one and say which:
- **(preferred) ARC's own sweep**, `scripts/kprop_by_width.py` per R3 §5.2: `mlp_kwargs = dict(num_layers=17, nonlin='relu', init_kind='manual', w_var=2.0)` (16 ReLU layers, all 2/n; read act15), `trials = 8`, `sample_k_max = 1`, `output_d_max = 1` (≤ 2, else the error loop materialises factored κ3/κ4), a new `sampK_cache_name`, **`use_avg_metric = True`** (as in your Oct 6 / M2 / M8 chains; R3's snippet sets False — change it), **`torch.no_grad()`** for factor_k4 (it uses `out=`). It builds its own nets and sample labels, so it re-measures K=3 at the same widths — that is fine; check its K=3 numbers against your Oct 6 table at 256.
- or `m2_variants.py` with our nets (reserved seeds) and our MC labels (`tools/fresh/mc.py`, N = 1e8, floor-subtracted), if the K=4 classes plug in cleanly.
- Methods × widths: K=3 AUGMENT `factor=False` (the clean baseline — includes κ3×edge), K=3 AUGMENT `factor=True`, K=4 SIMPLE `factor=True` (n ≤ 256; ~13–17 GB at 256), K=4 AUGMENT `factor=True` (n ≤ 128–160). n ∈ {64, 96, 128, 192, 256}. Compare K=4 against the **unfactored** K=3 baseline (K=4 SIMPLE ⊇ unfactored K=3 AUGMENT; against factored K=3 the κ4 effect mixes with κ3×edge).
- Report: final MSE per (method, n) with paired 95 % CIs; log-log slopes; the K4/K3 ratio per n; the extrapolation to n = 1024 from the fitted slopes, cross-checked against the measured K=3 values at 256–1024 (Oct 6 table).
- **Outcomes:** extrapolated K=4 headroom at 1024 ≥ 3× ⇒ M10 continues and the strategist writes the prototype spec; 1.5–3× ⇒ M10 only if a shared-factor arm is cheap, otherwise write-up material; < 1.5× ⇒ fourth order is not the lever at depth 16 (record it).
- Predictions (strategist): K4 SIMPLE / K3 AUG (unfactored) ≈ 0.3–0.6 at n = 256 [I]; extrapolated at 1024 ≈ 0.05–0.2 (5–20× headroom) [50 %]; the naive per-order extrapolation (~1e-10 raw) will overstate it.

## M10 — Which fourth-order pieces carry it, and is there a cheap one? (Oct 9; n ∈ {96, 128}, 8 nets each)
1. **K=4 SIMPLE ablation** (`factor_k4.py:478-506` birth loops): star only, path only, 3+2 (κ3×edge) only, star + path; plus "no κ4 transport memory" (κ4 rebuilt from the current births each layer) and "κ4→κ3 feedback through the trace part only".
2. **Shared-factor arm — the version our chain can afford.** Unfactored K=3 AUGMENT (`factor=False`, dense κ3) **plus shadow legs A = M·X, B = M carried per birth layer** (they exist only to read slices); add the **full transported** star/path slices (traceless only at birth, R3 §3.4.7) wherever the core's D4 / D31 / D22 are consumed (mean, covariance, K22 heavy leg); the path uses the birth-layer C̃. Readouts in ARC normalisation (R3 §0(b); J = B·diag(w2), w3 from the birth layer):
   - star D4[a] = 4·Σ_j w3_j A_aj³ B_aj (O(n²));
   - star D31 = 3·(A∘A∘B)·W3·Aᵀ + (A∘A∘A)·W3·Bᵀ (2 u); star D22 = 2·[(A∘B)·W3·(A∘A)ᵀ + transpose] (1 u);
   - path D4 = 12·rowsum((A∘J)∘V), V = (A∘J)·C̃ (1 u).
   Variants: D4-star only (mean channel) first; then + star D31/D22; then + path D4. If injection into ARC's dense terms is too invasive, report the D4-only arm.
3. Report each arm's share of the K=4 SIMPLE − K=3 AUGMENT (unfactored) gap and its unit price at n = 1024 in our lineage representation (5 exact recent lineages + rank-256 old tier; 1 u = one 1024³ product ≈ 0.1 % B before Strassen).
- Decision: GO to Lane P (twin prototype) if an arm captures ≥ 40 % of the gap at ≤ +10 % F in our accounting.
- Predictions (strategist): star + path carry ≥ 60 % of the K=4 gain [40 %]; the D4-star-only shared-factor arm ≥ 20 % [30 %].

## M11 — optional, if the GPU is free: anatomy of the complete-K3 floor
Anatomy 2 (`tools/anatomy2.py`, RESULTS § "Anatomy 2") on ARC's b+E chain against the exact FIT labels (keenanpepper `arc-whestbench-p2-full1000-N1e9`, cross estimate): which objects (μ, var, C_off, D3, D21, D4, D22, Q31) carry the 1.5e-8 floor. Write-up material: where the third-order ceiling comes from.

## Do not
Submit, edit estimator files on main, fit or select on FRESH panels, read FRESH-LOCK, contact organizers.

## Report
`docs/research/2026-10-0X_laneM3_report.md` with tables and both predictions per item; a dated paragraph in the strategist packet after each of M8, M9, M10; hand any GO form to Lane P with its unit price and expected gain. **M8 first — Lane P's Oct 10 decision depends on it.**
