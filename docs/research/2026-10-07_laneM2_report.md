# Lane M2 report — what makes AUGMENT better, the κ3×edge oracle, coherent early errors (draft, 2026-10-07)

Branch `lane-m`, `experiments/2026-10-07_laneM2/`. Reference chain: ARC's factored k_max = 3
propagator (`external/mlp_cumulant_propagation`, commit 93d091a), our nets (He 2/n on every layer,
16 × [W, ReLU], no biases), float32, TF32 off. Nets: the 16 organizer DEV mini nets at
n = 1024 (weights by the bit-exact mint), organizer N = 1e9 labels, floor-subtracted MSE.
v73 comparisons are paired per net against v73's cached DEV MSE. Predictions were recorded
before any run (`predictions.md`).

**Status:** M5 13/16 nets (last 3 queued); M6 and M7 queued behind Lane P's twin jobs
(`gatekeeper.py` submits Lane M work only when Lane P has nothing waiting).

## Summary for Lane P

1. **AUGMENT is not separable.** Removing any single ingredient from ARC's factored AUGMENT
   makes the chain 4–300× *worse than SIMPLE*, not partly worse than AUGMENT. The κ3↔κ4 loop
   (211→111 births and the K211 core contributions) blows up exponentially with depth if
   either leg is cut.
2. **Symmetric dressed legs (R1 §5.1) keep only ~40 % of AUGMENT's gain** in ARC's chain:
   final MSE +69 % ± 5 vs AUGMENT (13/13 nets worse), growing with depth (1.07 / 1.27 / 1.69× at
   layers 8 / 12 / 16). The spurious Sym(H ⊗ w2 e_j ⊗ H) term is not benign in a 16-layer chain.
3. **Shrinking the dressing makes it worse, not better:** s·H with s = 0.7 / 0.5 / 0.3 gives
   5.0 / 9.6 / 16× AUGMENT's MSE. Losing heavy-leg content hurts more than the s² spurious term.
4. **Exact dressing at SIMPLE leg count exists.** Sym is bilinear, so AUGMENT's light×light +
   light×heavy + heavy×light births are exactly Sym(X ⊗ 3w2 e_j ⊗ (X + 2H)), X = w1∘C — or,
   equivalently, symmetric dressing **minus** a symmetric lineage with leg H and hub weight
   −w2. Both reproduce AUGMENT to 3e-14 (n = 64) and to the last digit at n = 1024 (13 nets).
   The second form fits v77's one-leg lineage representation (Lane P: +0.7 % FLOPs per 1 % of hubs).
5. **Top-hub correction:** subtracting the spurious lineage on the top-f hubs (by w2_j‖H_j‖²)
   closes a share of the sym→AUGMENT gap ≈ the spurious energy captured; the energy is spread
   (top 5 % of hubs hold 10–16 %). f = 0.05 / 0.15 / 0.40 → 18 / 43 / 83 % of the gap closed;
   raw −7.5 / −17.8 / −34.5 % vs symmetric dressing at ≈ +3.5 / +10 / +28 % FLOPs (all pass the
   marginal rule by ~2× in ARC's chain, which has no learned gains — an upper bound for v77).

## M5 — ARC-b ablation at n = 1024 (13 of 16 DEV nets)

One ingredient changed at a time (driver `m5_ablate.py`, a flag-driven copy of ARC's
`factored_nonlin_kprop_k3`; with default flags it equals ARC exactly, `m5_calibrate.json`).
"Loses" = (arm − b)/(a − b), the fraction of AUGMENT's gain over SIMPLE given back.

| arm | final MSE | Δ vs b (paired, 95 %) | loses | vs v73 (paired) |
|---|---|---|---|---|
| a SIMPLE | 3.853e-08 | +118.8 % ± 15.5 | 100 % | +67.1 % ± 8.9 |
| **b AUGMENT** | **1.782e-08** | 0 | 0 % | **−22.9 % ± 5.6** (13/13) |
| 1 no heavy legs (SIMPLE births) | 5.441e-07 | +2950 % | 2541 % | |
| 2 dressed legs (symmetric, R1 §5.1) | 3.024e-08 | **+69.2 % ± 5.1** | **60 %** | +30.7 % ± 11.4 |
| 2 dressed, s = 0.7 / 0.5 / 0.3 | 8.9e-08 / 1.71e-07 / 2.90e-07 | +398 / +856 / +1523 % | | |
| 2c asym Sym(X ⊗ e ⊗ (X + 2H)) | 1.782e-08 | 0.0 % (exact) | 0 % | |
| 3a memoryless κ4 core | 3.151e-07 | +1692 % | 1435 % | |
| 3b gain-mode core replaces ARC's core | 1.743e-07 | +878 % | 756 % | |
| 3c memoryless + gain-mode core | 1.331e-07 | +636 % | 557 % | |
| 4 no 211→111 | 5.654e-06 | +31543 % | | |
| 5 no K211 | 4.202e-06 | +23117 % | | |

Per-layer MSE ratio to b at layers 1 / 4 / 8 / 12 / 16:

| arm | ratios |
|---|---|
| 1 no heavy legs | 1.00 / 1.23 / 4.50 / 13.3 / 30.5 |
| 2 dressed | 1.00 / 1.00 / 1.07 / 1.27 / 1.69 |
| 2 dressed s = 0.5 | 1.00 / 1.07 / 2.09 / 4.61 / 9.56 |
| 3a memoryless | 1.00 / 41.0 / 29.6 / 22.2 / 17.9 |
| 3b gain core | 1.00 / 63.7 / 27.5 / 14.9 / 9.78 |
| 4 no 211→111 | 1.00 / 1.05 / 1.56 / 12.4 / 316 |
| 5 no K211 | 1.00 / 1.75 / 8.16 / 27.7 / 232 |

Readings:
- The memoryless and gain-mode cores fail *early* (40–100× at layer 4, decaying with depth):
  ARC's κ4 core carries real information at layers 2–5, where R2 found the gain law weak
  (R² 0.16–0.45). Arms that switch the core only from layer 6 on are queued (3d/3e/3f).
- 4 and 5 grow exponentially: the κ3 → κ4 (K211) and κ4 → κ3 (211→111) legs of the loop
  each compensate the other. Cutting both (arm 45) and cutting both with SIMPLE births
  (arm 145) are queued, to show whether the loop *as a pair* carries gain.

**Predictions vs outcome.** Strategist: (1) loses 35–60 %; (2) recovers ≥ 70 % of (1)'s loss;
(3) loses 20–40 %, gain-mode recovers half; (4)/(5) < 10 % each. Worker: 30 / 85 / 20 / 40 % /
5 / 8 %. Both wrong in kind for 1, 3, 4, 5 (catastrophic, not partial). (2) recovered 98 % of
(1)'s loss in raw terms (from 2541 % to 60 %) but is still 69 % worse than b; (3c) recovered 61 %
of (3a)'s loss — the closest call — but both are > 7× b.

### Dressing-strength and top-hub correction (Lane P's questions, 7 nets)

| f (hubs corrected) | final MSE | ratio to b | sym→b gap closed | captured spurious energy, layers 2/4/8/12/16 |
|---|---|---|---|---|
| 0 (symmetric dressing) | 3.161e-08 | 1.714 | 0 % | — |
| 0.05 | 2.923e-08 | 1.585 | 18 % | .10/.10/.12/.14/.16 |
| 0.15 | 2.597e-08 | 1.409 | 43 % | .26/.28/.32/.36/.42 |
| 0.40 | 2.063e-08 | 1.122 | 83 % | .58/.65/.72/.78/.85 |
| 1.00 (= asym = b) | 1.837e-08 | 1.000 | 100 % | 1 |

## M6 — κ3 × covariance-edge family at n = 1024 (queued)

Built and validated, not yet run at full width:
- `m6_terms.py`: direct dense formulas for E3 (κ3 top slice, coefficient 3), E21a (κ4 core,
  12/(2n+8)) and E21b (κ4 core, 24/(2n+8)), each matching ARC's own unfactored term evaluation
  to ≤ 3e-16 relative; gain-mode closed forms of E21a (O(n²)) and E21b (O(n²) + one n³ product
  C diag(w22) C) exact against the dense template (2e-16).
- `m6_oracle.py`: on ARC's factored trajectory, materialises the pre-activation κ3 densely at
  the steps feeding layers 4/8/12/16 and measures E3's ΔD21/ΔD3 at pre-act L, E21a/E21b's core
  increments, and the set's ΔD21 one step later (E21 by re-running the next step with the
  core increment injected); spike-r (r = 1/4/16/64) and gain-mode substitutes through the same
  formulas; a ridge-proxy sample.
- `m6_e2e.py`: E21 / E21b exact, gain-mode and spike-16 forms injected into ARC's factored
  chain, scored on final MSE (exact factored E21 == unfactored d+E21 to 2.6e-13 at n = 64).

## M7 — coherent early errors (queued)

v77's own post-ReLU mean and covariance at layers 1–6 extracted for 8 DEV nets (`m7_state.py`,
real flopscope, `_layer` exactly as `_tower` runs it). MC labels N = 1e9 (`m7_mc.py`) for
per-neuron mean/variance and the projections on v77's m̂ and top-4 covariance directions.
Deviation from the handoff: DEV nets instead of FRESH-A/FIT (no labels needed from either;
the MC is exact).

## Infrastructure note

worker-q twice admitted Lane M's queued low-priority jobs ahead of Lane P's high-priority
exclusive jobs (stale free-VRAM readings), against the handoff's ordering. Lane M work now goes
through `gatekeeper.py`. Measured host RAM of the n = 1024 ARC jobs is ~28 GiB (first
submissions declared 8 — corrected). RunPod could not be used from this session (SSH to the pod
was refused by the session's permission check); the pod was stopped after 2 minutes.
