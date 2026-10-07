# Lane M2 report — what makes AUGMENT better, the κ3×edge oracle, coherent early errors (draft, 2026-10-07)

Branch `lane-m`, `experiments/2026-10-07_laneM2/`. Reference chain: ARC's factored k_max = 3
propagator (`external/mlp_cumulant_propagation`, commit 93d091a), our nets (He 2/n on every layer,
16 × [W, ReLU], no biases), float32, TF32 off. Nets: the 16 organizer DEV mini nets at
n = 1024 (weights by the bit-exact mint), organizer N = 1e9 labels, floor-subtracted MSE.
v73 comparisons are paired per net against v73's cached DEV MSE. Predictions were recorded
before any run (`predictions.md`).

**Status:** M5 complete (16/16 nets); M6 complete (GO, gain-mode E21b); M7 complete (early layers are not the lever) behind Lane P's twin jobs
(`gatekeeper.py` submits Lane M work only when Lane P has nothing waiting).

## Summary for Lane P

1. **AUGMENT is not separable.** Removing any single ingredient from ARC's factored AUGMENT
   makes the chain 4–300× *worse than SIMPLE*, not partly worse than AUGMENT. The κ3↔κ4 loop
   (211→111 births and the K211 core contributions) blows up exponentially with depth if
   either leg is cut.
2. **Symmetric dressed legs (R1 §5.1) keep only ~40 % of AUGMENT's gain** in ARC's chain:
   final MSE +70 % ± 5 vs AUGMENT (16/16 nets worse), growing with depth (1.07 / 1.28 / 1.70× at
   layers 8 / 12 / 16). The spurious Sym(H ⊗ w2 e_j ⊗ H) term is not benign in a 16-layer chain.
   Transfer (Lane P, graded): v79_dress = v77 + symmetric dressing + refit, #334576 at 3.7025e-9
   (raw 2.058e-8, −6.5 % vs v77). The learned gains absorb part of the spurious term; the ARC
   number (asym −41 % raw vs sym, no gains) bounds what the exact correction can still add.
3. **Shrinking the dressing makes it worse, not better:** s·H with s = 0.7 / 0.5 / 0.3 gives
   5.1 / 9.7 / 16× AUGMENT's MSE. Losing heavy-leg content hurts more than the s² spurious term.
4. **Exact dressing at SIMPLE leg count exists.** Sym is bilinear, so AUGMENT's light×light +
   light×heavy + heavy×light births are exactly Sym(X ⊗ 3w2 e_j ⊗ (X + 2H)), X = w1∘C — or,
   equivalently, symmetric dressing **minus** a symmetric lineage with leg H and hub weight
   −w2. Both reproduce AUGMENT to 3e-14 (n = 64) and to the last digit at n = 1024 (16 nets).
   The second form fits v77's one-leg lineage representation (Lane P: +0.7 % FLOPs per 1 % of hubs).
5. **Top-hub correction:** subtracting the spurious lineage on the top-f hubs (by w2_j‖H_j‖²)
   closes a share of the sym→AUGMENT gap ≈ the spurious energy captured; the energy is spread
   (top 5 % of hubs hold 10–16 %). f = 0.05 / 0.15 / 0.40 → 18 / 43 / 83 % of the gap closed;
   raw −7.5 / −17.8 / −34.5 % vs symmetric dressing at ≈ +3.5 / +10 / +28 % FLOPs (all pass the
   marginal rule by ~2× in ARC's chain, which has no learned gains — an upper bound for v77).

## M5 — ARC-b ablation at n = 1024 (16 DEV nets)

One ingredient changed at a time (driver `m5_ablate.py`, a flag-driven copy of ARC's
`factored_nonlin_kprop_k3`; with default flags it equals ARC exactly, `m5_calibrate.json`).
"Loses" = (arm − b)/(a − b), the fraction of AUGMENT's gain over SIMPLE given back.

| arm | final MSE | Δ vs b (paired, 95 %) | loses | vs v73 (paired) |
|---|---|---|---|---|
| a SIMPLE | 3.930e-08 | +121.2 % ± 13.6 | 100 % | +70.0 % ± 10.0 |
| **b AUGMENT** | **1.794e-08** | 0 | 0 % | **−22.6 % ± 4.9** (16/16) |
| 1 no heavy legs (SIMPLE births) | 5.536e-07 | +2983 % | 2508 % | |
| 2 dressed legs (symmetric, R1 §5.1) | 3.062e-08 | **+70.2 % ± 4.9** | **59 %** | +32.1 % ± 10.8 |
| 2 dressed, s = 0.7 / 0.5 / 0.3 | 9.09e-08 / 1.74e-07 / 2.95e-07 | +405 / +869 / +1543 % | | |
| 2c asym Sym(X ⊗ e ⊗ (X + 2H)) | 1.794e-08 | 0.0 % (exact) | 0 % | |
| 3a memoryless κ4 core | 3.062e-07 | +1629 % | 1350 % | |
| 3b gain-mode core replaces ARC's core | 1.687e-07 | +843 % | 706 % | |
| 3c memoryless + gain-mode core | 1.307e-07 | +620 % | 528 % | |
| 4 no 211→111 | 5.653e-06 | +31346 % | | |
| 5 no K211 | 4.597e-06 | +25113 % | | |

Per-layer MSE ratio to b at layers 1 / 4 / 8 / 12 / 16:

| arm | ratios |
|---|---|
| 1 no heavy legs | 1.00 / 1.23 / 4.48 / 13.4 / 30.8 |
| 2 dressed | 1.00 / 1.00 / 1.07 / 1.28 / 1.70 |
| 2 dressed s = 0.5 | 1.00 / 1.07 / 2.08 / 4.67 / 9.69 |
| 3a memoryless | 1.00 / 40.4 / 29.3 / 21.4 / 17.3 |
| 3b gain core | 1.00 / 63.7 / 27.5 / 14.6 / 9.43 |
| 4 no 211→111 | 1.00 / 1.05 / 1.55 / 12.5 / 314 |
| 5 no K211 | 1.00 / 1.73 / 8.12 / 28.9 / 252 |

Follow-up arms (16 nets):

| arm | final MSE | Δ vs b | per-layer ratio to b, layers 4/8/12/16 |
|---|---|---|---|
| 45 no κ3↔κ4 loop (no 211→111, no K211) | 4.37e-06 | +23829 % | 1.79 / 8.51 / 28.9 / 239 |
| 145 SIMPLE births + no loop | 4.29e-06 | +23430 % | 1.84 / 8.20 / 27.3 / 235 |
| 3d gain-mode core replaces ARC's, layers ≥ 6 | 5.98e-08 | +230 % | 1.00 / 3.88 / 4.06 / 3.30 |
| 3e memoryless + gain-mode, layers ≥ 6 | 3.92e-08 | +119 % (= SIMPLE) | 1.00 / 1.67 / 1.91 / 2.19 |
| 3f memoryless, layers ≥ 6 | 3.25e-07 | +1725 % | 1.00 / 18.8 / 20.9 / 18.3 |

Readings:
- The κ3↔κ4 loop is not a cancelling pair: cutting both legs blows up like cutting K211 alone.
  Once the chain carries an r = 1 κ4 core, the K211 contributions (κ3 → κ4 core) are what keep it
  bounded; SIMPLE escapes only because its κ4 is a scalar (r = 2). Any port of an off-diagonal κ4
  core channel must bring the K211 terms with it.
- The memoryless and gain-mode cores fail *early* (40–100× at layer 4 when applied from layer 1).
  Applied only from layer 6, the gain template is far better than a memoryless core (3.3× vs
  18× at layer 16; memoryless + gain ≈ SIMPLE), but in ARC's chain it cannot replace the
  transported core: the transported core carries information the zero-parameter gain law does
  not (consistent with R2's R² 0.55–0.80 for κ4 slices at depth).

**Predictions vs outcome.** Strategist: (1) loses 35–60 %; (2) recovers ≥ 70 % of (1)'s loss;
(3) loses 20–40 %, gain-mode recovers half; (4)/(5) < 10 % each. Worker: 30 / 85 / 20 / 40 % /
5 / 8 %. Both wrong in kind for 1, 3, 4, 5 (catastrophic, not partial). (2) recovered 98 % of
(1)'s loss in raw terms (from 2508 % to 59 %) but is still 70 % worse than b; (3c) recovered 61 %
of (3a)'s loss — the closest call — but both are > 7× b.

### Dressing-strength and top-hub correction (Lane P's questions, 7 nets)

| f (hubs corrected) | final MSE | ratio to b | sym→b gap closed | captured spurious energy, layers 2/4/8/12/16 |
|---|---|---|---|---|
| 0 (symmetric dressing) | 3.161e-08 | 1.714 | 0 % | — |
| 0.05 | 2.923e-08 | 1.585 | 18 % | .10/.10/.12/.14/.16 |
| 0.15 | 2.597e-08 | 1.409 | 43 % | .26/.28/.32/.36/.42 |
| 0.40 | 2.063e-08 | 1.122 | 83 % | .58/.65/.72/.78/.85 |
| 1.00 (= asym = b) | 1.837e-08 | 1.000 | 100 % | 1 |

## M6 — κ3 × covariance-edge family at n = 1024: **GO for gain-mode E21b**

Tools (all exact against ARC): `m6_terms.py` — dense E3 (coefficient 3), E21a (κ4 core,
12/(2n+8)) and E21b (24/(2n+8)), each ≤ 3e-16 from ARC's own unfactored term evaluation;
gain-mode closed forms of E21a (O(n²)) and E21b (O(n²) + one n³ product), 2e-16 from the dense
template. `m6_oracle.py` — one-step oracle on ARC's factored trajectory with the
pre-activation κ3 materialised densely (R = 109 568 factor columns at layer 16, 2.3 s).
`m6_e2e.py` — E21 forms injected into ARC's factored chain (exact factored E21 == unfactored
d+E21 to 2.6e-13 at n = 64), scored on final MSE.

### One-step sizes (6 DEV nets)

| layer L | E3 ΔD21/D21 (pre L) | E3 ΔD3/D3 | E21a Δcore/core_off | E21b Δcore/core_off | set ΔD21/D21 (post L) | c² |
|---|---|---|---|---|---|---|
| 4 | 0.009 | 0.006 | 0.054 | 0.131 | 0.002 | 0.0047 |
| 8 | 0.007 | 0.004 | 0.142 | 0.217 | 0.004 | 0.0061 |
| 12 | 0.006 | 0.004 | 0.195 | 0.254 | 0.006 | 0.0070 |
| 16 | 0.006 | 0.004 | 0.209 | 0.247 | 0.006 | 0.0077 |

(sd over nets ≤ 0.02 everywhere.) The family barely touches D21/D3 (0.2–0.9 %, against R1's toy
estimate 2.4–3.2 %) but moves the off-diagonal κ4 core by 13–25 %.

### Capture by cheap forms (6 nets; capture = 1 − ‖sub − exact‖²/‖exact‖²)

| layer L | family | spike1 | spike4 | spike16 | spike64 | gain (β = 0.7) |
|---|---|---|---|---|---|---|
| 4 | E3 ΔD21 | .02 | .08 | .27 | .65 | .03 |
| 4 | E21b core | .05 | .18 | .52 | .91 | .44 |
| 8 | E3 ΔD21 | .02 | .18 | .53 | .90 | .15 |
| 8 | E21b core | .10 | .37 | .80 | .99 | .72 |
| 12 | E3 ΔD21 | .03 | .29 | .68 | .96 | .24 |
| 12 | E21b core | .14 | .52 | .91 | 1.00 | .75 |
| 16 | E3 ΔD21 | .02 | .35 | .77 | .98 | .31 |
| 16 | E21b core | .18 | .59 | .95 | 1.00 | .77 |
| 16 | E21a core | .03 | .44 | .86 | .99 | .93 |

Full table (incl. ΔD3, E21a at every layer, the set one step later) in
`experiments/2026-10-07_laneM2/m6_table.md`. Ridge proxy (12 cheap features) held-out R²
≈ 0.00–0.05 for every target: **the learned-proxy route is dead.**

### End-to-end (16 DEV nets, organizer labels, paired vs ARC AUGMENT b)

| arm (b + κ4-core increment each layer) | final MSE | Δ vs b | wins |
|---|---|---|---|
| b | 1.794e-08 | — | — |
| E21b exact | 1.611e-08 | −10.1 % ± 2.0 | 16/16 |
| **E21b gain-mode closed form** | **1.583e-08** | **−11.6 % ± 1.4** | **16/16** |
| E21b gain-mode from layer 6 only | 1.602e-08 | −10.5 % ± 1.4 | 16/16 |
| E21b spike-16 | 1.642e-08 | −8.4 % ± 1.4 | 16/16 |
| E21 (a + b) exact | 1.730e-08 | −3.2 % ± 2.9 | 12/16 |
| E21 gain-mode | 1.676e-08 | −6.2 % ± 2.6 | 15/16 |
| E21 spike-16 | 1.696e-08 | −5.3 % ± 1.8 | 16/16 |

### Decision

The handoff's gate was (size ≥ 2 % of D21) AND (spike r ≤ 16 ≥ 70 % or gain ≥ 60 % or proxy
R² ≥ 0.6). The second half passes for E21b (gain .72–.77 at layers 8–16; spike-16 .80–.95). The
size half **fails as written** (0.6 % of D21) — but the end-to-end run shows the D21 size is the
wrong gauge for this family: E21b acts through the κ4 core's consumers (K22 heavy leg,
211→111, K211), not D21, and removing it costs 10 % of final MSE. I call **GO for the gain-mode
E21b form** on the end-to-end evidence and hand it to Lane P (done, 2026-10-07):

    X_jk = 2βc² [ Cm_jk(v_k − w22_j μ_j Cm_jk) + μ_j(q_k − w22_j Cm_jk²) + μ_k P_kj ]
    v = Cm(w22∘μ),  q = (Cm∘Cm)ᵀw22,  P = Cm diag(w22) Cm      (pre-activation μ, C; β = 0.7;
    ΔB = 24/(2n+8) · sym[w11_j X_jk w21_k], diag 0             c² = μᵀCμ/‖μ‖⁴)

Unit price: one n³ product per layer (2.1e9 FLOPs at n = 1024; ≈ 3.2e10 over 15 layers ≈ +8 % of
v77's ~4e11 before Strassen; ≈ +4 % if applied only from layer 6, which keeps −10.5 %) plus O(n²),
**plus the transport of an off-diagonal κ4 core if the host chain carries one** (W B Wᵀ, two
n³ per layer ≈ +14 % in v73 terms, Lane M §5 of the 10-06 report). Expected gain: ≤ 10–12 % raw
(an ARC-chain number; v77 has no off-diagonal κ4-core channel, so the gain depends on the host
channel — Lane P's gain-mode κ4 (P2.2) is the natural one).
Leave E21a and E3 out: E21a dilutes the gain (−3.2 % for the pair vs −10.1 % for E21b alone), and
E3 has no cheap form (gain .03–.31) and no proxy. Untested: whether B helps when it is not
transported (ARC's core memory runs through the nonlinear step, so a clean one-hop arm is not
available in this chain).

**Predictions vs outcome.** Strategist (R1): family size 2.4–3.2 % of D21, spike r ~ 64–128
needed. Outcome: 0.6 %; spike-16 already .80–.95 for E21b (but only .53–.77 for E3).

## M7 — coherent early errors: **early layers are not the lever**

v77's own post-ReLU mean and covariance at layers 1–6 (`m7_state.py`: `estimator_k3_v77_kash._layer`
exactly as `_tower` runs it, real flopscope) on 8 DEV nets, against MC labels N = 1e9 per net
(`m7_mc.py`, 10 blocks for standard errors) of the per-neuron mean/variance and of the
projections on v77's m̂ and top-4 covariance directions. Deviation from the handoff: DEV nets
instead of FRESH-A/FIT (the MC labels are exact either way; nothing is fitted).

Relative errors (v77 − truth)/truth, mean ± 95 % over nets (MC se in brackets):

| layer | coherent mean along m̂ | var along m̂ | var along u1 | tr C (variance level) | per-layer MSE |
|---|---|---|---|---|---|
| 1 | +2e-07 ± 1e-06 (2e-06) | +9.4e-04 ± 5e-05 | −4.2e-04 | −2e-07 ± 2e-06 | ≈ 0 |
| 2 | +5.6e-06 ± 2.5e-06 | +2.7e-03 ± 3e-04 | −1.2e-04 | −1.6e-05 ± 2.5e-06 | 1.1e-10 |
| 3 | +7.7e-06 ± 2.2e-06 | −7.7e-05 ± 2.4e-04 | +2.2e-04 | −4.5e-05 ± 5e-06 | 4.7e-10 |
| 4 | +7.8e-06 ± 3.5e-06 | −2.9e-03 ± 7e-04 | −3.8e-04 | −8.7e-05 ± 1.1e-05 | 1.2e-09 |
| 5 | +7.3e-06 ± 4.2e-06 | −5.1e-03 ± 1.2e-03 | −2.0e-03 | −1.25e-04 ± 1.4e-05 | 2.2e-09 |
| 6 | +8.0e-06 ± 4.9e-06 | −6.7e-03 ± 1.2e-03 | −3.6e-03 | −1.77e-04 ± 2.8e-05 | 3.3e-09 |

Against R2 §3.2's budgets (≤ 1e-9 of final MSE). R2's transfer coefficients are per η² of a
relative error of the *whole* covariance, so the along-m̂ variance error is converted to tr C
units (× m̂ᵀCm̂/tr C = 0.35 % at layer 2 … 2 % at layer 6) before comparing:

| layer | variance-level η (tr C) | m̂ error in tr C units | budget | coherent mean η | budget | projected final MSE (cov + mean) |
|---|---|---|---|---|---|---|
| 2 | 1.6e-05 | 9.6e-06 | 1.1e-04 | 6.2e-06 | 5e-05 | 2.6e-11 |
| 3 | 4.5e-05 | 1.8e-06 | 1.4e-04 | 8.1e-06 | 6e-05 | 1.2e-10 |
| 4 | 8.8e-05 | 3.0e-05 | 1.6e-04 | 8.8e-06 | 7e-05 | 3.2e-10 |
| 5 | 1.3e-04 | 7.5e-05 | 2.0e-04 | 8.6e-06 | 7e-05 | 4.2e-10 |
| 6 | 1.8e-04 | 1.3e-04 | 2.3e-04 | 9.6e-06 | 7e-05 | 5.6e-10 |

(Budgets at layers 3, 5, 6 and the transfer coefficients there are interpolated from R2's
layers 2/4/8.) The sum over layers 1–6, ≤ 1.4e-9, is an upper bound (the errors are one
propagating error, not independent): ≤ 6 % of v77's raw 2.2e-8.

**Verdict.** At layers 2–4 every coherent part is inside its budget — the mean scale by ~8×,
the variance level by 1.8–7× — so by the handoff rule no exact layer-2 covariance / κ3 prototype
is built: **early layers are not the lever.** One systematic item worth recording: v77's
variance level runs low by a same-sign, nearly linear −3e-5 per layer (−1.8e-4 at layer 6, all 8
nets), i.e. ~75 % of the budget by layer 6. If it keeps growing past layer 6 it is a cheap
deep-layer target (a per-layer variance-level gain), not an early-layer one. At layer 1 the
variance level is exact (the arc-cosine closure), and the along-m̂ error (+9e-4) is an
off-diagonal covariance truncation worth 1e-12 of final MSE.

**Predictions vs outcome.** R2 §3.3: "if below budget, spend nothing on early layers" — below
budget. Worker (unrecorded before the run, so not scored).

## Infrastructure note

worker-q twice admitted Lane M's queued low-priority jobs ahead of Lane P's high-priority
exclusive jobs (stale free-VRAM readings), against the handoff's ordering. Lane M work now goes
through `gatekeeper.py`. Measured host RAM of the n = 1024 ARC jobs is ~28 GiB (first
submissions declared 8 — corrected). RunPod could not be used from this session (SSH to the pod
was refused by the session's permission check); the pod was stopped after 2 minutes.
