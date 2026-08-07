# MOONSHOT.md — low-rank cumulant flow (Phase D-lite)

**Pre-registered 2026-08-06, before any measurement.** The gates, metrics, seed
blocks and decision rules below are fixed now precisely so that the outcome is
informative in either direction. If this is edited after a number is seen, the
edit is logged in `experiments/LOG.md` with its reason and the original text
stays visible in git history.

**Fenced.** This lane never touches `estimator.py`. It runs in the GPU venv only.
No result from it can reach a submission without passing G3 and a full
re-validation under `docs/VALIDATION.md` §7.

---

## Why reopen this at all

Score work is closed at the honest sampling frontier (~2.7e-7, GAME_PLAN rev 6)
and the reason is arithmetic: under `(2K−1)MN` shape-based billing the multiply
is the cost, so removing ReLUs cannot buy much. The only route to the 1e-8 regime
that physics allows is to stop sampling — i.e. depth-robust *mechanistic*
propagation. Phase D was shelved because plain cumulant propagation is known to
lose to sampling at depth ≥8 for K=4 (ARC's own paper), and because everyone who
has tried a correction has reported it dying by layer 32.

Two things changed since:

1. **Track P** measured that the closure's *own* covariance eigenvectors find the
   true fluctuation subspace — effective rank falls 25.9 → 2.8 over depth, and
   the closure basis lands within one point of data PCA. So there is a small,
   *identifiable* subspace where the interesting structure lives, and we get the
   basis for free.
2. **The §7b theory** says variance concentrates into a few low-degree modes with
   known contraction rates — which, if true, is the mechanism that makes a
   low-rank truncation principled rather than a guess.

Together these say: track higher cumulants **only in the top-r closure
eigendirections**, not in full. Full K=3 on 256 neurons is a 256³ object; rank-8
is 8³. That is the difference between hopeless and cheap.

Thread 18097's contribution is the other half: per-layer corrections must be
fitted against the chain's *own* rolled-forward trajectories, not against true
cumulants, because successive layers' errors anticorrelate and partly cancel.
We adopt that, in the low-rank basis.

---

## The method under test

Propagate the mean and covariance exactly as `estimator.py` does today (K=2
Gaussian closure). Additionally carry third and fourth cumulants restricted to
the top-r eigendirections of the closure covariance, r ∈ {4, 8, 16}:

* at each layer, form the closure covariance Σ_l and its top-r eigenbasis U_l
  (already computed — this is the alignment basis machinery);
* carry `k3` (r×r×r) and `k4` (r×r×r×r) in that basis;
* push them through the ReLU with Wick/Hermite coefficients (adapt
  `factor_k3.py` / `factor_k4.py` from
  `github.com/alignment-research-center/mlp_cumulant_propagation`, whose
  `kprop_harmonic.py` alternates harmonic and diagonal-slice decompositions);
* re-express in the next layer's basis (an r×r rotation, cheap);
* apply a **per-layer renormalization fitted offline on minted networks against
  the chain's own trajectories** (18097's DAgger-style calibration), shipped as
  literals if it ever ships at all.

Baseline for comparison is the **calibrated K=2 closure**, not raw closure — the
correction has to beat the thing we already have, not a strawman.

---

## Pre-registered gates

### G1 — can it win where closure is easy?

**Setting:** depth 8, width 256, fresh mints, seed block **D08**, 300 networks,
ground truth 2²⁵ samples/MLP (noise ~1.6e-9, reported with every number).

**Metric:** final-layer MSE of the low-rank cumulant flow vs the calibrated K=2
closure, on the same networks. Paired; report the ratio and the standard error of
the paired log-ratio.

**Threshold:** the flow must beat calibrated K=2 closure by **≥3×** at its best r.

**Rule:** if it cannot win at depth 8 — where the closure is nearly right and the
correction has the least work to do — it will not win at 32. **Fail ⇒ KILL.**

### G2 — does the depth divergence persist?

**Setting:** depth 32, width 256, fresh mints, seed block **D09**, 300 networks,
same truth precision. Run r ∈ {4, 8, 16}.

**Metric:** the *bias* component of the final-layer error — the part that does
not shrink when the offline renormalization is refitted — as a function of r.

**Threshold:** bias must **decay or plateau** as r grows.

**Rule:** if bias *grows* with r, the correction is amplifying what it is meant
to fix, and the divergence that killed K=3/K=4 for ARC, for 18063, for 18085 and
for 18097 is still there. **Fail ⇒ KILL, and write up why**, quantified by the
measured noise-attenuation / bias-amplification exchange rate (18097 measured
~16:1 end-to-end; ours would be an independent number in the right basis).

### G3 — is it affordable?

**Only if G1 and G2 both pass.**

**Setting:** honest flopscope pricing of the whole flow — every eigendecomposition,
every r×r×r contraction, every basis change — at the rates in RESEARCH_NOTES §3d.
No "assume a symmetric-tensor kernel" discounts; flopscope bills by shapes.

**Metric:** `V·c` against the vault, compared to v2.4.1's. One vault touch,
reason `estimator-change`, logged.

**Threshold:** must beat v2.4.1's vault median.

---

## Budget and kill-by

* **≤2 GPU-days per week.** The vault, the constants audit and Phase-2 operations
  outrank this lane for GPU time, always.
* **Kill-by-default 2026-09-01.** If G1 has not passed by then, the lane closes
  and the write-up gets the negative result. No extensions without a new
  mechanism, stated in advance.

---

## Both outcomes ship

This is why the lane is worth opening at a strict budget:

* **Pass** — a depth-robust mechanistic estimator, which is precisely ARC's
  stated wishlist from the townhall (low-rank cumulants + resummation), and the
  strongest possible contribution-prize entry.
* **Fail** — the cleanest characterisation yet of why the obvious fix fails: a
  measured bias-amplification exchange rate, in the basis that Track P proved is
  the *right* basis, with an oracle bound. Every prior negative result on this
  (ARC's own, 18063, 18085, 18097) was measured in the full space or with generic
  regressors. Ours would be the first that cannot be dismissed as "you used the
  wrong subspace."

The failure write-up is a real contribution and is planned for, not a
consolation. That is the point of pre-registering the gates.

---

## Implementation status — 2026-08-06

The first fenced adapter is in `local_sim/lowrank_cumulant_flow.py`. It wraps
ARC's factored K3/K4 nonlinear propagation, leaves K1/K2 full-dimensional, and
after every ReLU projects K3 and K4 onto the top-r eigenspace of the propagated
K2 covariance. K4 remains in ARC's paired-factor representation, so the adapter
never materializes an n^4 tensor. Layerwise K3/K4 renormalization scalars are
explicit inputs; fitting them against D08 trajectories is deliberately deferred
to the registered calibration run.

The adapted upstream revision is pinned for this work at
`93d091a4c26c042bfffa28f2e76a81bc0aba94bb`. Upstream currently declares Python
>=3.12, while the challenge GPU environment is Python 3.10, so the research
checkout needs only syntax-compatibility edits (`type` aliases and
`typing.Self`) before use; the exact patch is checked in as
`local_sim/mlp_kprop_py310.patch` (apply with `git apply --unidiff-zero`). This
dependency is never installed into or imported by the participant process.

Smoke verification on CUDA passed for both K=3 and K=4 on a three-layer toy
challenge-shaped network. The standalone rank-4 projection check at width 16
had maximum relative out-of-basis norm **1.81e-15**. This is an implementation
invariant only, **not G1**. No D08 labels have been generated and none of G1–G3
has been evaluated.

One semantic item remains explicit for the first gate: ARC's K1/K2 states are
full-dimensional but are not bit-for-bit the calibrated closure trajectory in
`estimator.py`. Before D08 scoring, the runner must retain the estimator closure
as a separate basis/control path and report both that registered baseline and
the ARC-flow prediction. Until that is done, the adapter is infrastructure, not
a candidate estimator.

---

## G0 — done 2026-08-06. `local_sim/moonshot_g0.py`, seed block **D15**.

G0 is the prerequisite work this section had been deferring. It decides nothing,
spends no gate, and **leaves D08/D09 untouched** so G1 and G2 still have unseen
data. **No cost is claimed**: the flow remains an accuracy oracle until G3.

**Layer-convention parity — VERIFIED.** The adapter's `W.mT` was an assertion;
it is now a measurement. Propagating the closure under both readings against
per-layer MC truth on 24 canonical mints, the row convention (`z = Wᵀμ`, what
`h @ W` means for column vectors) beats the column convention by **7,222× at
layer 1** and 125,585× by layer 2 — the divergence begins at the first matmul,
which is a transpose signature, not an accumulating modelling error. A mis-wired
G1 is therefore ruled out in advance as an explanation for a negative result.
Layer 1's own residual is **97.3% MC truth noise**, so the harness is calibrated
where a known closed form exists.

**The registered baseline** (this is the number G1 and G2 must beat by ≥3×):

| depth | raw closure | **calibrated K2 — the baseline** | GT noise | ≥3× target |
|---|---|---|---|---|
| **8** — G1 | 3.3872e-05 | **1.2835e-05** | 4.10e-08 (0.32%) | **≤4.278e-06** |
| **32** — G2 | 6.2759e-05 | **2.4428e-05** | — | ≤8.143e-06 |

Fixed on D15 *before any D08 network was drawn*. It had to be built rather than
imported: v2.4.1's low-budget analytic path returns **raw** closure and the
calibration table survives only in `experiments/calibration.json`, so the
"calibrated K=2 closure, not a strawman" baseline this document requires had no
implementation anywhere until now.

**The flow now runs here.** With Samuel's approval the pinned revision
`93d091a4` was cloned to `C:\Users\samsu\Documents\mlp_cumulant_propagation` and
`local_sim/mlp_kprop_py310.patch` applied cleanly. Set
`MLP_KPROP_SRC=C:\Users\samsu\Documents\mlp_cumulant_propagation\src`.

**Parity is now measured on ARC's own code, not inferred.** At `k_max=2` the
projection and renormalizers are inert, so the flow *is* ARC's Gaussian closure;
against ours the max relative per-layer MSE difference is **8.373e-15**.

**A defect was found and fixed on the way.** At `k_max=3` upstream still carries a
degree-4 slot as an `HTensor`, and the adapter's dense fallback tried to build a
256⁴ tensor — 34 GB — and OOM'd the device. `project_higher_cumulants` now takes
`k_max` and refuses over-large dense projections with a diagnostic. `k_max=3` runs
in 0.3 s/MLP.

### The G0 oracle reading — and why §"The method under test" needs revising

Depth 8, width 256, 8 MLPs, `k_max=3`, **unrenormalized**, against the registered
calibrated-K2 baseline of **1.2067e-05**:

| rank | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|
| **vs calibrated K2** | 0.427× | 0.466× | 0.545× | 0.764× | 1.300× | **2.996×** | **12.786×** |

* **The mechanism is real**: at full rank the K3 flow beats the calibrated closure
  **12.8×** at depth 8. First positive evidence this lane has produced.
* **The pre-registered rank range is wrong by an order of magnitude.** At
  r ∈ {4, 8, 16} the flow is *worse than the control*. G1's ≥3× gate needs
  **r ≈ 128 = width/2** — precisely where **Track P** found that a rank-k
  factorisation stops beating a dense matmul under shape-based billing. The
  compression hypothesis and the cost model break even at the same r.

**This is not a G1 result.** These runs use unfitted renormalization scalars;
the method specifies per-layer scalars fitted offline against the chain's own
trajectories (18097), for which **D10** is already allocated. Fitting them is the
next step and is the only thing that could lift the low-rank arms.

**Standing: D08 and D09 are untouched, G1 has not been run**, and the r ∈ {4,8,16}
grid in "The method under test" should not be spent as written until the
renormalization is fitted and a rank that clears the gate is known to exist inside
a cost model that still pays.
