# The Skeptic's follow-up, run — and what it settled

Jobs **1308** (`n=160`, float32) and **1307** (`n=64`, float64), both queued
`--priority critical` before any correction was written. Same 24 P03 seeds, so
common random numbers across widths.

## 1. The extrapolation machinery PASSES out of sample — and the Skeptic's diagnosis of *why* the result was fragile is refuted

Predicting `λ_l(160)` from the existing `{64, 96, 128}` fit, a 1.25× step:

| fit window | median rel. error | worst | within 15% |
|---|---|---|---|
| **{64, 96, 128}** (what I used) | **3.00%** | **15.00%** | **15 / 15** |
| {96, 128} ("the clean widths") | 11.93% | 32.42% | 9 / 15 |

**Dropping `n=64` makes the extrapolation worse, not better.** The Skeptic's
hypothesis was that `n=64` contaminates the fit; the out-of-sample test says the
opposite, and the reason is plain in hindsight — two points cannot see
curvature, so `{96,128}` has no leverage and the third point is what supplies
it. The `n=64` exclusion the Skeptic argued for would have degraded the one
thing we can actually check.

## 2. The negative-variance snapping is NOT a float32 artefact — and it does not move λ

`n=64` in **float64** produces **231** snaps, the identical count to float32,
and λ agrees between the two precisions to **0.000%** (worst 2e-6 relative, at
`l=14`).

So the symptom the Skeptic used to condemn `n=64` is structural rather than
numerical — the chain really does produce negative variances at that width —
**but it does not perturb the measured quantity at all.** That is not a blanket
acquittal: both precisions could be equally wrong if the snapping truncates
real signal. It does establish that precision is not the mechanism, and that
the exclusion argument as stated does not hold.

## 3. The core rejection STANDS, on better evidence than the Skeptic had

λ(1024) under every defensible window, against 504aldo's **shipped** column
(`0.95× mean`, which is what they actually run — not the `aldo_mean` column I
originally read):

| window | median ratio | range | within 15% |
|---|---|---|---|
| {64, 96, 128} | 0.784 | [0.525, 2.042] | 5/15 |
| {96, 128, 160} | **0.385** | [0.253, 2.094] | 1/15 |
| {64, 96, 128, 160} | 0.744 | [0.563, 2.057] | 3/15 |
| {64, 128, 160} | 0.755 | [0.598, 2.055] | 4/15 |
| {96, 160} | 0.404 | [0.294, 2.093] | 1/15 |
| {128, 160} | **1.005** | [0.674, 2.077] | 6/15 |

**The median moves by a factor of 2.6 across windows, and adding `n=160` does
not stabilise it.** Validating the family at 1.25× does not license it at 8×,
and this table is the measurement of that gap.

## The verdicts, restated

| claim | status |
|---|---|
| *λ agrees with 504aldo within a factor of two at every layer* | **WITHDRAWN.** Post-hoc criterion with no discriminating power (a random permutation of their table passes 14.5/15; a constant passes 15/15), and window-dependent besides |
| *`lam_own.npz` is a usable starting table for P1* | **NO.** λ(1024) is not determined by what we have measured |
| *the pre-registered 15% prediction is FALSIFIED* | **STANDS** — no window reaches more than 6 of 15 |
| *R² ≥ 0.6 by layer 4* | **STANDS**, and strengthens: `n=160` gives 0.744 at `l=7` rising to **0.843** at `l=14`, with zero snaps |
| *the κ4 mechanism is real* | **STANDS, on the R² evidence alone.** A coefficient that explains 62–84% of the off-diagonal κ4 core's variance, at every width and rising with depth, is a real mechanism. What is not established is its **value** at 1024 |
| *the AUGMENT chain cannot run at 1024×16* | **STANDS.** Retained state alone is ≥133 GiB, 4.2× the card. The specific peaks I quoted are withdrawn — see below |

## What I got wrong, itemised

1. **The headline was invented after the gate failed.** Pre-registered: within
   15%. It failed. I reported "within a factor of two", which is a criterion I
   chose knowing the answer. That is the move Rule 1 exists to prevent.
2. **I quoted three memory peaks with no artifact.** 1.61 GiB, 12.44 GiB and
   the `n=192` OOM came from an ad-hoc shell run I never saved. The only
   memory figure with provenance is `workerq-1305.json`'s
   `peak_vram_mib: 14983.0`, and it is none of the three.
3. **"The law is confirmed by the failure that tests it" is false.** An OOM at
   `n=192` on 32 GiB is consistent with any exponent ≥ 2.33.
4. **`6,369 GiB` is spurious precision.** Withdrawn in favour of "≥ 133 GiB
   from the retained state alone, more than 4× the card" — which the Skeptic
   derived independently from the state inventory.
5. **I compared against the wrong column.** 504aldo ship `0.95× mean` with
   per-MLP adaptive scaling; I read `aldo_mean`. Every ratio above now uses the
   shipped column.
6. **I filtered the retained log.** The 1,113 removed `Snapping negative
   variance` lines were the evidence bearing on the exclusion. Unfiltered logs
   are retained from here.
7. **My exclusion rule was a NaN detector wearing a degeneracy detector's
   docstring.** Fixed — see `p2_lambda_extrapolate.py`.

## What this changes for P1

P1 does **not** get a fitted λ table from this work. What it gets is narrower
and still useful: the mechanism is confirmed independently, the functional form
`G = diag(g) + λ·C_off` explains most of the core's variance at every width we
can reach, and λ's width dependence is smooth and predictable over short steps.

The route to a usable λ at 1024 is therefore **not** a longer extrapolation. It
is either (a) an estimator of the same projection that does not need the dense
`(n,n,n)` state, or (b) treating λ as **one scalar per layer to be fitted end
to end against score** on fresh design networks, with the shape from this work
as the initialiser. (b) is cheap and is the recommended next step; it also
sidesteps the unresolved normalisation question about whether upstream's
degree-4 core is the same object as their consumed `G`.
