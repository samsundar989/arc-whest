# Strategist packet — 2026-09-27: both rev-14 jumps measured and closed; direction needed

*Answers PLAN rev 14 / HANDOFF 09-27. Written early (the handoff scheduled Oct 1) because Steps 1–3
are done and neither goal condition was met. All numbers: `RESULTS.md` § Step 1, § Step 2,
§ Step 3.1–3.2, § Step 3.3.*

## Standing
- **A = v68_wino 4.7039e-09** (#332478, unchanged best). **B = v68_ws 5.0664e-09** (#332544, graded
  Sep 27 on the A/B slot; median wall 51.9 s, implied noisy worst 64 s). Samuel designates.
- Slice goal — (i) a graded row carrying the old tier at ≤ 2 units/layer within +10% raw, or (ii) a
  ≥ 10% graded gain from a calibrated closure: **neither met.**

## What was measured (one line each)
1. **Harmonic old tier: closed.** The metric-C projection is ill-posed (cov_out condition up to 1e15
   from near-dead neurons). Well-posed radial folds recover ~70% of the old tier's error budget but
   leave raw ×5.9 (DEV-100); dropping the tier outright is ×17.8 at −28% FLOPs. The old tier's value
   is in the traceless part of κ3. *Write-up result.*
2. **Refit ceilings** (substitute exact at every layer, then refit on FIT 80, read FIT 20):
   κ3 **53%**, cov 41%, κ4 28% (under the 30% bar → 4(b) stays closed), all-but-μ 91%.
3. **Closure bench:** on exact inputs the closure's transported variance residual equals the observed
   per-layer var drift (kill rule does not fire). The residual is a smooth nonlinear state function —
   16-hidden MLP R² 0.63–0.87 on held-out nets (ridge 0.03–0.61); the unobservable κ5/κ6 add 5–30 pp.
   The κ3 drift is NOT visible in the marginal/(2,1) closure outputs — it lives in the all-distinct
   κ3 births.
4. **Calibrated closure in the loop: ≤ 0.75% raw.** Offline init +5.1% (double counting); fine-tuned
   S5 −0.75% on HOLDOUT-100 but the per-pair net costs **+1.68% FLOPs** → score-negative, not shipped.
   Per-neuron only (free) = exactly the no-net control (−0.17% = continued training). The chain's
   trainable compensations (λ/θ/q31, term gains, μ corrector) had already absorbed the bias.

## What this says (worker's reading)
- The two biggest remaining error channels by ceiling are **κ3 (53%)** and **cov (41%)**, and neither
  is a closure-calibration problem: cov's closure bias is absorbed by existing gains, κ3's error is in
  the distinct-index births/propagation, which no marginal or pair closure can see.
- The only κ3-capacity lever ever shown to move raw is **more exact old tier** (rank 256→320: −5.5%
  raw at +7.9% FLOPs; age 4→5: −11% raw) — accuracy per FLOP sits almost exactly at the multiplier's
  price. Cost cuts that remove κ3 content lose more than they save (every fold/drop/rank-down arm).
- So both gaps to the leaders (accuracy 1.37×, cost 1.8×) point at a **different κ3 representation**
  (the leaders' bill ≈ ours minus the old tier, with better accuracy): something that carries
  distinct-index κ3 of old sources cheaply — the traceless part the radial fold loses.

## Options for Sep 28 – Oct 8 (worker's ranking)
1. **Step 4(a) layout/wall** (CPU, 1–2 days): no-concatenate block layout, earlier Winograd temp
   release, merged thin calls; refit the wall law with a glue term; price leaf 16. Score-neutral by
   itself; makes a leaf-16 A designable only if the noisy-implied worst stays ≤ 105 s. Low risk.
2. **Traceless old tier at low rank** (the handoff's "+10–50%" branch, never reached because the
   radial arms were ×6): per folded source keep a rank-r (r = 8–16) symmetric-CP / leading-column
   remainder alongside the free radial part. Priced before running. Speculative; the only idea that
   attacks the κ3 cost directly.
3. **S2t3** (TRAIN 900 + d8b 2,048, running at low priority now, ~7 h): expected −0.2 to −0.8%; a
   canary if it beats v68 under the gate. Background.
4. Stop score work early and move to the write-up (Oct 8 start could become Oct 1): the anatomy,
   ceilings, closure bench/regression and the harmonic negative are all write-up-grade mechanism
   results for the $20k contribution prize.

## Questions for the strategist / Samuel
- Q1: Continue score work (which of 1–3), or pull the write-up forward (4)?
- Q2: If 2: budget (days) and a kill line (e.g. rank-16 remainder must reach ≤ +10% raw at ≤ 5 units/layer).
- Q3: Machine contention: another project's process (arc-agi-3 sandbox, 35 GiB, outside workerq) and
  the worker-q snapshot store (106 GB, 236 snapshots) filled RAM and disk on Sep 27 and killed two of
  our jobs; the RunPod box (RTX PRO 5000, stopped, volume kept, ~$3.70 spent) is available if the
  local GPU is blocked again. Owner decisions, flagged only.
