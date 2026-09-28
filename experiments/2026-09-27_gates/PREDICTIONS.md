# G1tg — transport-gate-only arm (written before the run, 2026-09-27 ~21:30 UTC)

**Evidence going in** (`experiments/2026-09-27_k111/`, FIT 0–19, no refit): the model's all-distinct (AD)
κ3 projection is 17–25% wrong at depth (30–75× label noise); AD-only oracle substitution −33% final MSE.
Per-neuron gates on existing factors span only ~40% of the squared AD error, essentially a uniform
shrink of inherited transport, and applying that in-span part WITHOUT refit worsens final MSE +14–16%.

**Arm.** `estimator_k3_v62_gt.py` (identity transport gates on `_wick_tower`'s w1, hidden 8, bound 0.25,
features α, γ3, γ4), stage G1tg = gate leaves ONLY; every other parameter frozen at the S2t3 FINAL (step-736, = v71)
set. Because nothing else moves, the step-0 VAL reading IS the equal-training control.
TRAIN 900, 2 epochs, lr 3e-4, zero-scale 1.0, eval every 28 on VAL-100 (full 900–999), patience 3.

**Hypothesis (mine).** End-to-end fitting finds a small state-dependent pattern the no-refit scalar
could not: VAL −0.0% to −0.4%; P(≥ −0.3%) = 25%. Cost +1.6e-6 of 2^41 and +291 ops (priced, worker-q
2249), so any VAL gain ≥ 0.05% pays for itself in FLOPs; the rank-4 leak (5–30% of the gate effect,
DESIGN.md) is inside what the twin trains.

**Resources.** ~16 GiB RAM (twin measured 10.6 GiB peak), 9 GiB VRAM, 4 CPU, ~26 s/step, ≤ 3 h.

**Stop.** Early stop (patience 3) or VAL gain < 0.1% after one epoch ⇒ close B2 (transport) and record;
birth arm then not run (diagnostic: two-leg/hub gates capture ≤ 8% of the error). A gain ≥ 0.2% ⇒ bake
(build_gate on the v71 source), DEV-100 paired vs v71, §5.
