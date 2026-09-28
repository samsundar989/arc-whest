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

## Result G1tg (2026-09-28) and the next two runs (written before they run)

**G1tg result:** VAL 2.232939e-08 (init = exact frozen control) → **2.215950e-08 at step 84 (−0.76%)**, early
stop at 168. My prediction (0 to −0.4%, P(≥ −0.3%) = 25%) was too pessimistic. Baked → `estimator_k3_v72_gt.py`
(twin DEV −0.41% vs v71; official DEV running).

**S3c — stricter control (worker-q, below).** The frozen-init control shows the gain is not from moving old
leaves, but not that it needs the GATES rather than *any* further fitting at lr 3e-4. S3c = stage S3 (the
existing leaves, no gates), init S2t3 final, lr 3e-4, same eval cadence/patience, TRAIN 900. Prediction (mine):
≤ −0.25% VAL (S2t3 ran its cosine to 0 and plateaued); if S3c reaches ≥ −0.6%, the gate's gain is not
specific to the gate and v72's mechanism claim is withdrawn (the candidate itself stays valid on DEV).

**G1bg — birth-gate arm.** Stop rule said: run only if transport moved; it did. Same recipe, birth gates
(fac1 ends g_e, w2b middle g_m, bound 0.5), init S2t3 final. Prediction: smaller than transport, −0.1% to
−0.4% VAL (diagnostic: two-leg/hub gates span ≤ 8% of the AD error). Stop: patience 3. Cost +1.9e-6, +306 ops.
