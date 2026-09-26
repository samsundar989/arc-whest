# CURRENT_STATE — 2026-09-25 04:35 UTC

*Where we stand right now. The plan is `docs/PLAN.md` (rev 13); the ordered queue is
`docs/HANDOFF_2026-09-25.md`; every candidate and result is `RESULTS.md`.*

## Best graded

**4.8134e-09** — `estimator_k3_v65_s2t.py`, #332393 (v59 + the twin's end-to-end corrector, λ/θ/feed
and per-term closure gains), 100/100, 0 failed, worst wall 94.3 s (quiet). Prev 4.9124e-09 (v64_s2).
Twin total vs v59: −3.3% score in four graded steps.

## The designation pair (Samuel designates; agents never do)

| role | file | graded | score | implied noisy worst |
|---|---|---|---|---:|
| **A** | `estimator_k3_v65_s2t.py` | #332393 | **4.8134e-09** | 110.7 s (policy ≤ ~115 s) |
| **B** | `estimator_k3_v64_ws.py` | #332364 | 5.2699e-09 | 63.6 s MEASURED on a noisy day (policy ≤ 95 s) |

B is v64_s2 at leaf 64, graded on a noisy-day server (worst/median 1.150). Rules v14 §7 ranks only the better of the two.

## In flight (HANDOFF 09-25)

- **Step 1 — channel-drift anatomy** (`tools/anatomy2.py`, worker-q 2132): v59 against
  keenanpepper's exact N=1e9 pre-activation labels on 20 **FIT** nets (the dense labels
  exist only for the `full` split, which is disjoint from DEV). Predictions in
  `experiments/2026-09-25_anatomy2/PREDICTIONS.md`.
- **Step 2 — differentiable twin** (`tools/twin/`): the shipped estimator source run on a
  torch `flopscope` shim, so autograd flows through the exact chain. Weights export to
  `cache/twin/` (worker-q 2133) because `.venv-gpu` has torch but no dataset reader.

## Calibration facts corrected today

- **Label noise is layer-dependent**: 6.8e-10 at layer 1 but **7.7e-11 at the scored
  layer 16** (two independent N=1e9 label sets, 20 FIT nets). The Sep 24 note that put
  6.8e-10 on the final layer was wrong by ~9×; AGENTS.md now says so.
- DEV→graded factor stays **1.038** (ninth observation 1.0366).
