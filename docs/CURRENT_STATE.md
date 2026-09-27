# CURRENT_STATE — 2026-09-27 ~17:40 UTC

*Where we stand right now. The plan is `docs/PLAN.md` (rev 15); every candidate and result is
`RESULTS.md`; the designation evidence is `docs/DESIGNATION.md`.*

## Best graded / recommended pair (Samuel designates; agents never do)

| role | file | graded | score | wall median / worst | implied noisy worst |
|---|---|---|---:|---|---:|
| **A** | `estimator_k3_v68_wino.py` (`98e40c83`) | #332478 | **4.7039e-09** | 84.0 / 93.9 s | 104.2 s |
| **B** | `estimator_k3_v68_ws.py` (`0db9afba`) | #332544 | 5.0664e-09 | 51.9 / 55.7 s | 64.3 s |

Board Sep 27: J2W 1.761e-9, Luna 1.907e-9; us #33. Gap ≈ 2.7× (raw 1.36×, cost 1.97×).

## Done under rev 14 (all closed or measured; PLAN §2)
Harmonic radial old tier closed (×5.93 raw); refit ceilings κ3 53% / cov 41% / κ4 28%; closure
bench + regression; learned closure S5 −0.75% raw at +1.68% FLOPs, not shipped.

## Repaired today
- Panels: VAL = full 900–999 (checkpoint selection), TRAIN = full 0–899, `--panel holdout`
  refused (it scored 800 trained nets), `tools/panels.py --check` passes.
- `tools/ev.py`: DEV-100, paired baseline and the WSL memory gate are hard gates; exit 2 unless
  accepted. PLAN §5 is now the full acceptance checklist.
- Ledger: #332544 row reconstructed. UTC Sep 27: 1 of 5 canary slots used before this session.

## In flight
- **Lane A:** `estimator_k3_v70_s2t3.py` (`45f26986`) = S2t3 step-368 parameters (VAL −0.79% vs
  S3), data-only diff from v68_wino; twin DEV −0.60%; official DEV-100 + gates running.
- S2t3 (worker-q 2229, 368/736 steps, low priority, repeatedly preempted by biohub) and its
  equal-step control (worker-q 2236, queued).
- **Lane B:** all-distinct κ3 projected diagnostic (`tools/k111_diag.py`) and the factor-gate
  design/prototype (`tools/build_gate.py`, `experiments/2026-09-27_gates/`), both by subagents.
