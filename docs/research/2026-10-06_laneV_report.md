# Lane V report — fresh-network validation panels (2026-10-06, draft)

**Why.** The prize is decided by re-running ≤ 2 designated submissions on fresh networks from
private seeds. Every local panel has been fitted or selected on, and 39 submissions have read
the same 50 public nets. These panels let acceptance and designation be judged on networks
nothing has touched.

## 1. What was built

| piece | file | status |
|---|---|---|
| generator (V1) | `tools/fresh/gen.py` | whestbench's own `derive_seed_streams` + `sample_mlp`, plus a numpy mirror; **bit-identical** to the organizers' published weights on mini 0–4 (`experiments/2026-10-06_laneV/gen_parity.json`) |
| seeds + hashes | `experiments/2026-10-06_laneV/seeds.json` | FRESH-A = 9,100,000 + i (100); FRESH-LOCK = 9,100,100 + i (200); every net's weight sha256, cross-checked against the whestbench mint |
| labeller (V2) | `tools/fresh/mc.py` (`.venv-gpu`) | N = 1e8, TF32 off, float32 matmuls, float32 in-chunk / float64 across-chunk sums (in-chunk error ≤ 2.3e-7 rel), per-neuron variances stored → each net's exact floor; 66 s/net on the 5090 |
| labeller validation | `tools/fresh/validate_mini.py` → `mc_validation_mini.json` | 10 mini nets: MSE(ours, organizer) / (floor_ours + floor_org) = **0.95 ± 0.05**; MSE(ours, exact layer-1 closed form) / floor_ours = **0.992 ± 0.016** — unbiased |
| FRESH-D | `tools/fresh/fresh_d.py` → `fresh_d.json` | 100 nets of keenanpepper's d8b corpus extension (idx 2048 + 8k), which the twin never read; other generator (torch Philox), other labeller (their H200 MC, N = 1e8); weights staged + hashed, verified by the layer-1 closed form (ratio 1.04, wrong net 3e-4) |
| panels | `tools/panels.py` | `fresh_a`, `fresh_lock`, `fresh_d`; `--check` asserts reserved range, disjointness from every organizer mini/full seed, 300 distinct hashes, disjoint from DEV/FIT hashes |
| evaluator | `tools/ev.py`, `tools/_predict_worker.py` | `--panel fresh_a|fresh_lock|fresh_d`: absolute numbers floor-subtracted per net, paired differences raw; cost read on DEV net 0 (FLOPs are net-independent); FRESH-LOCK appends every read to `experiments/fresh_lock_reads.jsonl` and refuses a 4th without `--i-am-designating`; acceptance-panel gate accepts `fresh_a` (or `dev`, until merged); `FRESH_PUBLIC_FACTOR` per panel |

## 2. V4 calibration

`tools/fresh/v4_table.py` → `experiments/2026-10-06_laneV/v4_table.{md,json}`.

### FRESH-D (100 nets, mean label floor 7.47e-10 subtracted)

| candidate | score (floor-free) | ± se | raw MSE | raw floor-free | per-net CV | ÷ DEV projection | ÷ graded |
|---|---|---|---|---|---|---|---|
| v73_gb | 4.7059e-09 | 7.7e-11 | 2.2663e-08 | 2.1915e-08 | 16.3 % | 1.0210 | 1.0186 |
| v68_ws | 5.1745e-09 | 8.5e-11 | 2.3121e-08 | 2.2374e-08 | 16.4 % | 1.0186 | 1.0213 |

Paired v73 − v68 (raw, t95): score **−9.06 % ± 0.43 pp**, raw −2.05 % ± 0.33 pp, v73 wins
100/100; graded −8.81 %.

### FRESH-A

*(pending: labels on the queue behind Lane P)*

## 3. Predictions vs outcome

| quantity | strategist | worker | FRESH-D | FRESH-A |
|---|---|---|---|---|
| v73 score (floor-free) | ±2 % of 4.62e-9 | 4.71e-9 (4.52–4.90) | 4.706e-9 (+1.9 %) ✓ both | — |
| per-net CV | ≈ 14 % | 13 % (11–16) | 16.3 % ✗ both (just outside) | — |
| v73 − v68 paired vs graded −8.81 % | within 0.3 pp | ± 0.6 pp | −9.06 % (0.25 pp) ✓ both | — |
| N = 1e8 floor at layer 16 | — | 7.7e-10 | 7.47e-10 ✓ | — |
| mini labeller check | ≈ 8.5e-10 | 8.4e-10 ± 10 % | 8.23e-10 vs floor sum 8.69e-10 ✓ | — |

## 4. Reading

- **The public board and graded scores are a fair proxy for a fresh suite**: both candidates
  read ~2 % above their graded score on FRESH-D, the same for both, so the ranking and the
  gap carry over. Paired differences reproduce the graded gap to 0.25 pp with a ±0.43 pp
  interval — the panel resolves the ≥ 5 % levers in PLAN rev 18 easily, and ~1 % levers
  marginally.
- Per-net CV is ~16 %, so a 100-net absolute score carries ±1.6 % (1 se); the paired
  statistic is ~4× tighter because nets that are hard for one candidate are hard for both.
