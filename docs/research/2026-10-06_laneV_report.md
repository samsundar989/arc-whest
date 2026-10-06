# Lane V report — fresh-network validation panels (2026-10-06)

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

### FRESH-A (100 nets, mean label floor 7.48e-10 subtracted)

| candidate | score (floor-free) | ± se | raw MSE | raw floor-free | per-net CV | ÷ DEV projection | ÷ graded |
|---|---|---|---|---|---|---|---|
| v73_gb | 4.7015e-09 | 6.0e-11 | 2.2642e-08 | 2.1895e-08 | 12.8 % | 1.0201 | 1.0177 |
| v68_ws | 5.1663e-09 | 6.7e-11 | 2.3086e-08 | 2.2339e-08 | 12.9 % | 1.0170 | 1.0197 |

Paired v73 − v68 (raw, t95): score **−9.00 % ± 0.45 pp**, raw −1.99 % ± 0.40 pp, v73 wins
100/100; graded −8.81 %.

**Calibration constant.** fresh / graded = 1.0177, 1.0197 (FRESH-A), 1.0186, 1.0213 (FRESH-D);
mean **1.019** → `tools/ev.py FRESH_PUBLIC_FACTOR` (projected public = fresh score ÷ 1.019).
Fresh nets sit between the mini split (DEV ÷ 1.038) and the public 50, as expected of three
samples of one distribution.

## 3. Predictions vs outcome

| quantity | strategist | worker | FRESH-D | FRESH-A |
|---|---|---|---|---|
| v73 score (floor-free) | ±2 % of 4.62e-9 | 4.71e-9 (4.52–4.90) | 4.706e-9 (+1.9 %) ✓ both | 4.702e-9 (+1.8 %) ✓ both |
| per-net CV | ≈ 14 % | 13 % (11–16) | 16.3 % ✗ both (just outside) | 12.8 % ✓ both |
| v73 − v68 paired vs graded −8.81 % | within 0.3 pp | ± 0.6 pp | −9.06 % (0.25 pp) ✓ both | −9.00 % (0.19 pp) ✓ both |
| N = 1e8 floor at layer 16 | — | 7.7e-10 | 7.47e-10 ✓ | 7.48e-10 ✓ |
| FRESH-D vs FRESH-A | — | same within 3 % | v73 4.706 vs 4.702e-9: 0.1 % ✓ | |
| mini labeller check | ≈ 8.5e-10 | 8.4e-10 ± 10 % | 8.23e-10 vs floor sum 8.69e-10 ✓ | — |

## 4. Reading

- **The public board and graded scores are a fair proxy for a fresh suite**: both candidates
  read ~2 % above their graded score on both fresh panels, the same for both, so the ranking and the
  gap carry over. Paired differences reproduce the graded gap to 0.25 pp with a ±0.43 pp
  interval — the panel resolves the ≥ 5 % levers in PLAN rev 18 easily, and ~1 % levers
  marginally.
- Per-net CV is ~16 %, so a 100-net absolute score carries ±1.6 % (1 se); the paired
  statistic is ~4× tighter because nets that are hard for one candidate are hard for both.

## 5. Hand-off to Lane P (merge list, branch `lane-m`)

- `tools/fresh/{gen,mc,fresh_d,validate_mini,v4_table}.py` (new).
- `tools/panels.py` (fresh panels + `--check`), `tools/ev.py` (fresh panels, floor
  subtraction, FRESH-LOCK log, acceptance gate, `FRESH_PUBLIC_FACTOR = 1.019`),
  `tools/_predict_worker.py` (fresh nets rebuilt from seed, hash-checked).
- `experiments/2026-10-06_laneV/` (seeds.json, fresh_d.json — the pinned panels — plus
  evidence). Labels live untracked in `cache/fresh/` of the MAIN checkout (`labels/`,
  `fresh_d_weights/`); `gen.cache_dir()` resolves the main checkout from a worktree.
- From now on: accept candidates with `python tools/ev.py <cand> --panel fresh_a --baseline
  <incumbent>`; read FRESH-LOCK only for designation (labels: worker-q 2393 N = 1e8,
  2394 the 20-net N = 1e9 floor check, queued).
- Note `tools/twin/twin.py` on lane-m also carries M4's `nbr` arm, a `TWIN_CACHE` override
  and a fix (`cmd_train` now passes `--arm`); main's working copy has Lane P's uncommitted
  `mask` arm, so merge that file by hand.
