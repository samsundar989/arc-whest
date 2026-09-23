# RESULTS — ARC White-Box Estimation Challenge, Phase 2

One row per candidate evaluation or graded run. This file replaces
`experiments/LOG.md` (archived 2026-09-21). Append; never rewrite history.
The plan is `docs/PLAN.md`; the submission gates are its §5.

## Column meaning

| column | meaning |
|---|---|
| **date** | UTC date of the evaluation or of the submission |
| **file** | candidate source in the repo root |
| **sha8** | first 8 hex of sha256 of the *estimator source bytes* (not the package) |
| **panel** | `DEV` = the 100 organizer mini nets · `FIT` = 100 full-split nets disjoint from mini · `HOLDOUT` = the remaining **900** (≤2 reads/week, never fitted) · `GRADED` = AIcrowd server, 100 nets (50 public + 50 sealed) |
| **raw MSE** | final-layer MSE, unweighted |
| **F/B** | FLOPs per MLP ÷ 2^41, i.e. the score multiplier before the 0.1 floor |
| **ops** | flopscope operation calls per MLP (drives residual wall time) |
| **wall** | worst per-MLP wall seconds. Local rows are marked `(local)` and come from the one-net grader twin; the server figure is `local × 1.5981` (§ Calibration) |
| **RSS** | VmPeak under the WSL `RLIMIT_AS` 8,192 MB gate |
| **projected** | predicted public score = DEV score ÷ 1.037 |
| **graded** | the AIcrowd adjusted score actually returned |
| **paired gain** | paired arithmetic gain vs the stated baseline, common random numbers |
| **wins/N** | nets on which the candidate beat that baseline |

## Calibration constants (recheck after every graded row)

All four are implemented in `tools/ev.py`; change them there and here together.

- **DEV → graded**: divide the 100-net DEV adjusted score by **1.042**
  (raised from 1.037 on 2026-09-22). Six observations: **1.0386 / 1.0385 /
  1.0339 / 1.0426 / 1.0471 / 1.0458**, mean 1.0411, last three averaging 1.0452.
  At 1.037 the last three graded rows all came in *better* than projected — by
  0.53%, 0.97% and 0.84% — which is a consistent bias, not noise. 1.042 sits
  under the recent mean, so projections stay slightly pessimistic and the gate
  slightly strict. The constant is deliberately left at the *low* end of that
  range: projected = DEV/1.037 comes out slightly pessimistic, which makes §5's
  "≥3% better than the best graded" gate harder rather than easier. C10 #331816
  landed 0.53% *better* than projected. If a graded row ever deviates by >2%,
  stop submitting and recalibrate before the next slot (PLAN §5).

  The factor is entirely a raw-MSE effect: C10's graded multiplier matched its
  DEV multiplier to 2e-10, while raw MSE moved 3.094966e-08 → 2.968645e-08. The
  graded 100 nets are simply a little easier than the mini 100; nothing about
  cost changes between the two.

- **FLOPs: there is no offset to add.** Take the number from the **f64 read-only
  grader twin** (`tools/_cost_worker.py`, one net) and use it as-is. Verified
  2026-09-21 against both graded rows we hold:

  | candidate | twin FLOPs/MLP | twin multiplier | graded multiplier | delta |
  |---|---:|---:|---:|---|
  | F21 #330950 | 928,457,637,478 | 0.42221365105 | 0.4222136511 | **0** |
  | v31 #331317 | 661,961,618,994 | 0.30102529263 | 0.3010252926 | **0** |
  | C10 #331816 | 480,850,933,121 | 0.21866568801 | 0.21866568778 | **491 FLOPs of 4.8e11**, i.e. 1e-9 relative |

  (A third confirmation is on record: the twin predicted #330212's `flops_used`
  to delta 0 out of sample, `docs/GOTCHAS.md` 2026-09-07.) What *does* read low
  is `whest run --split mini`, by exactly **2^25 = 33,554,432** — its local
  dataset hands out float32 weights so the candidate's opening cast never runs.
  That is the "+2^25" PLAN §4 mentions; taking the number from the twin makes
  the correction unnecessary rather than approximate. C10 measured: twin
  480,850,933,121 vs `whest run` 480,817,378,689, difference 2^25 exactly.

- **server worst-of-100 wall**: fitted on **FLOPs and op calls**, plus the
  model's own error band:

      wall = 60.251 s per 1e12 FLOPs + 2.478 s per 1000 op calls + 4.1 s

  Refit on five graded rows. The 4.1 s is the fit's own worst residual and it is
  **added** rather than ignored, because the wall cap is a cliff: a net over
  120 s is TIME_EXHAUSTED and scores as zeros.

  | candidate | FLOPs | ops | graded WORST | projected |
  |---|---:|---:|---:|---:|
  | F21 #330950 | 928.5 G | 14,091 | 93.160 s | 95.0 s |
  | v31 #331317 | 662.0 G | 13,053 | 68.132 s | 76.3 s |
  | C10 #331816 | 480.9 G | 23,421 | 87.556 s | 91.1 s |
  | v50 #331863 | 473.4 G | 25,728 | 88.747 s | 96.4 s |
  | v53 #331880 | 502.4 G | 25,140 | **96.488 s** | 96.7 s |

  Conservative on every anchor now. **Two successive corrections were needed
  here.** The original model (`twin wall × 1.5981`) was 23% low on v50. Its
  replacement, fitted on four rows without an error band, said 90.6 s for v53
  which graded 96.488 s — inside the residual band, but on the wrong side of
  PLAN D7's 95 s finals bound, so a candidate could look final-ready and not be.
  Refit again when a sixth row grades.

- **ops → server residual**: use **11.09 µs/op**, the worst of three graded
  anchors, because the cliff FAILS the MLP outright and over-projecting is free:

  | candidate | twin ops | graded residual | µs/op | headroom vs 0.4 s |
  |---|---:|---:|---:|---:|
  | F21 #330950 | 14,091 | 0.156232 s | 11.087 | 2.56× |
  | v31 #331317 | 13,053 | 0.140244 s | 10.744 | 2.85× |
  | **C10 #331816** | **23,421** | **0.236255 s** | **10.087** | **1.69×** |

  The rate is **not** constant — it falls as ops rise (11.09 → 10.09), so a flat
  µs/op is a crude model of a mix-dependent cost. It is not even monotone in
  ops between F21 and v31, so do not fit a line to three points; take the worst
  rate and keep margin. Implied ceilings at 11.09 µs/op: the 0.4 s cliff is
  36,069 ops, **1.5× headroom is 24,046 ops**, and PLAN D7's 0.33 s finals bound
  is 29,757 ops.

  The in-process local `residual_wall_time_s` is an instrument artefact (C10
  reads 5.9–9.1 s locally against 0.236 s on the server) and must never be
  quoted — project from ops.

## Rows

| date | file | sha8 | panel | raw MSE | F/B | ops | wall | RSS | projected | graded | paired gain | wins/N | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-14 | `estimator_k3_strassen_L4_f21.py` | `d77dab0e` | GRADED | 3.779e-08 | 0.42221 | 13,936 (local meter) | 93.160 s | — | — | **1.5957e-08** | — | — | sub #330950, 100/100, 0 failed. Worst residual 0.156232 s (2.56× headroom). The designated fallback and the anchor for `predict_public.py`. |
| 2026-09-16 | `estimator_k3_v31.py` | `6dde5fd4` | GRADED | 2.9074e-08 | 0.30103 | 13,037 | 68.132 s | 5,434.6 MB | 8.713e-09 | **8.7520e-09** | 1.8233× vs F21 | 50/50 | sub #331317, 100/100, 0 failed, smoke 8/8. Worst residual 0.140244 s (2.85× headroom, 10.76 µs/op). Projection landed +0.45% — the best graded score to date and the §5 baseline. |
| 2026-09-18 | `estimator_k3_v31.py` | `6dde5fd4` | DEV | 3.006065e-08 | 0.30101 | 13,037 | 52.02 s (local) | 5,434.6 MB | 8.713e-09 | see above | baseline | — | The DEV control. `experiments/2026-09-16_slice7_canary/`. Reproduce this row in D1: DEV adjusted 9.0486e-09 ± 1%. |
| 2026-09-18 | `estimator_k3_v43_complete_c10.py` | `fd8d497d` | DEV | 3.094966e-08 | 0.21865 | 23,405 | 60.259 s (local) → 78.9 s proj | 5,754.8 MiB | **6.526e-09** | PENDING | 1.3371× vs v31 DEV (score ratio, unpaired) | — | C10. DEV adjusted 6.767156e-09, 100/100, 0 failed, official `v2-phase2` mini, subprocess runner (`experiments/2026-09-18_c10_diagnostic/c10_mini.json`). Raw MSE is 3.0% WORSE than v31; the whole gain is cost (0.30103 → 0.21865 F/B). All §5 gates pass — see below. |
| 2026-09-21 | `estimator_k3_v43_complete_c10.py` | `fd8d497d` | DEV | 3.094966e-08 | 0.21867 | 23,421 | 49.881 s (local) → 79.7 s proj | 3842 MiB | **6.5262e-09** | PENDING | — | — | D1 acceptance on the instrument of record: raw MSE reproduces the Sep-18 official whest run EXACTLY (3.094966e-08). Multiplier is the f64 twin's, hence 2^25 above whest run's. |
| 2026-09-22 | `estimator_k3_v43_complete_c10.py` | `fd8d497d` | GRADED | 2.968645e-08 | 0.21867 | 23,421 | — | — | 6.526e-09 | **6.4914e-09** | 1.3482× vs v31 (graded/graded) | — | **sub #331816, THE NEW BEST. 100/100, 0 failed.** Server residual **0.236255 s** of the 0.4 s cliff (1.69× headroom, 10.09 µs/op) — the first reading at 23k ops, and the reason D0 existed. Graded landed 0.53% BETTER than the 6.5262e-09 projection. Server multiplier matched the f64 twin to 2e-10. best net 5.029e-09, worst 8.43e-09. Report: `experiments/run_reports/c10_canary_331816_graded_report.json`. |
| 2026-09-22 | `estimator_k3_v50_safe.py` | `f9cb9439` | DEV | 3.011684e-08 | 0.21529 | 25,728 | 45.065 s (local) → 72.0 s proj | 5,784.9 MiB | **6.2524e-09** | READY, NOT SENT | raw −2.691% / cost −1.544% vs C10 | — | **D2 fitted coefficients + D3 item-1 Strassen routing.** All seven §5 gates pass, 0 failed nets, 3.7% better than the 6.4914e-09 graded. HOLDOUT (100 of 900) agrees: +2.871% raw against DEV's +2.687%, 0.18 pp apart. Package `experiments/2026-09-22_v50_canary/v50-canary.tar.gz`. |
| 2026-09-22 | `estimator_k3_v53_age5.py` | `8257c055` | GRADED | 2.434000e-08 | 0.22848 | 25,140 | **96.488 s** | 6,457.1 MiB | 5.607e-09 | **5.5603e-09** | 1.1136× vs v50 (graded/graded) | — | **sub #331880, THE NEW BEST. 100/100, 0 failed.** D5's result: old-tier compression age 4→5. Graded 0.84% better than projected — the third row running to beat its projection. Residual 0.268490 s (1.49× headroom, 10.68 µs/op). Multiplier matched the twin exactly; FLOPs delta −491. **WALL 96.488 s EXCEEDS PLAN D7's 95 s finals bound** — a good canary, not a designable entry. |
| 2026-09-23 | `estimator_k3_v50_safe.py` | `f9cb9439` | GRADED | 2.876e-08 | 0.21529 | 25,728 | **87.663 s** | — | — | **6.192e-09** | identical to #331863 | — | **sub #331939, identical bytes rerun (HANDOFF 09-22 Bank). 100/100, 0 failed.** Same score to every digit (the chain is deterministic). Worst wall 87.663 s vs 88.747 s first run (−1.2%); worst residual 0.26291 s vs 0.26550 s. Run-to-run wall spread on identical bytes ≈1.1 s — inside the wall model's ±4.1 s band. v50 stays the safe designate-B candidate. |
| 2026-09-23 | `estimator_k3_v55_p2r4.py` | `aac641b4` | DEV | 2.569027e-08 | 0.21618 | 23,645 | 54.385 s (local) → 91.3 s proj | VmPeak 6,470.5 MB (WSL gate PASS, 23/23) | **5.3300e-09** | PENDING | 1.0470× vs v53 (Δ +2.610e-10, t95 [+2.432e-10, +2.788e-10]) | 100/100 | **v55 = v53 + `_P2_RANK` 32→4** (elasticity table). Raw +0.94% vs v53, cost −5.38%, ops −1,495. All seven §5 gates pass; projected wall 91.3 s is under D7's 95 s bound (v53: 96.7 s). An earlier run of this row was INVALID — 5 MemoryError nets while the box was loaded (`v55_ev_INVALID_memoryerror.log`); rerun clean, 0 failed. Package `experiments/2026-09-23_v55_canary/v55-canary.tar.gz`. |

<!-- ev.py --append inserts new rows directly above this marker. Keep it here. -->

## The evaluator (D1, 2026-09-21)

`python tools/ev.py <candidate.py> [--panel dev|fit|holdout] [--n 100]
[--baseline <file>] [--mem] [--append]` produces every column above in one
command. `tools/panels.py` pins the panels; `tools/check_backends.py` reports
what the GPU backend costs in accuracy.

### Panels — HOLDOUT is 900 nets, not 800

PLAN §4 assumed the 100 `mini` nets were a subset of the 1000-net `full` split.
They are not: **the overlap is zero**. So DEV = 100 mini, FIT = the first 100
full nets, HOLDOUT = the remaining **900**. DEV and FIT were verified disjoint
by sha256 of the weight tensors across all 200 nets, not just by seed.

### Two accuracy backends

| backend | s/net | 100 nets | host RSS | agreement with the official 100-net mini run |
|---|---:|---:|---:|---|
| `numpy` (default) | ~50 | **38 min** (4 workers) | 3.8 GiB | **exact** — 100-net mean 3.094966e-08, the official figure to every digit it reports |
| `cupy` on the 5090 | **2.03** | **3.4 min** | 8.0 GiB | panel mean **+3.2e-05**; worst single net 1.9e-03 |

PLAN §4 asked the GPU to match numpy to 1e-6 relative. **It does not, and it
cannot.** Both run the chain in float32, the GPU reduces in a different order,
and sixteen layers amplify that to ~1e-3 on a *per-net* MSE. It is not TF32 —
on a 1024³ f32 product against a float64 reference cupy is *more* accurate than
the CPU BLAS (3.5e-7 vs 7.2e-7) — it is float32 associativity, and nothing
turns it off without changing the candidate.

But the per-net errors are signed and roughly independent, so **the panel mean
is 30× tighter than the tolerance that would matter and 1000× inside the 3%
decision gate**. Measured on C10 over all 100 DEV nets:

    cupy  100-net mean raw MSE   3.095064010e-08     adjusted 6.767843e-09
    official whest run           3.094965995e-08     adjusted 6.767156e-09
    relative                     +3.2e-05

So: **the GPU is sound for panel-level decisions and is what the D2 fit will
run on**; `ev.py` still uses numpy for anything that becomes a RESULTS.md row or
a §5 gate, because a submission is worth 40 minutes. Do not run the two
backends concurrently — 8.0 + 15.2 GiB puts this box into paging.

### Acceptance — met on both backends

C10 on DEV, 100 nets, against the Sep-18 official `whest run`
(`experiments/2026-09-18_c10_diagnostic/c10_mini.json`):

| | raw MSE | adjusted | vs official raw |
|---|---|---|---|
| official `whest run` | 3.094965995e-08 | 6.767156e-09 | — |
| **numpy (instrument of record)** | **3.094966e-08** | 6.767629e-09 | **exact** |
| cupy (screening) | 3.095064010e-08 | 6.767843e-09 | +3.2e-05 |

PLAN D1 asked for ±1%; numpy reproduces the official raw MSE to every digit the
official report carries. The small spread in *adjusted* is not accuracy — it is
the multiplier: ev.py takes FLOPs from the f64 twin, which is 2^25 above what
`whest run` bills, and that is the correct, graded-verified number.

Evidence: `experiments/2026-09-21_d1/c10_dev_numpy.json`,
`experiments/2026-09-21_d1/c10_dev100_cupy.json`.

### Operational notes that cost time

- **A pipe deadlock made the pool look unparallelisable, and it was our bug.**
  `ev.py` gave each worker `stderr=subprocess.PIPE` and drained it only at
  `communicate()`, which does not run until the process exits. The candidate
  emits Python warnings with source context, so a worker that wrote ~64 KiB
  filled the OS pipe buffer and blocked **forever**, at ~0% CPU, a few hundred
  MiB resident, having just entered `predict`. Workers now write to per-worker
  `.out`/`.err` files. The `.err` files are kept: a candidate that starts
  emitting new warnings is telling you something.

  The measurement that settled it — and that no earlier hypothesis was checked
  against — was per-process CPU time: a "slow" worker had accumulated **3 CPU
  seconds in three minutes**. Three CPU-seconds is not slow, it is stopped. A
  control confirmed it: the panel index a pooled worker had sat on for 144 s ran
  standalone in 62 s *while that worker was still stuck*.

  **Three wrong diagnoses preceded this**, each plausible, each acted on, each
  costing a cancelled run: the HF hub's rate limiter, BLAS thread
  oversubscription, and the Windows memory manager trimming working sets. They
  are recorded so the next session does not requeue them.

- **The pool is still bounded by free RAM, not cores** — ~3.8 GiB per worker
  against ~30 GiB genuinely free on a 16-logical / 8-physical-core box. Four is
  the default. That ceiling is real on its own and does not depend on the
  deadlock it was briefly confused with.
- Workers write a per-net progress sidecar, because the driver reads stdout only
  at the end and a slow worker is otherwise indistinguishable from a hung one.
- `HF_HUB_OFFLINE=1` in the worker env: otherwise every worker resolves the
  dataset revision against an unauthenticated, rate-limited endpoint.

## D3 — billing polish: measured at −2.26%, not PLAN's −6–10% (2026-09-22)

**PLAN's D3 target is not reachable from the five items it names.** Every item was
measured under the real meter. Two are infeasible, one is worth ~0.05%, and the
two that work only work *together*. The deliverable is
`estimator_k3_v46_d3.py`: **−2.26% score at flat op count**, which is real but
**below §5's 3% gate**, so it cannot ship alone.

| item | measured | verdict |
|---|---|---|
| 1. Strassen for plain-`@` products | blanket: −2.60% FLOPs, **+30.3% ops**; high-value subset: −1.55% FLOPs, +9.7% ops | only viable once item 2 funds the ops |
| 2. 15-add Winograd instead of 18-add Strassen | −0.69% FLOPs, **−8.2% ops**, +29% RSS | small alone; its op saving is what pays for item 1 |
| 3. pooled preallocated `out=` buffers | **INFEASIBLE** | see below |
| 4. remove recomputed `A_L*M` / `C_L` | ~0.2–0.3% by inspection | marginal, not pursued |
| 5. aliasing / `as_symmetric` | ~0.05% | **closed**, see below |
| **1 + 2 combined** | **−2.26% FLOPs, ops 23,421 → 23,484** | the deliverable |

### Why the op budget, not the FLOP budget, governs this work

Measured by forcing `_STRASSEN_DEPTH`:

| depth | FLOPs | multiplier | ops |
|---|---:|---:|---:|
| 0 (off) | 723,666,642,305 | 0.329085 | **8,724** |
| 3 | 529,917,341,057 | 0.240979 | 20,017 |
| 5 (C10) | 480,850,933,121 | 0.218666 | **23,421** |

**63% of C10's op budget is Strassen glue.** Strassen buys a 33.6% FLOP
reduction with 14,697 op calls. So routing *more* products through it spends the
residual budget fast: the blanket version of item 1 reached 30,528 ops = 0.339 s
projected, which busts both §5's 24,500 ceiling and PLAN D7's 0.33 s finals
bound. `tools/d3_strassen_table.py` prices the trade per shape — break-even is
~61,000 FLOPs saved per extra op call, and every site clears it, so the
constraint is the residual cliff, not efficiency.

### Item 3 is infeasible, and the reason is a hard flopscope property

- `fnp.add(a, b, out=buf)` bills **exactly** what `a + b` bills: 1 op call, same
  FLOPs. `out=` saves nothing.
- Slice assignment raises: *"flopscope arrays are immutable, so item assignment
  (arr[i] = ...) and indexed in-place updates are not supported."*
- `fnp.empty(...)` itself costs an op call.

So there is no way to write into a pooled buffer, and nothing to gain if there
were. PLAN's note that "504aldo ships `out=` and is graded" may be true, but it
buys no FLOPs and no ops here.

### Item 5 is closed: symmetry discounts elementwise ops, not matmuls

| operation | untagged | symmetric-tagged |
|---|---:|---:|
| `X @ Y` (n=256) | 33,488,896 | **33,488,896** |
| `einsum('ij,jk->ik')` | 33,488,896 | **33,488,896** |
| `X + Y` | 65,536 | **32,896** |
| `X * 2` | 65,536 | **32,896** |
| `as_symmetric(X)` itself | — | **458,751** (≈7n², it verifies) |

Matmuls — which are essentially the whole bill — get **no discount at all**.
Elementwise ops get a genuine 2×, but `as_symmetric` costs ~14 elementwise ops
to pay for itself, and n×n elementwise work is a negligible share of 481 G.

### The deliverable, on DEV (100 nets)

    C10   raw 3.095064010e-08   multiplier 0.218666   adjusted 6.767843e-09
    v46   raw 3.094959965e-08   multiplier 0.213724   adjusted 6.614663e-09
          raw change  -0.0034%      score change  -2.2634%

**Accuracy is unharmed.** Winograd perturbs individual activations by up to
4.3e-3 relative — it is algebraically exact (verified to 2.7e-15 in float64) but
numerically noisier than classical Strassen in float32 — yet the panel raw MSE
moves −0.0034% and 51/100 nets improve. That is a coin flip, i.e. pure
rounding-order noise with no systematic direction. It is **not** the ≤1e-6
bit-parity PLAN assumed the item would have, so it is validated by panel
measurement rather than by parity.

Projected public **6.378653e-09** = only **+1.74%** better than the 6.4914e-09
best graded. Below the 3% gate. `estimator_k3_v46_d3.py` is a **base to stack
D2's accuracy gain on**, not a candidate.

**The memory margin is the real price.** WSL 8 GiB gate PASSES, but VmPeak rises
**5,754.8 → 6,801.6 MiB** — 83% of the 8,192 MiB cap against C10's 70%, leaving
1,390 MiB. PLAN D7's finals bound is 7 GiB, so this clears it by 200 MiB.
Winograd is the cause: it holds eight S/T temporaries simultaneously where the
classical form folds its sums into the `concatenate` call.

### Two bases, and the choice is deferred to D2

Neither ships alone, and which one D2 should stack on depends on D2's own
memory footprint (it re-enables P3 feedback), which does not exist yet:

| base | FLOPs | score vs C10 | ops | VmPeak |
|---|---:|---:|---:|---:|
| C10 (unchanged) | 480,850,933,121 | — | 23,421 | 5,754.8 MiB |
| item 1 subset only | 473,416,794,497 | −1.55% | 25,698 | ~5,800 MiB |
| **v46 = items 1 + 2** | 469,983,381,889 | **−2.26%** | 23,484 | 6,801.6 MiB |

The middle row exceeds §5's 24,500-op ceiling, but that ceiling was set *before*
any graded residual existed and PLAN §5 says to revisit it "after the first
graded residual reading from C10". At the worst observed rate 25,698 ops is
0.285 s (1.40× headroom, inside D7's 0.33 s); at C10's own measured rate it is
0.259 s. Raising the ceiling to ~26,000 is defensible on that evidence — but it
is a gate change, so it waits for Samuel rather than being taken silently.

## D4b and K4 — both CLOSED on cost (2026-09-22)

D4 measured accuracy only. This competition scores **MSE × FLOPs**, so the same
arms were priced with ARC's own FLOP counter (`tools/d4b_price_k4.py`) and the
two sides combined. Against factored AUGMENT K3, the arm our chain is closest to:

| arm | n=128 L=8 cost× | MSE× | **score×** | n=256 L=8 cost× | MSE× | **score×** |
|---|---:|---:|---:|---:|---:|---:|
| factored SIMPLE K3 | 0.125 | 2.744 | **0.343** | 0.124 | 2.509 | **0.312** |
| unfactored AUGMENT K3 | 1.361 | 0.843 | 1.147 | 2.359 | 0.924 | 2.180 |
| **factored K4** | **21.19** | 0.294 | **6.222** | **42.15** | 0.154 | **6.483** |

**K4 is closed.** Its 3.4–6.5× accuracy costs 21–42× the FLOPs — a 6.2–6.5×
*worse* score — and the cost ratio doubles from n=128 to n=256, so it degrades
further toward our width of 1024. The accuracy lever D4 found is real and
unaffordable.

**D4b is closed**, now on cost rather than on the shrinking-gain argument alone:
restoring the dropped diagrams is score-negative at both shapes (1.15×, 2.18×
worse) and worsens with width.

**The surprise, and it points at D2.** Factored *SIMPLE* K3 beats factored
AUGMENT on score by ~3× at both shapes: the extra Edgeworth diagrams cost 8×
and buy only 2.5–2.7×. If that transfers, the frontier is not reached by richer
term sets at all — it is reached by a cheap term set with well-fitted
coefficients, which is exactly D2's bet and exactly what the leaders' 0.13 B
cost profile looks like.

*Limits, stated because the numbers are seductive:* torch's counter uses its own
conventions and counts a different program from the one flopscope bills for our
chain, so only ratios between arms under the same counter mean anything; ARC's
implementation is not optimised the way ours is (Strassen, f32); the shapes are
(128,8) and (256,8), not (1024,16); and our chain is not exactly any of these
arms. This is a screening result that closes two lanes, not a prediction of our
score.

## PLAN §5 gate change — op ceiling 24,500 → 26,000, residual now a hard gate

PLAN §5 provides for this explicitly: "ops ≤ 24,500 (tighten or loosen after the
first graded residual)". C10 #331816 supplied that reading — 23,421 ops billed
0.236255 s, 10.087 µs/op, 1.69× headroom, better than the 0.252–0.260 s
projected.

Two changes, made together:

- **Op ceiling 26,000.** At the worst of three graded rates (11.09 µs/op) that is
  0.288 s: 1.39× headroom against the cliff and inside D7's bound. At C10's own
  measured rate it is 0.262 s.
- **Projected residual is now a real gate at ≤ 0.33 s**, not advisory. It is
  gated at PLAN D7's *finals* bound rather than at the 0.4 s cliff, so no canary
  is ever built on margin a designated entry could not keep. This is the binding
  constraint; the op ceiling is the cheap proxy that catches a regression early.

The cliff FAILS the MLP, and a failed net scores ~0.9 against ~6e-9. It is the
one number where being wrong is unrecoverable, which is why the gate sits well
inside it.

## D5-H3 first cut — the old-tier RANK TAPER is worth nothing (2026-09-22)

PLAN D5-H3 proposes a final-layer-aware schedule: "taper old-tier rank and
switch off D21/old readouts over the last 2–4 layers", expecting −5–10% FLOPs.
The taper half is measured and **dead**.

`estimator_k3_v51_h3.py` adds `_SHARED_RANK_TAIL = (h, rank)`: the last `h`
layers compress the old tier at `rank` instead of `_SHARED_RANK`. With it set to
`None` the file is bit-identical to v50 in both FLOPs and ops. Swept on DEV:

| taper | multiplier | ops | raw MSE | score |
|---|---:|---:|---:|---:|
| off (v50) | 0.215285 | 25,728 | 3.011892e-08 | — |
| last 2 layers @ 128 | 0.215285 | 25,728 | 3.011560e-08 | −0.01% |
| last 3 layers @ 128 | 0.215285 | 25,728 | 3.011560e-08 | −0.01% |
| last 4 layers @ 128 | 0.215285 | 25,728 | 3.011560e-08 | −0.01% |

Not a rounding artefact — a structural fact about the implementation, and the
mechanism was **verified by probe rather than inferred**, because the first
explanation written here was imprecise.

The taper *does* fire: instrumenting `_shared_rank` over one net shows 28 calls,
two per layer at layers 1–14, of which **6 take the taper branch and return 64
instead of 256** (layers 12, 13, 14 at h=4). So the constant is reaching the
code.

It is inert for a narrower reason. `_shared_rank` is consulted only at *guard*
sites, each of the form `if _SHARED_ON and _shared_rank(n) and (tw.k - tw.ko) >=
_SHARED_AGE`, and in the tail the second condition is false: **no compression
event occurs in those layers at all.** The old-tier basis `Q` is built once,
earlier, when enough lineages have accumulated; the tail layers only transport
it. A smaller rank requested where nothing is being compressed is never used to
build anything, which is why the multiplier does not move and why h = 2/3/4 and
rank 64 vs 128 are all byte-identical.

**That points at the real lever.** To cut tail cost the existing basis has to be
*re-compressed* smaller on entry to the tail — an explicit rank reduction of `Q`
at layer `L-h`, which then makes every remaining transport cheaper — rather than
a request that silently finds no compression to attach to.

To cut tail cost the basis itself would have to be *re-compressed* smaller, or
the readouts switched off. That is the other half of H3 ("births at layers
≥ L−2 keep only the (3)-diagonal readout"), and it is real surgery on the birth
logic rather than a constant: the last layer already prunes to
`{(1,), (3,), (4,)}` and skips every term with `ip != (1,)`, but layer L−2 still
has to produce the full state that L−1 consumes, so pruning backwards trades
accuracy for cost rather than removing waste outright. Not attempted yet.

## Single-leg kill tests T0–T3 (HANDOFF 2026-09-22, started 2026-09-23 UTC)

Probe file `estimator_k3_p60_probe.py` = v53 verbatim plus knobs `_D21_AGE`
(T1), `_LBLOCK_AGE` (T2), `_RHO` (T3), and two diagnostics (`_KEEP_AGE`: drop
old lineages from both readouts; `_LB_PART`: which part of the L-block T2
drops). Runner `tools/probe.py`: every arm on every net in one persistent cupy
process, so arms are paired by construction. **Probe arms are accuracy-only and
run on cupy** (panel-mean agreement with numpy 3.2e-05); nothing here is a §5
row. Age = `l-1-b` for lineage `b` at layer `l`; the newborn is age 0.

Parity, 2 DEV nets: the probe at v53's settings reproduces v53 to every digit
(2.521407e-08 both); `_LBLOCK_AGE=99`, `_D21_AGE=99` and `_RHO=0` reproduce
their bases exactly.

**Predictions, recorded before the 100-net run.** Strategist: T1 ≤8% raw loss
at a=3; T2 10–40% at a=1 ("I don't know"). Mine, and **made after** a 2-net
smoke test that already showed T2 a=1 at +5,700% and T1 a=3 at +1,500%: T2 a=1
loses >40× raw on DEV 100; T1 a=3 >10×; no a ∈ {0..3} of either gate comes
within 40% of T0; the loss falls roughly geometrically with a (a=10 ≈ +30%).
Diagnostic prediction: the pure two-leg block (`w2b`) carries most of T2's loss,
the rank-32 `Z` residual little.

### T0–T2 on DEV 100 — the single-leg hypothesis is falsified (2026-09-23)

`experiments/2026-09-23_single_leg/t0t2_dev100.json` (per-net MSE, every arm),
cupy, 100/100 nets, all arms paired against T0 on the same nets.

| arm | what is removed for lineages older than `a` | raw MSE | × T0 | ±95% | wins/100 |
|---|---|---:|---:|---:|---:|
| **T0** | nothing: unshared exact chain (`_SHARED_ON=False`) | **2.3642e-08** | 1 | — | — |
| v53 settings | old tier compressed at age 5, rank 256 | 2.5451e-08 | 1.077 | ±0.008 | 1 |
| T1 a=2 | old lineages' (2,1) slice | 6.356e-07 | 26.9 | ±0.8 | 0 |
| T1 a=3 | ″ | 3.675e-07 | 15.5 | ±0.4 | 0 |
| T1 a=4 | ″ | 2.092e-07 | 8.85 | ±0.24 | 0 |
| T2 a=0 | L-block (`A_L`, `eb`, `Z`) → `A_D = M d(sb)` | 1.737e-06 | 73.5 | ±2.5 | 0 |
| **T2 a=1** | ″ (the decisive arm) | **1.138e-06** | **48.1** | ±1.6 | 0 |
| T2 a=2 | ″ | 7.561e-07 | 32.0 | ±0.9 | 0 |
| T2 a=3 | ″ | 5.002e-07 | 21.2 | ±0.6 | 0 |
| T2 a=6 | ″ | 1.363e-07 | 5.77 | ±0.15 | 0 |
| T2 a=10 | ″ | 3.017e-08 | 1.276 | ±0.015 | 0 |
| T1 a=3 ∩ T2 a=1 | both | 1.259e-06 | 53.3 | ±1.7 | 0 |
| diag: K a=1 | the whole lineage, from both readouts | 1.623e-06 | 68.6 | ±2.3 | 0 |
| diag: K a=5 | ″ | 3.122e-07 | 13.2 | ±0.3 | 0 |
| diag: T2 a=1, two-leg only | `w2b` (the `A_L ⊗ 3M ⊗ C_L` block) | 3.304e-07 | 14.0 | ±0.4 | 0 |
| diag: T2 a=1, `Z` only | the rank-32 born-(2,1) residual | 1.516e-07 | 6.41 | ±0.21 | 0 |

**Reading.** Every prediction the strategist wrote is wrong by one to two orders
of magnitude (T1 a=3: predicted ≤8%, measured +1,454%; T2 a=1: predicted 10–40%,
measured +4,714%). My post-smoke prediction (>40× at a=1, geometric decay, a=10
≈ +30%) held. Four facts:

1. **The old D-block alone is nearly worthless.** Keeping `M d(sb)` for old
   sources (T2 a=1, 48×) is barely better than dropping them outright (K a=1,
   69×). The single-leg object the design rests on does not carry the accuracy.
2. **Every part of the L-block matters on its own**: the pure correlation-born
   two-leg block 14×, even the rank-32 `Z` residual 6.4×. None is cheap to drop.
3. **Old (2,1) content matters as much as old (3,) content** (T1 a=3 15.5×).
   Leaving D21 at age ≤3 is not the 60%-of-cost free saving §1 of the response
   assumed.
4. **It is consistent with a number we already had.** v53's own docstring and
   `experiments/submissions.jsonl` record the fully memoryless (regenerating) κ3
   flow, #329896: graded 8.50e-08 at the 0.1 floor, i.e. **raw ≈ 8.5e-07 ≈ 36×
   the exact tower**. "Regenerate the correlation-born κ3 memoryless" was in
   effect measured in September. v53's shared basis, by contrast, costs only
   +7.7%: the old content can be *compressed* cheaply but cannot be *dropped or
   reduced to its diagonal*.

**T3 — fitted regeneration, CLOSED.** `tools/probe_fit.py`, CMA-ES on FIT, base
T2 a=1. The 2-parameter age-exponential form (50 evaluations, 25-net batch)
recovered 0.59% in-sample; the free 13-entry ρ vector (155 evaluations, 10-net
batch) recovered 2.99% in-sample and **0.09% on DEV 100** (T3 1.1370e-06 vs T2
1.1380e-06, still 48.1× T0). Gate was "within 15% of T0". No HOLDOUT read was
spent on a 48× arm. Fitted ρ: `experiments/2026-09-23_single_leg/t3_fit_vec16_best.json`.

**Step 6 decision: the third branch.** T2 loses >40% and T3 does not bring it
under 40%, so the single-leg chain (T5/v60) is not built. T4 (carried κ4 on M)
was specified *on the T3 arm*; on an arm 48× worse than T0 it cannot change this
decision and was not run. The κ4-on-M idea itself is not refuted by anything
here and could be tested on the v53 chain's exact tier later.

### Fallback (response §6): the elasticity table on v53 (2026-09-23)

Accuracy: `tools/probe.py` on v53, DEV 100, cupy, paired vs v53
(`experiments/2026-09-23_single_leg/elastic_dev100.json`, `p2_dev100.json`).
Cost: `tools/variants.py --cost`, the real flopscope f64 read-only twin, one net
(`.../elastic/cost.json`, `.../p2/cost.json`). Projected wall uses the 5-anchor
model + 4.1 s band. Negative Δscore is better.

| arm | Δraw | ±95% | wins | ΔFLOPs | ops | Δscore | proj. wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| v53 | — | — | — | — | 25,140 | — | 96.7 s |
| no hubs (`_HUB_ROUTES=()`) | +5.41% | 0.59 | 4 | −0.57% | 24,312 | +4.81% | |
| closure last layer only | +18.03% | 1.22 | 0 | −1.03% | 21,690 | +16.81% | |
| closure off | +18.14% | 1.23 | 0 | −1.03% | 21,634 | +16.92% | |
| shared rank 224 | +6.09% | 0.66 | 3 | −1.71% | 23,346 | +4.27% | |
| age 4 | +18.33% | 1.26 | 0 | −5.77% | 25,728 | +11.49% | |
| age 6 | −5.53% | 0.67 | 94 | +5.15% | 24,552 | −0.66% | |
| P2 rank 16 | +0.40% | 0.18 | 29 | −3.00% | 24,519 | −2.61% | |
| P2 rank 8 | +0.82% | 0.25 | 25 | −4.56% | 23,990 | −3.78% | 92.4 s |
| **P2 rank 4** | **+0.94%** | 0.26 | 24 | **−5.38%** | **23,645** | **−4.49%** | **91.3 s** |
| P2 rank 2 | +11.17% | 4.60 | 10 | −5.82% | 23,645 | +4.70% | |
| P2 rank 1 | +1,175% | 39.6 | 0 | −6.04% | 23,645 | +1,098% | |
| age 6 + P2 rank 8 | −4.69% | 0.74 | 92 | +0.59% | 23,402 | −4.13% | 92.5 s |
| age 6 + P2 rank 4 | −4.51% | 0.75 | 91 | −0.23% | 23,057 | −4.73% | 91.4 s |
| Strassen depth 6 | 0 | — | — | 0 | 25,140 | 0 | (no-op: `_STRASSEN_MIN_BLOCK=32` caps 1024 at depth 5) |

**Reading.** Of the exact tier's components only one has score elasticity < 1:
the **P2 residual rank**. Everything else (hubs, closure, shared rank, age) pays
for itself. Rank 4 keeps rank 8's per-net tail (worst net +4.2% vs +3.8%);
rank 2 is a cliff (worst net 2.3×). Re-spending the saving on age 6 buys only
0.25% more for ~700 MiB of memory, which D7's 7 GiB bound forbids. **v55 =
v53 + `_P2_RANK = 4`** is the fallback canary: −4.49% score, projected wall
91.3 s (v53: 96.7 s), i.e. it also moves under D7's 95 s bound. The T0–T3
single-leg work is closed; H1/H4-type ideas are "measured, superseded by T1–T3".

## D5 — PLAN's premise is BACKWARDS: the old tier is under-funded (2026-09-22)

PLAN D5 calls the old tier "the cost bet" and proposes cheapening it — H1
replaces it with a carried aggregate, H2 refits under "aggressive truncation",
H3 tapers its rank in the tail. **Every one of those directions is measured
wrong.** The chain is not over-spending on the old tier; it is compressing too
early and too hard.

Cost and accuracy on 100 DEV nets, against v50 (rank 256, age 4):

| setting | cost | raw MSE | **score** | projected public |
|---|---:|---:|---:|---:|
| rank 64 | −23.21% | +545.6% | +395.8% | 3.1000e-08 |
| rank 128 | −15.64% | +170.4% | +128.1% | 1.4262e-08 |
| rank 192 | −6.81% | +43.9% | +34.1% | 8.3849e-09 |
| **v50: rank 256, age 4** | — | — | — | 6.2528e-09 |
| rank 320 | +9.34% | −14.89% | **−6.95%** | 5.8185e-09 |
| rank 384 | +19.37% | −19.94% | −4.42% | 5.9762e-09 |
| **age 5** | +6.13% | −15.50% | **−10.32%** | 5.6075e-09 |
| **age 6** | +11.59% | −20.17% | **−10.91%** | **5.5704e-09** |

Two things follow, and both are larger than anything D2 or D3 produced.

**Truncation is dead, and not recoverable by fitting.** Even the mildest cut —
rank 192, saving 6.8% of cost — costs 43.9% of raw MSE and is 34% worse on
score. PLAN H2 proposes rerunning D2's fit on exactly this configuration; D2's
entire lever was 2.7%, against a 44% loss. H1's "carried aggregate" would have
to reproduce content worth 44%+ of raw MSE at 2 units/layer instead of 7, which
the rank curve says is not a small ask but a very large one.

**Spending more wins.** Raising the compression age from 4 to 6 — keeping
sources exact two layers longer — costs 11.6% more FLOPs and returns 20.2% of
raw MSE, a net **−10.91% on score**. Age is a better axis than rank: at
comparable accuracy (≈−20% raw) age 6 costs 11.6% where rank 384 costs 19.4%.

This is the largest single lever found so far. For context, D2's fitted
coefficients were worth −2.7% and D3's billing polish −1.5%, and each took a day.

## D6 — more fitting adds NOTHING on this chain (2026-09-22)

PLAN D6 asks for "a second fitting round on the final structure with more
parameters (per-term-class × layer, ~130) and per-net adaptation". The first
step of that — refitting the existing profiles on v53's structure, plus a newly
wired **old-tier readout gain** (the one profile never fitted, and the one D5
showed carries the chain's accuracy) — returns **0.00%**.

    identity (all profiles 1.0)      2.628931e-08
    D2's shipped coefficients        2.509809e-08    -4.531%
    best of 302 further evaluations  2.509809e-08     0.000%

The search was genuinely exploring, not mis-scaled: of 302 evaluations, 9 landed
within +1% of the incumbent and the closest non-baseline point was +0.196%. None
was better. D2's coefficients sit at a local optimum for this structure, and 16
parameters do no better than 12.

**A first run of this reported the same 0.00% for the wrong reason** and nearly
went on the record as a result about the old-tier gain. `tools/fit.py` seeded
CMA-ES at identity, discarding whatever coefficients the candidate already
carried, so it restarted 4.5% behind and never recovered the point D2 had
already found. Fixed: the search now seeds from the shipped coefficients and
prints both references. The second run above is the honest one.

**What this means, taken with D5.** On this chain, *structure* beats *fitting* by
a wide margin: one integer (compression age 4 → 5) was worth −10.32% on score,
while the entire fitted-coefficient apparatus is worth −2.7% and will not go
further. PLAN's ordering — fitting as "the main accuracy bet", structure as a
cost afterthought — is exactly inverted.

## D2 prep — the six profiles already exist as constants (read 2026-09-21)

PLAN §6 D2(a) asks for six per-layer profiles parametrised as 4-knot splines,
"defaults = current constants exactly". They do not need to be invented: five of
the six are already module-level constants in `estimator_k3_v43_complete_c10.py`,
so the refactor is *read these from an embedded `coeffs.json`*, not a rewrite,
and bit-parity at defaults is close to free.

| PLAN's profile | where it lives in C10 | current form |
|---|---|---|
| λ (κ4 regen) | `_K4_LAM` (line 130) via `_own_lambda()` (1487) | **already per-layer**: a 16-entry table, times `_OWN_S_LAM = 1.225`, times a linear depth tilt `(1 + _OWN_T_LAM·phase)`, `_OWN_T_LAM = -0.1` |
| closure scale θ | `_OWN_THETA_FAM = 0.625` (122), applied at 432 | global scalar |
| κ4→κ3 feed gain | `_OWN_S_FEED = 1.0` (125), applied at 1771–1773 | global scalar |
| D21-feedback gain | `_P3_ON = False`, `_P3_RANK = 32` (117–118) | **off**; D2's base turns it on at rank 16 |
| (2,1)-consumer gain | `_P2_SEPARABLE = True` (79) | boolean, no gain exposed yet |
| old-tier readout gain | `_SHARED_*` (89–95), age 4 / rank 256 | no gain exposed yet |

So four of the six are a scalar or a table to generalise, one is a switch to
turn on, and two ((2,1)-consumer, old-tier readout) need a gain introduced
before they can be fitted. `_own_lambda`'s existing "table × scale × linear
tilt" is already a crude 3-parameter profile — the spline generalises it rather
than replacing a constant, which is a useful sanity check that the
parametrisation is in the right place.

Note `_K4_LAM[15] = 0.0`: the last layer's κ4 channel is already off, and
`_K4_LAM` is documented as "an INITIALISER measured on our own fresh design
networks, not a fitted value". Fitting it on FIT's 100 noise-free organizer nets
is exactly what D2 is for.

## Open items

### C10 canary — PACKAGED AND GATED, NOT YET SUBMITTED (2026-09-21)

Package `experiments/2026-09-21_c10_canary/c10-canary.tar.gz`
(sha256 `1d476ab2d3527eae4ec4729ce012f8e2399bdee7c9d54ccbdc751cca24d3ec00`;
inner `estimator.py` = `fd8d497d…6152d1bf`, byte-identical to
`estimator_k3_v43_complete_c10.py`; LICENSE = the ARC MIT notice shipped with
#330950 and #331317). `whest validate-package` passes.

PLAN §5 gate readings, all green:

| gate | threshold | C10 | verdict |
|---|---|---|---|
| DEV projected vs best graded | ≥3% better than 8.7520e-09 | 6.526e-09 = **25.4% better** | PASS |
| `whest validate` | passes | passes, 96 ms | PASS |
| WSL `linux_memory_gate.py`, RLIMIT_AS 8 GiB | PASS | PASS, VmPeak 5,754.8 MiB, peak RSS 3,890.2 MiB | PASS |
| projected server wall | < 100 s | 60.259 × 1.3097 = **78.9 s** | PASS |
| op calls | ≤ 24,500 | **23,405** | PASS |
| setup | < 2 s | 0.00 s | PASS |
| no bare `except` returning a constant | — | one `except Exception` in `setup`, best-effort, returns nothing; `predict` has no broad handler | PASS |

Advisory (not a §5 gate): projected server residual = 23,405 ops × 10.8–11.2 µs
= **0.252–0.262 s** against the 0.4 s cliff, i.e. 1.53–1.59× headroom. Thinner
than v31's 2.85× and F21's 2.56×. Measuring it is half the point of this canary
(PLAN §6 D0: "the first server residual measurement at 23k ops").

**Blocked:** the submit call was refused by the agent session's own permission
classifier, not by any project gate. Samuel runs it, or grants the permission:

```powershell
.\.venv\Scripts\whest.exe submit experiments\2026-09-21_c10_canary\c10-canary.tar.gz `
  --description "C10 canary" --watch --format plain
```

Slot ledger at the time of packaging: 0/10 used on 2026-09-21 UTC.
