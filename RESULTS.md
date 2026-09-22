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

- **DEV → graded**: divide the 100-net DEV adjusted score by **1.037**. Observed
  **1.0386 / 1.0385 / 1.0339 / 1.0426** over four submissions — mean 1.0384,
  spread ±0.45%. The constant is deliberately left at the *low* end of that
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

- **one-net twin wall → server WORST-of-100 wall**: multiply by **1.5981**.
  Two anchors, both measured with `tools/_cost_worker.py`:

  | candidate | twin wall (1 net) | graded worst wall | factor |
  |---|---:|---:|---:|
  | F21 #330950 | 58.295 s | 93.1597 s | 1.5981 |
  | v31 #331317 | 43.547 s | 68.1319 s | 1.5646 |

  The factor deliberately bundles *server slowness* with *worst-of-100 vs our
  one net*, because the gate asks whether the worst net finishes. The larger
  anchor is used: over-projecting wall costs nothing, under-projecting costs the
  submission.

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
