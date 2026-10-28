# L3 — Oct 21: canary B, private-rerun rehearsal (A1, B, + A1s), cross-host determinism, B verification

Handoff: `_claude_snap/handoff_2026-10-19/HANDOFF_2026-10-21_L3_B_CANARY_RERUN_REHEARSAL.md`. Scripts and raw data: `rehearsal/`.
No nomination or designation made (Samuel's).

## 0. Summary

- **Canary B** (#337462) graded **2.0005580379785525e-9**, forecast written first 2.0007e-9 (−0.007 %) ⇒ B is nominee B (done 04:22 UTC,
  record in `experiments/2026-10-20_L3/RESULT.md` §4; repeated in §1 below).
- **300-fresh-seed rehearsal, real flopscope 0.12.3 client ↔ server, one participant process per 100 calls: A1 300/300, B 300/300
  (and A1s 100/100) — 0 errors, 0 non-finite, 0 shape faults, one FLOP count and one op count per build on every net.** Server
  VmPeak ≤ 4.6 GiB under RLIMIT_AS 8 GiB. Pass, with one finding outside the handoff's memory line (§2.4): the *flopscope server*
  grows ≈ 0.55 KB per dispatched op (≈ 11–13 MB per call, ≈ 1.1 GB per 100 calls) — server bookkeeping, identical per op for all three
  builds, no live arrays and no lost frees on our side; ample room under 8 GiB.
- The first rehearsal attempt crashed at call 35–42 per server; **cause was my harness** (weights released outside an open budget
  context ⇒ the client's deferred `free` rode the next `budget_open` and was dropped ⇒ 128 MB/net stranded). Fixed and verified
  flat before the rerun (§2.3). Worth knowing for anyone driving the client by hand; it does not touch the estimators.
- **Determinism:** bill identical across hosts (pod Linux, Windows native at 2 threads and default threads); pod client/server ≡ pod
  in-process bit-for-bit (6/6); Windows 2-thread ≡ Windows default-thread bit-for-bit; **Linux vs Windows not bit-identical**, max
  |Δ| ≤ 7.75e-6 (final layer rms ≈ 1–1.7e-6) ⇒ expected score effect ≈ 2e-4 relative. WSL leg deferred (refit guard, §5).
- **Consumed from L1** (`experiments/2026-10-21_L1/RESULT.md`): smoke shapes (§3) and the worst-day margin (§4). B passes the
  ≥ 10 % worst-day rule on residual (+22.4 %) and wall (+45 %); **A1 does not (+2.5 % residual)**, nor does the λ 1e6 twin (#337470,
  +5.1 %). L1's A1s (`cand_r3s736_lam30.py`, e823d62f; +1.68 % F, 20,630 ops; graded #337481 1.897391e-9) clears it (+13.1 % /
  +10.6 %); the rehearsal had independently predicted its served bill to the FLOP and its residual median (§2.5).

## 1. Canary B (record)

| | forecast (written first, 03:5x UTC) | graded #337462 | A1 #337456 |
|---|---|---|---|
| score | 2.0007e-9 = A1 graded × 306,840,769,522 / 286,123,309,398 (± 0.5 %) | **2.0005580379785525e-9** (−0.007 %) ✔ | 1.8656192469864824e-9 |
| served FLOPs | 306,840,769,522 − offset | 306,840,355,133 | 286,122,895,009 |
| served residual max / median | ≈ 0.28 s | 0.2760 / 0.2594 s | 0.3349 / 0.3261 s |
| served wall max / median | ≈ 83 s (law) | 61.06 / 53.5 s | 83.60 / 78.0 s |
| failed | 0 | 0/100 | 0/100 |

Ledger: 337462 recorded (`experiments/submissions.jsonl`). Report: `experiments/run_reports/canaryB_L3_337462_graded_report.json`.

## 2. Private-rerun rehearsal (handoff §2.1)

### 2.1 Method
- **Nets:** new REHEARSAL seed block **9,250,000 + i, i ∈ [0, 300)** — never used before today; outside FIT/VAL/DEV/FRESH-A/
  FRESH-LOCK/LANE-M (no hit for `9_25…` anywhere in the repo). Same 300 nets for A1 and B (paired); A1s on 0–99.
  Minted by `rehearsal/mint_net.py` (= `tools/fresh/gen.py` `mint_numpy`, protocol 3.0 seed streams). **Mint parity vs whestbench
  0.16.1's own `derive_seed_streams` + `sample_mlp`: weights sha256 and `MLP.seed` identical** on seeds 9,250,000 / 9,250,150 /
  9,250,299 (`rehearsal/pod/mint_parity.json`).
- **Rig:** `flopscope-server` 0.12.3 (+ flopscope 0.12.3, numpy 2.2.6) under **RLIMIT_AS 8,192 MB**, 2 BLAS threads; participant in a
  **client-only venv** (`flopscope-client` 0.12.3, no numpy) with a whestbench stub (MLP/BaseEstimator only), as the grader's
  participant sees it. Weights ingested float32 via the client's free `fnp.load`, cast to float64 server-side. **One server + one
  participant process per 100 calls**, `setup()` once, then `predict()` on 100 distinct nets (`rehearsal/rehearse.py`,
  `client_child.py`). Per call: finite, shape, `flops_used`, client dispatch count, wall / backend / overhead / residual
  (client-side BudgetContext timing), server and client `/proc` VmPeak/VmSize/VmHWM/VmRSS, prediction sha256.
- **Host:** RunPod `gjv6ablp1xm7ce` (32 vCPU, native Ubuntu 24.04 / glibc 2.39, Python 3.12), 7 blocks concurrently.
  Pod times are LOCAL readings, not server figures (AGENTS Measurement) — used only A-vs-B and via the graded calibration in §2.5.

### 2.2 Results (`rehearsal/pod/final/runs/`, `rehearsal/pod/summary.json`)

| build | calls | distinct nets | errors | non-finite | shape faults | FLOPs (one value) | client dispatches (one value) | server VmPeak | client VmPeak | setup |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| **A1** 7d381fc5 | 300 | 300 | **0** | **0** | **0** | 286,122,862,241 | 23,357 | 4,623 MiB | 439 MiB | ≤ 0.002 s |
| A1s e823d62f | 100 | 100 | 0 | 0 | 0 | 290,938,806,455 | 21,324 | 4,517 MiB | 451 MiB | 0.002 s |
| **B** 7a146f84 | 300 | 300 | **0** | **0** | **0** | 306,840,322,365 | 19,577 | 3,550 MiB | 438 MiB | ≤ 0.003 s |

- **Bill constant to the FLOP** on every net, and equal to the graded per-MLP `flops_used` **minus exactly 32,768** for both graded
  builds (A1 286,122,895,009 − 32,768; B 306,840,355,133 − 32,768): the grader also bills one (16 × 1024) × 2 output
  materialisation that this harness does not. ⇒ **A1s will serve at 290,938,839,223 F** (same offset).
- Client dispatch counts exceed the in-process op counts (A1 22,633, A1s 20,630, B 18,943) by 724 / 694 / 634: the client also
  counts free handle traffic. Not used for any projection.

| build | pod residual p50 / p90 / max (s) | worst net | pod wall p50 / max (s) | server RSS growth call 10→100 (per call) |
|---|---|---|---|---|
| A1 | 0.530 / 0.544 / 0.570 | 9,250,106 | 102.1 / 107.5 | +1,072 / +1,070 / +1,188 MiB (12.0–13.4 MB) |
| A1s | 0.481 / 0.494 / 0.498 | 9,250,015 | 94.7 / 99.7 | +1,034 MiB (11.6 MB) |
| B | 0.419 / 0.445 / 0.475 | 9,250,163 | 77.4 / 81.6 | +944 / +974 / +1,084 MiB (10.6–12.2 MB) |

### 2.3 Harness incident (first attempt, 04:56–06:00 UTC; `rehearsal/pod/runs_leaky_partial/`)
Every server hit RLIMIT_AS at call 35 (A1) / 42 (B): `MemoryError` → handshake timeout → cascading "Nested BudgetContext".
Attribution: an ingestion-only run (no `predict`) grew the server by **exactly 128.06 MB per net** = one float64 16×1024×1024 stack.
Mechanism (flopscope-client 0.12.3 `_connection.py` / `_handles.py`): a RemoteArray's finalizer only *enqueues* its free; the queue
rides the next request. My harness dropped the weights after the predict context closed, so the free rode the next context's
`budget_open` — no session yet — and was discarded ("handles then linger until the session closes"). The float32 stack, dropped
*inside* a context, was freed normally. **Fix:** drop the harness's references inside an open context and flush with one trivial
op; verified flat (server RSS 75 MB over 10 nets, `pod/final/leak/fixcheck_load_only_*`) before the full rerun. A leak probe
(`rehearsal/leak_probe.py`) confirms neither estimator retains any weight proxy after `predict`.

### 2.4 Server growth after the fix — attribution
The handoff's pass line is "no growth beyond the known native-glibc +2 MB/call"; the fixed rerun shows **11–13 MB/call on the
server** (client flat). It is not the estimators:
- `rehearsal/free_probe.py` (same net repeated, local rig, B): **13,615 frees enqueued = 13,615 sent, 0 dropped**; after each call the
  only live RemoteArrays are the 4 weights; `est.last_run` holds scalars only.
- Predict-only on one net (`pod/final/leak/*_same_net_*`): the same ≈ 10–11 MB/call without any ingestion.
- Growth per dispatched op is the same for all three builds: **A1 0.53, A1s 0.54, B 0.57 KB/op** — it scales with op traffic,
  i.e. flopscope-server per-op bookkeeping (or its allocator), not anything our code keeps.
Consequence: in a server that lived through 100 MLPs, ≈ +1.1–1.3 GB on top of a ≈ 3.3–4.4 GB working peak ⇒ VmPeak ≤ 4.6 GiB here,
≥ 1.7× under 8 GiB. Both canaries served 100/100 with no memory failure, consistent with this.

### 2.5 Calibration of pod residual to served (cross-check of the A1s projection)
Pod/served residual ratio from the two graded builds: **median 1.626 (A1) / 1.616 (B); max 1.703 / 1.720** — the same factor for
both, so the pod residual is a faithful *relative* instrument. Applied to A1s: served-equivalent residual median **≈ 0.296 s**
(L1's projection 20,630 ops × 14.26 µs = 0.294 s ✔); A1s/A1 pod p50 ratio 0.906 vs ops ratio 0.911. Worst-day central (× 1.196) ≈
0.354 s ⇒ **+11.5 % headroom** (L1: +12.0 %). Pod within-run tails (max/p50: A1 1.075, A1s 1.036, B 1.133) are pod noise, not grader
tails, and are not used.

## 3. Smoke shapes (owned by L1 — consumed; `experiments/2026-10-21_L1/RESULT.md` §1)

| shape | A1 | B |
|---|---|---|
| 256×4, 512×16, 1024×16, 2048×4 | PASS (setup ≤ 0.003 s) | PASS |
| **1024×32** | PASS, 9.67e11 F, local 186 s (⇒ ≈ 220 s served: a 1024×32 net would time out under 120 s) | PASS, 1.068e12 F, local 149 s (≈ 165 s served) |
| 1024×16 at weight scale × 0.9 / × 1.1 | PASS | not run (same code path) |
| **1024×16 at × 0.5 / × 2** | **FAIL IndexError** (`_yp_contract`, known S8.4 limitation) | **FAIL IndexError** |

Exact 8be (#336993) crashes identically at × 0.5 and passed the organisers' smoke test, as have all 82 graded runs (their smoke is a
small He shape). Hardening exists but is a finals-path code change — **strategist's call**; not made here.

## 4. Slow-node margin (owned by L1 — consumed; L1 RESULT §2, 82 graded runs / 23 UTC days)

Per-day telemetry (worst / p90 / median residual µs/op): era-set, not day-set — 10.1–10.6 µs/op to Oct 12, 13.5–14.2 from Oct 15
(materialisation-era builds); worst single reading 15.92 µs/op (#336147, Oct 15). Kernel ps/FLOP 88–100 (leaf 64) / 150–273 (leaf-32 /
planner). Worst within-run tail on record: residual 1.196 (#334487, Oct 6), wall 1.242 (#336083, Oct 14). Full table: L1 RESULT §2.

**3-row headroom table** (central = graded median × worst tail on record; pessimistic adds the worst same-code day shift 1.029/1.031):

| nominee | ops | worst-day residual central / pess. | residual headroom (to 0.40 s) | worst-day wall central / pess. | wall headroom (to 120 s) | first fails at (× graded run) |
|---|---:|---|---|---|---|---|
| A1 7d381fc5 (#337456) | 22,633 | 0.390 / 0.401 s | **+2.5 % / −0.3 %** ✗ | 96.8 / 99.9 s | +19 % / +17 % | 1.194 (residual) |
| λ 1e6 twin 73321458 (#337470, graded 1.8708e-9) | 22,235 | 0.379 / 0.390 s | +5.1 % / +2.4 % ✗ | 94.3 s | +21 % | ≈ 1.23 (residual) |
| **B 7a146f84 (#337462)** | 18,943 | **0.310 / 0.319 s** | **+22.4 % / +20.2 %** ✔ | 66.4 / 68.5 s | **+45 % / +43 %** ✔ | 1.449 (residual; wall 1.979) |
| A1s e823d62f (#337481, graded 1.897391e-9) | 20,630 | from graded (L1): **+13.1 % central / +10.6 % pess.**; pre-grade projections 0.352 s (L1) / 0.354 s (§2.5 rehearsal) | ✔ | wall max 74.0 s served | ✔ | 1.323 (residual) |

Decision rule (handoff §3, strategist applies it): B survives the worst day with ≥ 10 % on both. **A1 does not; the λ 1e6 twin does not
either.** L1's A1s is the planner-only build that clears it, with A1's accuracy at +1.68 % F: graded #337481 1.897391e-9 (forecast
1.8970e-9), served 290,938,839,223 F — **exactly the rehearsal's bill + 32,768** (§2.2, predicted before the grade) — 100-call pod
memory PASS (L1, commit c44bbfb). L2's A3 (#337495, canary in flight) is the same code with refitted constants: the rehearsal's
code-path results (finite, constant bill, server memory) carry over to it and to its planner variants; its own residual comes from
its op count.

## 5. Determinism across hosts (handoff §2.4; `rehearsal/det_*.jsonl`, `rehearsal/pod/det_pod.jsonl`)

In-process real flopscope 0.12.3, float64 read-only weights, REHEARSAL nets 9,250,000–002, A1 and B.

| comparison | result |
|---|---|
| FLOPs, all hosts | identical: A1 286,123,276,630, B 306,840,736,754 on every net and host (A1 22,632 / B 18,942 ops; +32,768 F / +1 op is verify_nominee's materialise call) |
| pod client↔server vs pod in-process (native Linux) | **bit-identical 6/6** |
| Windows native 2 BLAS threads vs default threads | **bit-identical 6/6** |
| pod Linux (2 threads) vs Windows native (2 threads) | **not bit-identical**: max \|Δ\| 3.9e-6 … 7.75e-6 (all layers); final layer rms 0.89–1.70e-6 |

Expected score effect of the Linux–Windows difference: δ uncorrelated with the error (rms error 1.2e-4) ⇒ ΔMSE/MSE ≈ δ²/MSE + cross
term ≈ 2e-4 (Cauchy–Schwarz worst case 1.5–2.9 %, not credible); empirical support: both canaries graded within 0.007 % of
forecasts built from locally computed raw MSE. Same magnitude as the documented 2-vs-4-thread difference (≤ 1.1e-5), benign.
**WSL leg not run:** L2's A3 refit (#3645) and L4's refit (#3650) were running/queued all morning; a WSL job tripped exactly that
guard on Oct 20 (R3 death). Windows native is the cross-host substitute; WSL can run when no refit is live.

## 6. Nominee B verification (handoff §4)

| check | result | evidence |
|---|---|---|
| checkout / sha | committed == working == DESIGNATION `7a146f84` | verify_nominee #3611 |
| imports / static | flopscope/whestbench/stdlib only; client-only import (no numpy); contract checks PASS | #3611 |
| meter / bill | 306,840,769,522 F / 18,943 ops on DEV0–7, = graded + 414,389 offset ✔ | #3611, #337462 |
| determinism | 2 processes × 3 DEV nets bit-identical; cross-host §5 | #3611 |
| project | residual-eq 0.2887 s, wall-eq 83.0 s (served actual 0.276 / 61.1 s) | #3611, #337462 |
| rig | 1024×16 / 512×16 / 256×4 PASS through the real client | #3625 |
| memory, 100 calls, RLIMIT_AS 8 GiB | native Linux in-process: VmPeak 3,668 MB, settled −12.1 MB ✔; client/server: server VmPeak 3,550 MiB over 100 distinct nets ✔ (server bookkeeping §2.4); WSL on final B not run (#3634 cancelled 04:38 by Samuel); B-on-8be WSL 3,691 MiB / +32 MiB | `experiments/2026-10-20_L3/pod_artifacts/mem/`, §2 |
| graded | #337462 2.00056e-9, 0/100 failed | §1 |
| rehearsal | 300/300 fresh nets clean, bill constant | §2 |
| worst-day headroom | residual +22.4 % / wall +45 % (L1) | §4 |

**All B checks PASS.** `docs/DESIGNATION.md` B row updated with the rehearsal, determinism and headroom lines.

## 7. Spend and open items
- RunPod `gjv6ablp1xm7ce` 04:46–09:01 UTC (4.24 h × $1.19) ≈ **$5.05**; stopped. L3 total for the lane ≈ $13.2.
- Open (not mine to decide): the × 0.5 / × 2 IndexError hardening (§3); A vs A1s choice (§4). WSL determinism leg when no refit runs.

## 8. Close-out (Oct 21 15:55 UTC, re-dispatch of the unchanged L3 brief)
Every handoff item was already complete and committed (503d987): canary B graded, 300/300 rehearsal for A1 and B, determinism,
B verification, DESIGNATION B row. Nothing re-run. **B3 (B on A3's chain) not built:** DECISIONS D6 (16:00) fixes the nominees
as A4s + B (#337462) and closes the finals path except L7's A5 arms; A3 itself is off the pair (cliff). **WSL determinism leg stays
deferred:** L7's local A5 arms are the kind of live refit a WSL job tripped on Oct 20; Windows native remains the cross-host
substitute (§5). L3 lane closed; no pod running, no spend since §7.
