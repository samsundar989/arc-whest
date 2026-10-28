# L1 — Oct 21: rerun-risk numbers (smoke shapes, slow-node margin), write-up ledger

Handoff: `_claude_snap/handoff_2026-10-19/HANDOFF_2026-10-21_L1_RERUN_RISK_WRITEUP_LEDGER.md`. L5 closed (K_pooled_carrier.md), not started.
Scripts here: `smoke_shapes.py` (real client↔server 0.12.3), `telemetry.py` → `telemetry.json`, `margin.py` → `margin.json`,
`ops.json` (op calls of every submitted source, with evidence path per id), `repro_scale.py` (in-process traceback).

## §1 Smoke shapes — A1 `7d381fc5` and B `7a146f84`

Real `flopscope-client` 0.12.3 ↔ local `flopscope-server` 0.12.3 (`cache/verify_rig`), one fresh server + client process per shape,
weights N(0, (k·√(2/W))²) seeded per shape, f64, `BudgetContext(2^41)`. workerq #3635 (A1), #3637 (B); raw rows `smoke_A1/smoke.json`,
`smoke_B/smoke.json`. Times are LOCAL (not server figures); F = `flops_used` of predict alone.

| shape (W×D, scale) | A1 | A1 F | A1 setup / predict s | B | B F | B setup / predict s |
|---|---|---:|---|---|---:|---|
| 256×4 | PASS | 7.864e8 | 0.001 / 1.3 | PASS | 8.244e8 | 0.001 / 1.4 |
| 512×16 | PASS | 7.212e10 | 0.001 / 23.6 | PASS | 8.146e10 | 0.000 / 23.0 |
| 1024×16 (suite) | PASS | 2.86123e11 (= graded bill + 16k) | 0.001 / 65.4 | PASS | 3.06840e11 (= bill) | 0.001 / 53.9 |
| **1024×32** | PASS | 9.670e11 (3.38× of D16) | 0.001 / 185.9 | PASS | 1.0678e12 (3.48×) | 0.002 / 148.9 |
| 2048×4 | PASS | 1.722e11 | 0.003 / 25.9 | PASS | 1.909e11 | 0.003 / 25.8 |
| 1024×16, ×0.5 | **FAIL** IndexError | — | — | **FAIL** IndexError | — | — |
| 1024×16, ×2 | **FAIL** IndexError | — | — | **FAIL** IndexError | — | — |
| 1024×16, ×0.9 / ×1.1 (#3646) | PASS / PASS | 2.86123e11 both | 0.001 / 134.1, 96.9 | not run | | |

×0.9 and ×1.1 are exactly homogeneous (last-layer mean 0.1124 = 0.606 × 0.9¹⁶; 2.787 = 0.606 × 1.1¹⁶), so the crash band starts
somewhere between ×1.1 and ×2 (and ×0.9 and ×0.5).

All PASS rows: finite, (D, W) shape, outputs identical between A1 and B to 4 s.f. (same chain), setup ≤ 0.003 s (≪ 2 s).
Bill scaling: depth 16→32 costs 3.4–3.5× (the exact-age/band carriers keep growing with depth; per-layer cost ×1.7), width 512→1024
×3.97 (A1) / ×3.77 (B). **1024×32 is 0.44–0.49 B — fits the budget; local predict 186 s / 149 s ⇒ served ≈ 220 s / 165 s, so a
1024×32 net under a 120 s wall would time out.** Our graded smoke tests (all 82 graded runs) passed in 20–25 s over 5 workers, i.e.
the organisers' smoke shape is small; the scored shape is 1024×16.

**Non-He scale crash.** Traceback (in-process, workerq #3642): `predict` → `_tower` → `_layer` → `_dslice21` →
`_yp_contract` (`cand_r3s736_lam06.py:973`, `fnp.take(LA, kr, axis=0)`) `IndexError: index 928 is out of bounds for axis 0 with size 928`
(×2: 896/896), preceded by overflow / invalid-value warnings in the κ4 / λ-law terms. This is the **known limitation recorded in
`RESULTS.md` § S8.4 (line 3331; every build since v87_B; "not fixed, by decision")**: a non-finite α puts NaN scores past the 1e30
sentinel in the k1 argsort and `_oinv` maps rows out of range. What is new: before, only planted pathological weights (×1e3 column,
×1e-3 layer, rank 1) triggered it; **a global weight scale ×0.5 or ×2 at width 1024 now triggers it in A1 and B** (512×16 and 256×16
at ×0.5 pass in-process, with overflow warnings). **8be (#336993, the current official best) crashes identically at ×0.5 (workerq #3644)** — pre-existing, and that build passed the organisers' smoke test, so their smoke does not use this scale. It is a hard crash (contract rule 7: a FAILED MLP, not a silent wrong
answer). Phase-2 nets — public, private rerun — are He by construction; risk is confined to an organiser smoke test with non-He
scale, which every graded submission so far has passed. Hardening exists (`_sc = where(isfinite(_sc) & (_ok > 0), _sc, 1e30)`
before the argsort) but is a code change on the finals path (rig + canary): the strategist's call, not made here.

## §2 Slow-node margin

### Data
82 graded submissions from Sep 14 to Oct 21 on **23 distinct UTC days**; per-MLP grader telemetry (100 MLPs each) from
`experiments/run_reports/` (14 missing reports fetched today with `local_sim/fetch_graded_report.py`, saved as
`L1fetch_<id>_graded_report.json`). Ops: 81 of 82 from `ops.json` (80 high-confidence, evidence path per id; #336253 ≈ 22,846;
#330950 none). Full per-submission table: `telemetry.json` (res max/median, wall max/median, kernel ps/FLOP, overhead share,
µs/op, start time). Board "wall" = our `participant_predict_wall_s` (min/median match exactly on #336993).

### Per day (pooled over that day's 100-MLP runs)

| day (UTC) | runs | residual µs/op worst / p90 / median | kernel ps/FLOP worst / p90 / median | overhead share (median) | worst residual tail (max/med) | worst wall tail |
|---|---:|---|---|---:|---:|---:|
| 2026-09-14 | 1 | — (no ops) | 94 / 93 / 90 | 0.064 | 1.041 | 1.041 |
| 2026-09-16 | 1 | 10.76 / 10.60 / 10.44 | 95 / 93 / 92 | 0.077 | 1.031 | 1.033 |
| 2026-09-22 | 3 | 10.68 / 10.46 / 10.06 | 173 / 170 / 159 | 0.101 | 1.056 | 1.053 |
| 2026-09-23 | 5 | 11.86 / 11.26 / 10.22 | 219 / 191 / 164 | 0.105 | 1.142 | 1.234 |
| 2026-09-24 | 2 | 10.50 / 10.36 / 10.22 | 176 / 173 / 154 | 0.101 | 1.023 | 1.044 |
| 2026-09-25 | 4 | 11.22 / 10.39 / 10.11 | 194 / 177 / 170 | 0.096 | 1.121 | 1.137 |
| 2026-09-26 | 5 | 11.10 / 10.39 / 10.14 | 193 / 174 / 166 | 0.097 | 1.096 | 1.150 |
| 2026-09-27 | 1 | 10.65 / 10.36 / 10.25 | 95 / 90 / 88 | 0.132 | 1.039 | 1.073 |
| 2026-09-28 | 4 | 11.20 / 10.76 / 10.31 | 181 / 169 / 159 | 0.098 | 1.067 | 1.135 |
| 2026-10-06 | 4 | 12.43 / 10.64 / 10.28 | 203 / 181 / 138 | 0.126 | 1.196 | 1.153 |
| 2026-10-07 | 3 | 10.76 / 10.50 / 10.33 | 205 / 199 / 177 | 0.112 | 1.041 | 1.058 |
| 2026-10-08 | 3 | 11.17 / 10.62 / 10.40 | 228 / 205 / 196 | 0.108 | 1.074 | 1.135 |
| 2026-10-09 | 3 | 11.46 / 10.60 / 10.37 | 237 / 203 / 192 | 0.108 | 1.129 | 1.206 |
| 2026-10-10 | 4 | 11.43 / 10.82 / 10.52 | 221 / 197 / 188 | 0.105 | 1.093 | 1.146 |
| 2026-10-11 | 4 | 12.29 / 11.46 / 10.57 | 236 / 219 / 197 | 0.104 | 1.158 | 1.133 |
| 2026-10-12 | 4 | 11.10 / 10.68 / 10.46 | 242 / 220 / 202 | 0.102 | 1.069 | 1.120 |
| 2026-10-13 | 4 | 12.73 / 12.32 / 11.82 | 204 / 199 / 175 | 0.134 | 1.085 | 1.118 |
| 2026-10-14 | 5 | 14.12 / 13.68 / 12.24 | 273 / 230 / 195 | 0.131 | 1.090 | 1.242 |
| 2026-10-15 | 7 | 15.92 / 14.33 / 13.77 | 244 / 223 / 196 | 0.143 | 1.141 | 1.183 |
| 2026-10-16 | 7 | 15.47 / 14.15 / 13.78 | 248 / 226 / 211 | 0.132 | 1.112 | 1.105 |
| 2026-10-17 | 5 | 15.54 / 13.94 / 13.47 | 237 / 203 / 195 | 0.137 | 1.166 | 1.221 |
| 2026-10-18 | 1 | 15.24 / 15.01 / 13.86 | 245 / 238 / 219 | 0.135 | 1.100 | 1.114 |
| 2026-10-21 | 2 | 14.80 / 14.59 / 14.15 | 252 / 239 / 196 | 0.139 | 1.064 | 1.133 |

Readings. (1) **Per-op residual rate is set by the code mix, not the day**: day medians 10.1–10.6 µs/op through
Oct 12, 11.8–12.2 on the mixed Oct 13–14 days, 13.5–14.2 from Oct 15 on (materialisation-era builds only); within each era every day
median is within ±3 % of the era's.
The worst per-op reading on record is #336147 (Oct 15, v90a_Bws) 15.92 µs/op max. (2) **Kernel ps/FLOP is also mix-set** (Strassen
leaf: 88–100 on leaf-64 builds, 150–250 on leaf-32/planner builds). (3) **Day noise shows as the within-run tail**: max/median of
residual and wall per run; worst on record: **residual 1.196 (#334487 v77, Oct 6)**, **wall 1.242 (#336083 v89_r2, Oct 14)**; deciles
of the residual tail 1.02–1.12, wall 1.03–1.15. Per MLP, residual and wall move together (corr 0.87–0.98 on the noisy runs): both are
node speed. Across runs the residual tail inflates at 0.67× the wall tail (through-origin slope). (4) **Day-level median shift** of
byte-identical code: v50 ×3 (Sep 22 → Sep 23 noisy) +2.9 % residual / +3.1 % wall; v55 ×2 +0.7 % / +0.4 %.

### Projection method
Our bill is constant over MLPs, so the per-MLP spread inside one graded run is grader-node noise. For each nominee:
- **central** = own graded median × worst tail on record (residual 1.196, wall 1.242);
- **pessimistic** = central × worst same-code median shift (1.029 / 1.031) — two worst cases compounded;
- **per-op cross-check** = ops × worst per-op rate on record (15.92 µs/op) — biased low for A1, whose own rate (14.41 median) is the
  highest median on record.
- fail slowdown s* = cap / max: on top of the graded run (uniform), and on top of the central worst day.

### Nominee table (the 3 rows; L3 consumes this)

| nominee | ops | graded residual med / max (s) | **worst-day residual** central / pess. / per-op | **residual headroom** central / pess. / per-op | graded wall med / max | **worst-day wall** central / pess. | wall headroom | first fails at (× graded run) | (× central worst day) |
|---|---:|---|---|---|---|---|---|---|---|
| **A1** `7d381fc5` λ 0.6e6 (#337456) | 22,633 | 0.3261 / 0.3349 | **0.390 / 0.401 / 0.360** | **+2.5 % / −0.3 % / +9.9 %** | 78.0 / 83.2 | 96.8 / 99.9 | +19 % / +17 % | residual 1.194 (wall 1.442) | residual **1.025** (wall 1.239) |
| A1 λ 1e6 twin `cand_r3s736_lam10` (proj. from 8be #336993: same planner, same 22,235 ops) | 22,235 | 0.3081 / 0.3389 | 0.369 / 0.379 / 0.354 | +7.9 % / +5.2 % / +11.5 % | 73.1 / 81.4 | 90.8 / 93.6 | +24 % / +22 % | residual 1.180 (wall 1.474) | residual 1.085 (wall 1.322) |
| — same twin, alt. (A1's run × 22,235/22,633) | 22,235 | 0.3203 / 0.3290 | 0.383 / 0.394 / 0.354 | +4.2 % / +1.4 % / +11.5 % | 78.0 / 83.2 | 96.8 / 99.9 | +19 % / +17 % | residual 1.216 | residual 1.044 |
| **B** `7a146f84` (#337462) | 18,943 | 0.2594 / 0.2760 | **0.310 / 0.319 / 0.302** | **+22.4 % / +20.2 % / +24.6 %** | 53.5 / 60.6 | 66.4 / 68.5 | +45 % / +43 % | residual 1.449 (wall 1.979) | residual 1.289 (wall 1.806) |

**Verdict against the L3 rule (survive the worst day on record with ≥ 10 % headroom on residual and wall):**
- **B passes** under every method (≥ 20 % residual, ≥ 43 % wall).
- **A1 fails the rule on residual** (central +2.5 %; pessimistic touches the cliff; only the low-biased per-op cross-check reaches
  +9.9 %). Wall is fine (+17–19 %). A1's binding constraint is the 0.4 s residual cliff, which a uniform 1.19× slowdown of its graded
  run reaches.
- **The λ 1e6 twin does not fix it robustly**: −1.8 % ops buys +2–5 points; it passes only under the per-op cross-check (+11.5 %) and
  fails central (+4–8 %) and pessimistic (+1–5 %). Its value depends on whether A1's 4 % higher per-op rate vs 8be is the λ 0.6e6 mix
  (then twin ≈ 8be row) or the Oct 21 node (then twin ≈ alt. row); one graded run cannot separate the two.
- **What would pass**: ops ≤ **20,889** at A1's own per-op rate (≤ 21,719 at 8be's) gives ≥ 10 % central residual headroom; B (18,943)
  is the existing build inside that line (same chain, same accuracy, +7.24 % F).

### Leaders (top entries from G §3; wall only — their residual is not shown on the pages)

| entry | adj | wall med / max (page) | first fails at (× graded run) = 120 / max | central worst-day wall (med × 1.242) | × worst day to fail | fails before A1 (1.194)? |
|---|---|---|---:|---:|---:|---|
| marius_binner #337106 | 1.310e-09 | 87.8 / 98.2 | 1.222 | 109.0 | 1.100 | no |
| oliver_chen #337003 | 1.418e-09 | 103.3 / 110.6 | 1.085 | 128.3 | 0.935 | yes |
| J2W #337185 | 1.418e-09 | 109.5 / 118.3 | 1.014 | 136.0 | 0.882 | yes |
| mliston #337142 | 1.433e-09 | 100.1 / 112.0 | 1.071 | 124.3 | 0.965 | yes |
| Luna #336449 | 1.484e-09 | 98.6 / 103.4 | 1.161 | 122.5 | 0.980 | yes |
| Luna (newest) #337205 | 1.503e-09 | 96.4 / 104.8 | 1.145 | 119.7 | 1.002 | yes |
| AndreasHad04 #337136 | 1.614e-09 | 104.6 / 109.2 | 1.099 | 129.9 | 0.924 | yes |
| thegamers #337296 | 1.830e-09 | 96.3 / 98.5 | 1.218 | 119.6 | 1.003 | no |
| andrei_bulzan #337327 | 1.833e-09 | 86.2 / 116.3 | 1.032 | 107.1 | 1.121 | yes |
| megan_park #337303 | 1.930e-09 | 105.4 / 107.1 | 1.120 | 130.9 | 0.917 | yes |
| **us: A1** #337456 (residual) | 1.866e-9 | 78.0 / 83.2 | **1.194** (residual; wall 1.442) | 96.8 | 1.025 (residual) | — |
| **us: B** #337462 (residual) | 2.001e-9 | 53.5 / 60.6 | **1.449** (residual; wall 1.979) | 66.4 | 1.289 (residual) | — |

Readings. (1) On a node uniformly slower than the one that graded them, **8 of these 10 entries fail before A1 does** (J2W at
+1.4 %, andrei_bulzan +3.2 %, mliston +7.1 %, oliver_chen +8.5 %, AndreasHad04 +9.9 %, megan_park +12 %, Luna +14.5–16.1 %); marius
(1.222) and thegamers (1.218) fail just after A1 (1.194); **B (1.449) outlasts every entry**. (2) Replaying our worst day on record
(wall tail 1.242) on their medians, 7 of 10 exceed 120 s; only marius (109 s), andrei_bulzan (107 s by median — but his own graded max
is already 116.3 s) and, marginally, Luna #337205 / thegamers (119.6–119.7 s) stay under. (3) If node noise inflates residual at the
empirical 0.67× of wall (rather than 1×), A1's wall-equivalent fail point moves from 1.194 to ≈ 1.29 — then A1 outlasts marius and
thegamers too. (4) Caveats: their nominated entries may be different (safer) builds; page maxima may understate by ~1.6 % (ours:
80.1 page vs 81.4 telemetry); their residual cliff, invisible to us, can only make them fail earlier.

**What we are insured against.** A1 is the score entry on a normal node; a node 1.20–1.45× slower removes A1 *and* every leader
measured here, and **B is then the only entry standing** — the slow-node route to the top 5 runs through B, not A1.

## §3 Write-up ledger
`docs/WRITEUP_LEDGER_2026-10-21.md`: 39 rows for mechanisms in A1 (each with symbol:line in `cand_r3s736_lam06.py`) and 47 rows for
measured levers outside it (incl. negatives: rank-k covariance state, pooled carrier K, X1-B on A1, X3 e₁), numbers with path:line.
Spot-checked against source: X2 B1/C/D (`experiments/2026-10-20_x2/RESULT.md:16-19`), r288 (`RESULTS.md:5723`). Gaps (no measured
number found): floor-theorem derivation file; `first-principles-sandbox-2026-10-19` (no such path; mapped to F_math.md — confirm);
Edgeworth readout, λ law, corrector and KAS as single ablations on A1/8be; Q31 alone (only inside C1); batched-vs-unbatched Strassen
F Δ; band "triangular variant"; A3 (running). Six numbers marked "(check)" are second-hand.

## §4 Diagnostic canary — A1's λ 1e6 twin (triggered by §2: A1 fails the ≥ 10 % worst-day residual rule; L3 handoff §2.3 fallback)

Twin = `experiments/2026-10-20_L2/src/cand_r3s736_lam10.py` sha **73321458** (A1 with `_SW_OP_FLOPS` 1e6, 8be's planner; the step-736
export unchanged). Package `twin-r3s736-lam10-73321458.tar.gz` sha256 `08927fdc49df0b19…`, inner estimator = source, validate +
validate-package PASS. WSL 100-call memory #3627 PASS (VmPeak 4,523 MiB, +0.0 MiB). verify_nominee (rig/meter/project) #3649.

**Forecast (written before submission, 05:10 UTC).** Score = A1 graded 1.86562e-9 × F ratio 286,868,957,368 / 286,123,309,398
(1.002606) × raw ratio 1.00006 (X2-D: λ 0.6e6 vs 1e6 raw +0.006 % [−0.029, +0.038]) = **1.8706e-9** (interval 1.866–1.876e-9).
Served bill 286,868,542,979 (= 8be graded). 0/100 failed.

**What it discriminates.** A1's served residual median per op was 14.41 µs vs 8be's 13.86 µs (+4 %). If the twin's median per op ≈ 13.9
(±3 % day noise), the difference is A1's λ 0.6e6 op mix and the twin's worst-day residual is the 8be row (0.369 s central, +7.9 %);
if ≈ 14.4, it was the Oct 21 node and the twin row is the alt. (0.383 s, +4.2 %). Either way the twin's central headroom stays short of
the 10 % rule; the canary's purpose is to give the fallback A a graded number (L3 rule) and to fix which residual model applies.

**Graded #337470 (05:15–05:42 UTC): 1.8708071209909215e-9** (forecast 1.8706e-9, +0.011 % ✔; A1 × 1.00278), raw 1.43409e-8,
0/100 failed, F 286,868,542,979 (= 8be graded). Served residual median 0.3171 / max 0.3256 s (tail 1.027), wall 75.9 / 79.2 s.
**Per-op 14.26 µs median (A1 same morning 14.41; 8be Oct 18 13.86) ⇒ A1's +4 % was the Oct 21 grader node, not the λ 0.6e6 mix.**
So the twin's worst-day row is the "alt." one: **central 0.379 s (+5.1 %), pessimistic 0.390 s (+2.4 %)**, wall 94.3 s (+21 %).
The twin buys 2.6 points of residual headroom over A1 for +0.28 % score and still fails the ≥ 10 % rule; it is not the fix.
Today's node is ≈ 3–4 % slower per op than Oct 18 — the worst-day multipliers above apply on top of it.

## §5 Residual-safe A from the planner — A1s (beyond the handoff; feeds the 12:00 nominee decision)

Planner-only sweep on A1's chain (`planner_sweep.py`, workerq #3651, real 0.12.3 meter, DEV0; bills are shape-determined; output
max |Δ| vs A1 5.7e-6 … 9.8e-6 = the planner-reorder level, B's VAL/FRESH Δraw −0.01 %). Headroom = central worst-day residual at
today's node rate (14.26 µs/op, #337470) × 1.196:

| cap, pad, leaf, λ_op | F | ΔF vs A1 | ops | worst-day residual | headroom |
|---|---:|---:|---:|---:|---:|
| 6, pad, 8, 6e5 = **A1** | 286,123,309,398 | — | 22,633 | 0.386 s | +3.5 % |
| 6, pad, 8, 2e6 | 289,458,528,215 | +1.17 % | 20,952 | 0.357 s | +10.6 % |
| **6, pad, 8, 3e6 = A1s** | **290,939,253,612** | **+1.68 %** | **20,630** | **0.352 s** | **+12.0 %** |
| 6, pad, 8, 5e6 | 297,038,124,359 | +3.81 % | 19,328 | 0.330 s | +17.6 % |
| 6, pad, 8, 8e6 | 311,862,970,590 | +9.00 % | 17,293 | 0.295 s | +26.2 % |
| 5, pad, 8, 1e6 / 3e6 | 297.74 / 301.08 G | +4.06 / +5.23 % | 21,773 / 20,225 | 0.371 / 0.345 s | +7.1 / +13.7 % |
| 5, pad or no pad, 8, 5e6 = **B** | 306,840,769,522 | +7.24 % | 18,943 | 0.323 s | +19.2 % |

Pad has no effect at cap 5 (identical bills with and without). **A1s** = `src/cand_r3s736_lam30.py` sha **e823d62f** (one line vs A1:
`_SW_OP_FLOPS = 3000000`), package `a1s-r3s736-lam30-e823d62f.tar.gz` sha256 `14f94773…`, inner = source, validate(-package) PASS.
verify_nominee #3652. **Forecast (before submission): A1 graded × 290,939,253,612 / 286,123,309,398 × raw 1.0000 = 1.8970e-9**
(interval 1.894–1.900e-9; −5.2 % vs B, +1.68 % vs A1). Pessimistic worst-day residual 0.362 s (+9.5 %). **Not yet run:** 100-call WSL
memory (deferred — a local WSL job can trip L2's running A3 refit guard), FRESH-A100.

**A1s graded #337481 (06:13–06:38 UTC): 1.897391135725812e-9** (forecast 1.8970e-9, +0.02 % ✔), raw 1.43412e-8 (A1 1.43384e-8,
+0.02 %), 0/100 failed, smoke PASS, F 290,938,839,223 (= bill − 414,389, the usual offset). Served residual median 0.2905 / max
**0.3023 s** (14.08 / 14.66 µs/op), wall 70.2 / 74.0 s. **Worst day: residual 0.348 s central (+13.1 %), 0.358 s pessimistic
(+10.6 %) — passes the ≥ 10 % rule under both; wall 87.2 s (+27 %).** Uniform slowdown to first fail 1.323× (A1 1.194×, B 1.449×;
marius 1.222×, thegamers 1.218×). Score: A1 × 1.0170, B × 0.9484.
Still owed before A1s can be a nominee: 100-call WSL memory (deferred: L2's A3 refit is running locally), FRESH-A100 paired vs A1.

## §6 D4 follow-up — nominee checks for A1s (and A3s/B3 when posted)

Pod (Samuel approved pods, 07:12 UTC; D4 ≤ $15): **`arc-whest-L1-memgate` v0fza2b41jnbyw**, created by L1 07:14 UTC, Secure CPU
`cpu3g` 16 vCPU / 64 GB, EU-RO-1, **$0.64/hr**. Python 3.10 venv: whestbench 0.16.1, flopscope 0.12.3, numpy 2.2.6. Bundle =
`local_sim/`, `tools/`, cands A1s e823d62f, 8be 8be40031 (same-host control), A1 7d381fc5 (reference). Gate = `linux_memory_gate.py`
RLIMIT_AS 8,192 MB, 100 predicts, f64 read-only, 2 BLAS threads, `--reset-accumulator`, seed 90000000101 (verify_nominee's arguments).
Started 07:20 UTC, three in parallel. Expected on native glibc: padded depth-6 builds grow ≈ +130…+200 MB / 100 (8be control
+204 MB on L3's host), which is host behaviour — the reading is A1s vs 8be on the same host, plus VmPeak ≪ 8 GiB.
FRESH-A100 A1s vs A1: local workerq #3657 (CPU only, no WSL; A1's predictions cached) — `fresh_a_a1s.json`.

**FRESH-A100 A1s vs A1 (workerq #3657, `fresh_a_a1s.json`): raw 1.547531e-8 vs A1 (cached), Δ −1.4117e-13 = −0.009 %,
t95 [−6.69e-13, +3.86e-13], 50/100 wins ⇒ accuracy = A1's (planner reorder only). PASS.** (ev.py exits 2 on the cost/memory gates,
which `--no-cost` skips by design — as for L2's FRESH runs.)

### A3s viability (D4's A-safe preference) — planner bill of A3's step-276 export (workerq #3659, `planner_sweep_a3.json`)

Real 0.12.3 meter, DEV0, built from L2's `cand_a3s276_lam06.py` 7b3d722b (only `_SW_OP_FLOPS` changes; output max |Δ| vs it ≤ 9.4e-6).
Headroom = ops × 14.26 µs/op (today's node, #337470) × worst tail 1.196 (pess. × 1.029). Score vs A1 ≈ VAL Δraw −0.918 % + ΔF.

| λ_op | F | ΔF vs A1 | ops | worst-day residual | headroom central / pess. | score vs A1 (approx.) |
|---|---:|---:|---:|---:|---|---:|
| 6e5 (L2 bake 7b3d722b) | 287,565,166,545 | +0.50 % | 23,374 | 0.399 s | +0.3 % / −2.5 % | ≈ −0.42 % |
| 2e6 | 290,900,385,362 | +1.67 % | 21,693 | 0.370 s | +7.5 % / +4.8 % | ≈ +0.75 % |
| **3e6 = D4 "A3s"** | 292,381,110,759 | +2.19 % | 21,371 | 0.365 s | **+8.9 % / +6.2 %** | ≈ +1.27 % |
| 5e6 | 298,479,981,506 | +4.32 % | 20,069 | 0.342 s | +14.4 % / +12.0 % | ≈ +3.4 % |
| *A1s (graded)* | 290,938,839,223 | +1.68 % | 20,630 | 0.348 s | +13.1 % / +10.6 % | **+1.70 %** |

**Reading.** NEW10 adds +741 ops, so D4's A3s (λ 3e6) misses the ≥ 10 % rule (+8.9 % central). The A3 bake that clears it (λ ≈ 4–5e6)
costs more F than A3's −0.918 % raw buys back, so at step 276 **A1s dominates every rule-passing A3s** (1.70 % vs ≈ 3.4 % behind A1).
Break-even: a λ-5e6 A3s beats A1s only if A3's raw gain reaches ≈ −2.6 % vs A1. Strategist's call (D4 names λ 3e6 explicitly).
Caveat: A3's own per-op rate is unmeasured on the grader; NEW10's extra ops are small elementwise ops, and if they run cheaper than the
A1-chain average the headroom column is conservative. A canary of A3s would settle it.

### A3s at step 368 (L2 A3 = 5ddf3290, VAL −1.418 % raw vs A1, 85/100) — fine λ_op sweep (workerq #3666, `planner_sweep_a3s368.json`)

Bills are shape-determined, so step 368 = step 276 at λ 3e6 (292,381,110,759 F / 21,371 ops) as expected. Score vs A1 ≈ VAL Δraw + ΔF.

| λ_op | F | ΔF vs A1 | ops | worst-day residual | headroom central / pess. | score vs A1 | vs A1s (graded +1.70 %) |
|---|---:|---:|---:|---:|---|---:|---:|
| 6e5 (L2 A3) | 287,565,166,545 | +0.50 % | 23,374 | 0.399 s | +0.3 % / −2.5 % | −0.91 % | — (fails) |
| 3e6 | 292,381,110,759 | +2.19 % | 21,371 | 0.365 s | +8.9 % / +6.2 % | +0.77 % | −0.93 % (fails rule) |
| **3.5e6** | 293,790,066,123 | +2.68 % | 20,991 | 0.358 s | **+10.5 % / +7.9 %** | +1.26 % | **−0.44 %** |
| **4e6** | 294,555,261,193 | +2.95 % | 20,817 | 0.355 s | **+11.2 % / +8.7 %** | +1.53 % | **−0.17 %** |
| 4.5e6 | 296,782,224,138 | +3.73 % | 20,409 | 0.348 s | +13.0 % / +10.5 % | +2.31 % | +0.61 % |

**Reading for the strategist.** A rule-passing A3s exists at λ_op 3.5e6–4e6: it beats A1s by 0.2–0.4 % (VAL-based, ±≈ 0.2 %) and
gives up 2–3 points of headroom (passes central, not pessimistic; A1s passes both). At equal headroom (λ 4.5e6) A1s is better. A3's own
served per-op rate is unmeasured; if L2 canaries A3 (λ 6e5) its served residual gives it directly and fixes this column. If the A3s
route is chosen, the candidate is `cand_a3s368_lam06.py` with `_SW_OP_FLOPS = 3500000` (or 4e6); it still needs bake + verify +
canary + memory (the pod's provisional A3 gate is shape-only but at λ 3e6 — a different planner, so it does not carry over).

### 100-call memory gate on pod v0fza2b41jnbyw (native Linux, RLIMIT_AS 8,192 MB; raw reports `pod_mem/`)

All controls (negative/positive/lifecycle) PASS; 100/100 predicts completed, finite, FLOPs identical on every call. The gate's printed
"settled growth" is first→last and FAILs every build including the 8be control: call 0 → 1 is a +300…+530 MB allocator warm-up
(GOTCHAS item 9). verify_nominee's criterion (call 10 → last ≤ 64 MiB) and block medians of the per-call settled VmSize:

| build | sha | VmPeak | settled call 10 → 99 | block medians 10–30 / 30–55 / 55–80 / 80–100 (MiB) | verdict |
|---|---|---:|---:|---|---|
| 8be (control, same host) | 8be40031 | 4,555.4 MB | +1.0 MiB | 2223 / 2204 / 2204 / 2222 | PASS |
| A1 (reference) | 7d381fc5 | 4,588.3 MB | +276.8 MiB (one late allocator step) | 2099 / 2141 / 2145 / 2134 | PASS on medians (+35) |
| **A1s** | **e823d62f** | **4,542.6 MB** | **+33.7 MiB** | 2161 / 2148 / 2127 / 2205 | **PASS** |
| A3s λ 3e6 (provisional, from A3 s276) | 5966e286 | 4,648.8 MB | −8.0 MiB | 2280 / 2251 / 2292 / 2253 | PASS (shape-only ⇒ valid for any A3 step at λ 3e6) |

No build trends: settled VmSize plateaus at 2.1–2.3 GB after the first call; VmPeak 1.76–1.80× under the cap. **A1s: every nominee
check now PASS** (graded #337481 0/100, verify #3652, FRESH-A100 #3657, 100-call memory). A3s λ 3.5e6 (7e1e1a2e) gate running.

### A3 graded (L2 canary #337495, λ 6e5, 5ddf3290): 1.8527052525413123e-9 = A1 × 0.9931 — A3s re-anchored on it

Served residual median 0.3331 / max 0.3447 s at 23,374 ops = **14.25 µs/op** (A1-chain builds 14.26 today) ⇒ the A3s headroom column
above stands. Projected A3s = A3 graded × exact F ratio (raw identical under a planner change):

| A3s λ_op | ops | projected score | vs A1s (1.8974e-9 graded) | headroom central / pess. |
|---|---:|---:|---:|---|
| 3e6 | 21,371 | 1.8837e-9 | −0.72 % | +8.9 % / +6.3 % (fails) |
| 3.5e6 | 20,991 | 1.8928e-9 | **−0.24 %** | +10.5 % / +8.0 % |
| 4e6 | 20,817 | 1.8977e-9 | +0.02 % (tie) | +11.3 % / +8.7 % |
| 4.5e6 | 20,409 | 1.9121e-9 | +0.77 % | +13.0 % / +10.5 % |
| *A1s* | 20,630 | *1.8974e-9 graded* | — | **+13.1 % / +10.6 %** |

**L1 recommendation (strategist decides): A-safe = A1s.** The best rule-passing A3s gains 0.24 % for 2.6 points less headroom and is
not yet baked, canaried or memory-gated at its own λ (3.5e6 gate running on the pod); at equal headroom A1s is 0.77 % better.

**A3s λ 3.5e6 (7e1e1a2e = A3 s368 5ddf3290 + `_SW_OP_FLOPS` 3.5e6) memory: PASS** — VmPeak 4,593.1 MB, settled call 10 → 99 +1.0 MiB,
block medians 1952 / 1984 / 1948 / 1952 MiB, 100/100, 0 errors (`pod_mem/A3s368_l35.json`). Pod v0fza2b41jnbyw **stopped 09:50 UTC**
(get-pod: EXITED; uptime 9,097 s × $0.64/hr ≈ **$1.62**); restartable for B3.

### A4s (L6) inherits the A3s λ 3.5e6 memory PASS

`experiments/2026-10-21_L6/src/cand_a4s368_lam35.py` (7b0d6800) vs the gated A3s λ 3.5e6 build (7e1e1a2e): same 4,264 lines; 35 differ,
**all `_NAME = …` fitted-constant assignments (0 code lines, no `_SW_*` planner line)**. Memory is shape-only, so the pod PASS (VmPeak
4,593.1 MB, settled call 10→99 +1.0 MiB) applies to A4s. Expected bill = A3s λ 3.5e6: 293,790,066,123 F / 20,991 ops (L6 verify #3669
confirms) ⇒ worst-day residual +10.5 % / +8.0 %. Projection: A3 graded × (1 − 0.006 VAL) × F ratio 1.02165 ≈ **1.881e-9 ≈ A1s × 0.992**.

### A4s graded (L6 canary #337511, 7b0d6800): 1.8800496320794433e-9 — worst-day margin on its own telemetry

Forecast 1.8819e-9 (−0.10 % ✔); **A1s × 0.9909**, A1 × 1.0077, B × 0.9398; 0/100 failed, smoke PASS; F 293,789,651,431 (= bill − 414,692).
Served residual median 0.2900 / max 0.3115 s at 20,991 ops = **13.82 µs/op** (below the 14.25 the A3s projection assumed); wall 71.4 /
80.2 s. **Worst day: residual 0.347 s central (+13.3 %), 0.357 s pess. (+10.8 %) — passes both, like A1s (+13.1 / +10.6 %); wall 88.7 s
(+26 %); first fails at 1.284× (A1s 1.323×).** Worst-of-N: N = 1,000 0.331 s, N = 10,000 0.343 s (A1s 0.331 / 0.343).
Memory: PASS by identity with the gated A3s λ 3.5e6 build (35 fitted constants differ, no code) + L6 rehearsal 100/100, VmPeak 3,673 MiB.
**L1 recommendation (strategist decides): A-safe = A4s** — 0.91 % better than A1s at the same headroom; A1s is its fallback.

## §7 D6 (16:00 UTC) — final nominee tables, forum 18264

- **`docs/DESIGNATION.md` § Final nominee tables**: side-by-side field table for **A = A4s** (#337511, 7b0d6800, package c0fdf5f2)
  and **B** (#337462, 7a146f84, package bfba72f7), both complete, nothing owed. A4s's margin is recomputed from its per-MLP telemetry with
  `margin.json`'s constants: residual 0.2900 / 0.3115 s, worst day 0.347 / 0.357 s (**+13.3 % / +10.8 %**), wall 88.7 / 91.4 s
  (+26 / +24 %), first fail 1.284× (wall 1.496×), 1.121× on top of the pessimistic worst day (B 1.253×). A4s = B × 0.9398. Package
  inner estimator re-hashed = source. The A5s row is a placeholder until L7's canary grades (≤ 20:30 UTC).
- **Forum "#337475"** is abhimanyu_singh's submission (2.966e-9); the write-up is discourse topic **18264** (dup 18261). It goes into
  `docs/WRITEUP_LEDGER_2026-10-21.md` §2b as external rows E1a–f: it independently supports N26 (sketching dead) and N7a (error
  transported at gain 1 ⇒ truncation bias does not attenuate). Nothing in it is buildable today. Submission page props were read into the
  scratchpad only, not `experiments/run_reports/`.
