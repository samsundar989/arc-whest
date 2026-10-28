# L7 — A5: A4's export + one more fresh-data epoch (arm a: local 5090)

Handoff: `_claude_snap/handoff_2026-10-19/HANDOFF_2026-10-21_L7_A5_FRESH_DATA_ARMS.md`. Started 15:52 UTC Oct 21.
This file covers **arm a** (L6's session). Arms b/c (pods) and soup(a,b) belong to the pod session; their tables go here when copied back.

## 0. Prediction and keep rule (written before any A5 number)

- Prediction (handoff): arm a **−0.5 % raw** VAL vs A4's export (80 %: −0.2 … −0.8 %), 0 FLOPs.
- Keep (best of a / b / c / soup by VAL-100 paired vs A4's export): ≤ −0.3 % raw, ≥ 60/100, 95 % CI excl. 0; then FRESH-A100 vs A4
  ≤ −0.25 % (CI excl. 0); bake `_SW_OP_FLOPS = 3500000` (bill 293,790,066,123 F / 20,991 ops asserted, 0 code lines differ);
  verifier; 100-fresh-seed sweep; canary ≤ 20:30 UTC with the forecast written first. Graded inside ±1.5 % ⇒ A = A5s, else A = A4s.
  Bake/gates/canary are L2's.

## 1. Corpus d8bext3 (arm a)

- `build_d8bext3.py` = L6's `build_d8bext2.py` recipe (re-proved array-equal on d8b shard00), shards **shard_ext03/04 rows 0..1471**
  (HF `keenanpepper/arc-whestbench-p2-d8b-corpus-14k@cc3405a9`, both `n_done = 1500`, `w_seed0 = 770000`, N = 1e8).
  idx = 2051 + 8k and 2052 + 8k, k < 1472 (2051 … 13820); asserted disjoint from FRESH-D and from d8bext2.
- `cache/d8bext3/d8bext3_mean.npy` (2944, 16, 1024) f32, sha256 **1e9711b28974df39…** (full hash + shard shas in `d8bext3_idx.json`).

## 2. Arm a run

`a5_train.py` = L6's `a4_train.py` with only the corpus generalised (`--corpus d8bext3`): roster = d8bext3 shuffled then full 0..899
shuffled (seed 20260925); 368 × 8 = 2,944 ⇒ **every update is a fresh net**, TRAIN900 omitted. Corrector LR ×10, cosine 368,
VAL 900–999 at 0/92/184/276/368, earliest strict best, step 0 = fallback. Parent = A4 s368 export `src/a4_export_s368.py`
sha **67ed6090…**; `--parent-val src/a4_val_step368.npz` (b93fcf18…) asserts VAL0 byte-identical.

workerq **#3674** queued 15:55 UTC (16 GiB RAM / 8 GiB VRAM / 3 CPU, as A4's #3660), output `a5a_run/`.
**Started 15:55 UTC. VAL0 = 1.4832448827e-08 = A4 s368, predictions and per-net MSE byte-identical (asserted in-run).**
Generator parity proof PASS. First 6 updates 30–40 s (A4's fresh-net phase: 32 s → 24 s after update 46); projected step 368
≈ 18:45 UTC ⇒ **368 kept** (the 27 s/update cut rule exists to make 19:15; step 276's export ≈ 18:05 is the fallback).

**Label check (pod vo69njqvq5qr8w, RTX 5090 Secure, 15:56–16:00 UTC, stopped)** — `labelcheck/mc_check.py` (L6's), our N = 1e8 MC vs corpus m1:

| net | role | weight sha256 | mean \|Δm1\| | max \|Δm1\| | layer-16 MSE |
|---|---|---|---|---|---|
| 2051 | ext03 row 0 | 2802e5fa… | 4.64e-5 | 5.5e-4 | 1.85e-9 |
| 2052 | ext04 row 0 | bcd620d4… | 4.55e-5 | 4.3e-4 | 2.01e-9 |

Gate mean |Δm1| ≤ 2e-4: **PASS ×2** — same agreement as L6's d8b control (4.25e-5). (L6's first pod's host had no free GPU, so a new pod was used.)

**16:31 UTC:** update 51; updates 18–47 ran 37–67 s while other lanes' jobs #3678–3680 shared the box, now back to 32 s.
Projected step 368 ≈ 19:00–19:15 UTC; no restart (a 276-update restart would finish no earlier). Step 276's export (≈ 18:25) is the fallback.

## 3. Arms b and c — DUPLICATE run by L4's session (16:05–17:02 UTC), stopped; NOT the arms of record
**Superseded:** a fork of L6's session ran the same arms b/c on pod cbets7hsyjiodl (5090, torch 2.13, VAL0 byte-identical; `pod_runs/NOTES.md`, commit 9851f1c), launched 16:33 — the arms of record. L4 found that commit at 17:01 and stopped its pod 14uah7xtcpq161 (uptime 57 min ≈ **$2.36**; arm c's pod j9urjym2zvazh4 ≈ $0.05, stopped at creation). Its partial visits are kept in `pod_runs/b/` and `pod_runs/armc/` as an independent cross-backend replicate only (same d8bext4 sha 82800023…):

| run (pod torch 2.8) | step | Δraw vs A4 recorded VAL | wins/100 | 95 % CI |
|---|---|---|---|---|
| b (×10) | 0 | −0.004 % | 56 | [−0.012, +0.004] % |
| b | 92 | +1.282 % | 29 | [+0.836, +1.729] % |
| b | 184 | +0.649 % | 40 | [+0.294, +1.004] % |
| c (×3) | 92 | +1.361 % | 27 | [+0.892, +1.830] % |

Setup of the duplicate, for the record:
- **Pod:** no RTX 5090 stock anywhere at 16:05 (community and secure create refused; zrdhrgb88enc0w start refused). One RTX PRO 6000
  Blackwell (sm_120, same arch as the 5090) `14uah7xtcpq161`, $2.49/h — the listing said $1.69; at the real price two pods would
  overrun $12, so the second (j9urjym2zvazh4) was stopped within a minute and **both arms run on the one pod** (96 GB; 10 s/update alone,
  17 s/update with both).
- **Trainer:** `pod/a5_pod_train.py` = `a5_train.py` with only the layout (l3_train's pod-native bundle) and the weight source changed:
  fresh-net weights from a persistent pod-image torch 2.8.0 worker (= the WSL generator's torch), all 7 generator fixtures checked at
  start (d8b 0/1/2/777/2047 R3 prefixes + ext 2049/2050 full sha) — PASS. Arm c's ×3: the pair adapter admits 1/10 only, so the ×10
  optimizer is built and the corr/ and hub/ groups rescaled by 3/10 (the adapter's own `rate *= mult`).
- **Runtime:** pod torch 2.8.0+cu128 / numpy 2.1.2 (PyPI unreachable from the host; torch 2.13+cu130's nccl pin not obtainable offline).
  **VAL0 not byte-identical:** A4 export VAL0 = 1.483189e-8 vs recorded 1.483245e-8 (paired −0.004 %, 56/100, max |Δpred| 2.1e-6) — the
  cross-torch f32 floor (J: +0.002 %). Rows are paired vs A4's recorded VAL and vs the pod's own VAL0; the soup/pick must re-VAL on one backend.
- **Corpora:** arm b = `cache/d8bext4` = ext05/06 rows 0..1471 (2,944 nets; `build_d8bext3.py d8bext4 shard_ext05,shard_ext06`, recipe
  re-proved on d8b shard00; sha256 82800023eb8fef56…; disjoint from d8bext3). Arm c = d8bext3, rebuilt on the pod from HF cc3405a9 —
  mean sha256 1e9711b2… and idx list identical to arm a's.
- Launch: arm b 16:16 UTC (×10), arm c 16:33 UTC (×3; a first launch at 16:31 died on the adapter's 1/10 assert).

### Arm a VAL table (paired vs A4 s368; `python val_table.py a5a_run src/a4_val_step368.json`)

| step | mean raw | mean ratio | paired Δraw vs A4 | wins/100 | 95 % CI |
|---|---|---|---|---|---|
| 0 | 1.483245e-08 | 1.00000 | +0.000 % | — | — |
| 92 | 1.505271e-08 | 1.01485 | +1.496 % | 24 | [+1.054, +1.938] % |
| 184 | 1.498180e-08 | 1.01007 | +1.042 % | 30 | [+0.621, +1.463] % |
| 276 | 1.485219e-08 | 1.00133 | +0.134 % | 46 | [−0.195, +0.463] % |
| **368** | **1.476130e-08** | **0.99520** | **−0.491 %** | **73** | **[−0.667, −0.315] %** |

Step 92 (16:50 UTC) = the peak-LR early-harm phase, about half A4's (+2.77 % at its 92); A4 recovered to −0.60 % by 368.

## L2 — A5s bake / gates / canary (L2's session; runbook `bake/RUNBOOK.md`, kit commit 529f78f)

**Speculative start 18:20 UTC on the current VAL leader, before the 19:20 pick** (selection stays VAL-only: if a, b or soup wins
at 19:20 it is rebaked and re-gated; bills are identical by construction, so only FRESH/parity/sweep repeat).
Arms of record b/c (`pod_runs/cb/*`, VAL0 byte-identical), paired vs A4 s368 (`val_table.py`):
b s368 **−0.379 %** 68/100 [−0.557, −0.202]; c s368 **−0.400 %** 67/100 [−0.548, −0.252] — both pass the VAL keep rule.

- Bake c s368: export d9a33043… → `src/cand_a5s_c368_207f6245.py` sha **207f6245…**; λ lines 1; **0 code lines differ from A4s**,
  35 fitted tables; `whest validate` PASS (commit 7397762). Dry run of the same script on A4's export reproduces A4s 7b0d6800 byte-exact.
- Gates (CPU-only, beside arm a's trainer): FRESH-A100 paired vs A4s **#3682**; `verify_nominee` (bill asserted 293,790,066,123 F /
  20,991 ops) **#3683**; parity VAL 900/901 **#3684** (floor: A4s vs its trainer 4.8e-6 / −0.20 %, `bake/parity_a4s_floor.json`);
  100-fresh-seed sweep on pod **q6xinsaucdy8u7** (cpu3g 16 vCPU $0.64/h, EU-RO-1, up 18:21 UTC; L6's seeds/blocks/harness).
- Slots: ledger 7/10 at 17:39 UTC (L3b's B4 #337562) ⇒ an A5s canary falls in Samuel's 3 reserved slots — the guard's ask goes to him.

### Pick (18:36 UTC, directive: soup(b,c), not soup(a,b); do not wait for arm a unless it beats the pick by 19:15)

soup(b,c) = `soup.py src/a4_export_s368.py cb/b/estimator_step368.py cb/c/estimator_step368.py` → `soup_bc/soup_bc368.py` sha
**c24b7f38…** (35 tables averaged). VAL-100 on the arms' own backend (pod cbets7hsyjiodl, torch 2.13, `val_only.py`; that backend's
VAL0 is byte-identical to A4's recorded VAL), paired vs A4 s368 (`soup_bc/pick_table.py`, `pick_table.json`):

| run | mean raw | paired Δraw vs A4 | wins/100 | 95 % CI |
|---|---|---|---|---|
| b s368 | 1.477498e-08 | −0.379 % | 68 | [−0.557, −0.202] % |
| c s368 | 1.477388e-08 | −0.400 % | 67 | [−0.548, −0.252] % |
| **soup(b,c) s368** | **1.476517e-08** | **−0.454 %** | **74** | **[−0.599, −0.309] %** |

soup vs b −0.072 % (55/100, [−0.148, +0.003]); soup vs c −0.053 % (54/100, [−0.127, +0.021]). **PICK = soup(b,c)** (lowest VAL; keep
rule ≤ −0.3 %, ≥ 60/100, CI excl. 0: PASS). Prediction was −0.6 … −0.9 % for soup(a,b); soup(b,c) gave −0.45 %, a small step over its parts.
Arm a at 18:38: step 315, +0.13 % at 276 and slowed to ~60 s/update under the gate jobs' CPU load ⇒ not expected to beat −0.454 % by 19:15.

**A5s = `src/cand_a5s_soupbc_180e488b.py` sha 180e488b…** (λ lines 1, 0 code lines vs A4s, validate PASS; commit 457d0d7).
The arm-c speculative gates were cancelled (#3682 FRESH, #3683 verify after rig PASS, #3684 parity; c's partial sweep kept on the pod
as `runs_c368_aborted/`, unused). Soup gates: FRESH **#3688**, verify **#3689**, parity **#3690**, sweep relaunched 18:38 UTC on pod
q6xinsaucdy8u7 (sha 180e488b checked on the pod).

## 4. PICK (18:46 UTC) — soup(b,c) step 368 → A5s candidate `cand_a5s_soupbc368_180e488b.py`

Samuel (18:34 UTC): form soup(b,c) now, pick the best of b / c / soup(b,c) by VAL, do not wait for arm a unless it beats the pick by 19:15.
Soup = `soup.py` elementwise mean of the 35 fitted tables (incl. the `_CORR_B64` float32 corrector blobs, f64 mean → f32) of
`pod_runs/cb/b/estimator_step368.py` (00bca58e) and `pod_runs/cb/c/estimator_step368.py` (d9a33043) → `soup/soup_bc368.py`
sha **c24b7f38…**; VAL on pod cbets7hsyjiodl (torch 2.13+cu130, the backend whose VAL0 is byte-identical to A4's), `pod_runs/soup_bc368/`.

| candidate | mean raw | ratio of means vs A4 | paired Δraw vs A4 | wins/100 | 95 % CI |
|---|---|---|---|---|---|
| b s368 (d8bext4 = ext05/06, ×10) | 1.477498e-08 | −0.387 % | −0.379 % | 68 | [−0.557, −0.202] % |
| c s368 (d8bext3 = ext03/04, ×3) | 1.477388e-08 | −0.395 % | −0.400 % | 67 | [−0.548, −0.252] % |
| **soup(b,c) s368** | **1.476517e-08** | **−0.454 %** | **−0.454 %** | **74** | **[−0.599, −0.309] %** |

Soup vs b: −0.072 % (55/100, [−0.148, +0.003]); soup vs c: −0.053 % (54/100, [−0.127, +0.021]) — the soup is the VAL pick but its edge
over either arm alone is inside noise. **Gate 1 PASS** (≤ −0.3 %, ≥ 60/100, CI excl. 0). Arm a (step 276 +0.13 %) is not waited for.

**Bake** (L2's `bake/bake_a5s.py`, 18:46): `src/cand_a5s_soupbc368_180e488b.py` sha **180e488bfb2e6587…**, λ lines 1
(`_SW_OP_FLOPS = 3500000`), **0 code lines differ vs A4s 7b0d6800**, 35 fitted tables differ; release `release_a5s_soupbc368/`;
`whest validate` PASS. **L2 had independently baked the same file** (`src/cand_a5s_soupbc_180e488b.py`, 457d0d7 — byte-identical; its
soup VAL npz cafe5224 = `pod_runs/soup_bc368/val.npz`) and runs the gates since 18:38: FRESH-A100 vs A4s #3688, verify #3689, parity #3690,
100-seed sweep on pod q6xinsaucdy8u7 (see "L2 / Pick"). The c368 gates were cancelled, so c368 207f6245 is **not** a gated fallback
(≈ 25 min of its own gates if the soup fails one). Ledger 7/10 ⇒ the canary falls in Samuel's reserved slots and needs his approval at the guard.

Pod cbets7hsyjiodl (5090 Community $0.69/h) stopped 18:46 UTC, uptime 8,819 s ≈ **$1.69**. L7 pod spend ≈ $1.69 + $0.30 (two no-internet
5090s terminated) + $0.08 (label check) ≈ **$2.07** of $12.

### Canary forecast A5s (written 19:01 UTC, before submission; Samuel approved the canary into a reserved slot)

Package `a5s-soupbc368-lam35-180e488b.tar.gz` sha256 **2ee01ae9…** (estimator inside = 180e488b; validate-package PASS).
**Forecast 1.8715e-9** (Samuel's figure = A4s graded 1.8800496e-9 × VAL raw ratio 0.99546, bills identical by assertion).
Keep window ±1.5 %: **1.8434 – 1.8996e-9**. Rule: graded inside the window **and** FRESH-A100 vs A4s ≤ −0.25 % with CI excl. 0 ⇒ **A = A5s**;
else A stays A4s. Expect 0/100 failed, served F 293,789,651,431 per MLP (= A4s).
Gates at submission: sweep **PASS** (100/100, 0 errors, 293,789,618,663 F every call = A4s, VmPeak 3,651 MiB; `bake/summary.json`);
verify #3689 (static/rig PASS, meter/project running); FRESH #3688 running; parity #3690 queued.

## 5. Arm a step 368 (landed 19:16:01 UTC) — after the cut-off; does not change the pick

#3674 complete 19:16 UTC (best = step 368, earliest strict best; peak CUDA 3.27 GB). **Gate 1 PASS on its own: −0.491 %, 73/100,
CI [−0.667, −0.315] %** — on the handoff prediction (−0.5 %). Arm a was the best single arm, but it landed after Samuel's 19:15 cut-off,
and paired head-to-head it is not separable from the pick:

| arm a s368 vs | paired Δraw | wins/100 | 95 % CI |
|---|---|---|---|
| soup(b,c) s368 (the canary, #337601) | −0.037 % | 49 | [−0.140, +0.065] % |
| c s368 (same corpus, LR ×3) | −0.092 % | 59 | [−0.171, −0.012] % |
| b s368 (other corpus, LR ×10) | −0.109 % | 54 | [−0.269, +0.052] % |

So LR ×10 beat ×3 on the same 2,944 nets by 0.09 % (CI excl. 0), and every fresh-data epoch so far landed at −0.4…−0.6 %.
Diagnostic for the write-up / post-deadline A6: VAL-only soup(a,b,c) (`soup/soup_abc368.py` 5270942f, #3701) and soup(a,b)
(`soup/soup_ab368.py` 534fd278, #3702) on the local real chain (`a5_train.py --updates 0`).

**Incident (19:18 UTC, repaired):** while generalising `soup.py` to N inputs, a failed edit left the 2-input version in place and a
4-argument call wrote soup(a,b) over `pod_runs/cb/c/estimator_step368.py`. Restored from git (561319b) within a minute, sha d9a33043
re-checked; soup(b,c) c24b7f38, the bake 180e488b and every gate had used the earlier file and are unaffected. `soup.py` now refuses
an existing output path.

### A5s gates (all on 180e488b; posted after submission, as directed)

| gate | job | result |
|---|---|---|
| 100-fresh-seed sweep | pod q6xinsaucdy8u7 | **PASS** 100/100, 0 errors/non-finite/shape faults, **293,789,618,663 F every call** (= A4s), 21,634 dispatches, VmPeak 3,651 MiB (A4s 3,673), mint parity PASS; `bake/summary.json`, raw `bake/runs/` |
| verify_nominee | #3689 | **VERDICT PASS** — bill **293,790,066,123 F (+0) / 20,991 ops**, determinism 2 proc × 3 nets bit-identical, client-only import, served-equivalent residual 0.3200 s, wall 81.9 s; WARN DESIGNATION listing + A1's `_pkk_fast_ok` literal (as A4s) |
| FRESH-A100 vs A4s | #3688 (exit 2 = `--no-cost` "not accepted", as before) | **PASS** A5s raw 1.505352e-8 vs A4s 1.513606e-8, mean ratio 0.994547, paired **−0.531 % [−0.665, −0.397]**, **77/100** (gate ≤ −0.25 %, CI excl. 0) — agrees with VAL (−0.454 %) |
| parity VAL 900/901 | #3690 | **PASS** final-layer max\|Δ\| 6.9e-6 / 6.0e-6, rel MSE −0.02 % / −0.18 % vs the soup's trainer VAL (floor A4s 4.8e-6 / −0.20 %) |

FRESH-based cross-check of the forecast: 1.8800496e-9 × 0.994547 = **1.8698e-9** (−0.09 % from the 1.8715e-9 forecast).
Canary **#337601** submitted 19:02:14Z (Samuel-approved reserved slot; ledger 8/10). Pod spend: sweep pod ≈ $0.42, test pod ≈ $0.01.

### Soup diagnostics (19:38 UTC; local real chain, `a5_train.py --updates 0`, #3701/#3702) — write-up / post-deadline only

| soup (s368 exports) | mean raw | vs A4 | wins | 95 % CI | vs soup(b,c) (canary) | wins | 95 % CI | vs arm a | 95 % CI |
|---|---|---|---|---|---|---|---|---|---|
| soup(a,b,c) 5270942f | 1.475947e-08 | −0.497 % | 76 | [−0.647, −0.347] | −0.044 % | 58 | [−0.080, −0.008] | −0.005 % | [−0.074, +0.064] |
| soup(a,b) 534fd278 | 1.475692e-08 | −0.513 % | 75 | [−0.670, −0.356] | −0.060 % | 63 | [−0.100, −0.020] | −0.021 % | [−0.102, +0.060] |

Soups including arm a beat the canary's soup(b,c) by 0.04–0.06 % (CIs excl. 0) but are not separable from arm a alone. Every
combination of the three fresh-data epochs sits at −0.45…−0.51 % vs A4: the gain is the fresh data, and averaging adds ≤ 0.06 %.
0.06 % raw is ~1/25 of the canary's ±1.5 % keep window, so it does **not** justify one of Samuel's two remaining slots; the
post-deadline A6 build should start from soup(a,b) or arm a and spend the remaining ≈ 4,000 unused ext nets (ext00 ≥ 100,
ext01/02 ≥ 1024, ext03–06 rows ≥ 1472, ext07).

### Canary A5s — AIcrowd #337601 GRADED (completed 19:59:30Z)

**GRADED 1.8751280753233397e-9** (raw 1.403538e-8, multiplier 0.13360007, 0/100 failed; `experiments/run_reports/a5s_L2_337601_graded_report.json`).
vs forecast 1.8715e-9: **+0.19 %, inside ±1.5 %** (window 1.8434–1.8996e-9). FRESH vs A4s −0.531 % [−0.665, −0.397] (CI excl. 0).
**⇒ by Samuel's rule, A = A5s** (nomination itself is Samuel's).
vs A4s #337511 (1.8800496e-9): **−0.262 %** score = raw −0.262 % at the same served F (293,789,651,431). FRESH predicted −0.53 %, VAL −0.45 %;
the public panel's single draw gave half of that. (A partial read at 19:40, 20/100 MLPs, showed 1.837e-9 — not a result.)

| graded run | residual med / max | wall med / max | µs/op | worst-day residual (central / pess.) | headroom | first fail |
|---|---|---|---|---|---|---|
| **A5s #337601** | 0.2913 / 0.3017 s | 71.3 / 78.5 s | 13.88 | 0.349 / 0.359 s | **+12.9 % / +10.4 %** | **1.326×** (pess. day 1.116×) |
| A4s #337511 | 0.2900 / 0.3115 s | 71.4 / 80.2 s | 13.82 | 0.347 / 0.357 s | +13.3 % / +10.8 % | 1.284× (1.121×) |

(L1 `nominee_margin.py <report> 20991`; its A4s readout reproduces L1's committed table.) ≥ 10 % worst-day rule: PASS both ways.
Memory: PASS by identity with A4s (0 code lines differ; fitted constants only) + the 100-seed RLIMIT_AS sweep (VmPeak 3,651 MiB).

## 6. Outcome (L6/L7 session, 20:0x UTC)

- **A5s = soup(b,c) s368 (180e488b), canary #337601 GRADED 1.8751281e-9** (forecast 1.8715e-9, +0.19 %, inside ±1.5 %); vs A4s −0.262 % at the
  same served F; FRESH −0.531 % ⇒ **A = A5s** by the keep rule (L2's rows: §L2, `docs/DESIGNATION.md`, `RESULTS.md` 3594d83). A4s is the fallback A.
- The fresh-data law held for the third epoch running: one pass over ≈ 2,944 new corpus nets = −0.38 … −0.49 % VAL per arm; soups add ≤ 0.06 %.
- Arm a (#3674, local): s368 −0.491 % vs A4 (73/100), best single arm, after the cut-off; soup(a,b) −0.513 % is the best VAL measured and the
  natural parent for a post-deadline A6.
- Spend this lane: label-check pod ≈ $0.08 + arms pod cbets7hsyjiodl ≈ $1.69 + two no-internet pods ≈ $0.30 ≈ **$2.07** (all stopped/terminated).
