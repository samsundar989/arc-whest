# L3b — Oct 21: B4 (B on A4's chain) and the cut-and-refit arm

Handoff: `_claude_snap/handoff_2026-10-19/HANDOFF_2026-10-21_L3_B4_AND_CUT_REFIT.md`. No nomination or designation made (Samuel's).

## A. B4 — nominee B on A4's chain

**Build.** `src/estB4_a4s368.py` **4af12b5e** = A4's selected export `experiments/2026-10-21_L6/src/a4_export_s368.py` (67ed6090)
+ B's exact 3 planner lines (`_SW_CAP` 6→5, `_SW_OP_FLOPS` 1e6→5e6, `_SW_PAD` True→False; the same 3-line diff as A1 7d381fc5 →
B 7a146f84). Nothing else differs from A4s 7b0d6800 except its single λ line (A4s = 3.5e6, pad on, cap 6), so the fitted tables are
A4s's byte-for-byte. Release `release_b4/estimator.py` = source; package `b4-a4s368-4af12b5e.tar.gz` sha256 **882541f0**
(`whest validate` + `validate-package` PASS).

**Bill is not B's.** The handoff assumed F = B's 306,840,769,522 / 18,943 ops. A4's export carries A3's extra
correction terms (A3 bills +0.50 % F over A1 at the same planner), so the real 0.12.3 meter reads
**308,282,626,669 F / 19,684 ops** (VAL900/901 and all 22 parity nets, one value; verifier exact +0). That is +0.470 % F / +741 ops
over B, and +4.933 % F / −1,307 ops vs A4s (293,790,066,123 / 20,991). Served (A3-shape offset −414,692) **308,282,211,977 F**.

| gate | result | evidence |
|---|---|---|
| parity vs A4s, VAL 900–921 (22 nets, real 0.12.3, f64 read-only) | max \|Δpred\| **1.03e-5** (1 net of 22 above 1e-5; final layer max 8.9e-6) — the planner-reorder floor (B vs A1: 1.13e-5 / 1.23e-5) | `parity_bill_val*.json` (#3677–#3679) |
| raw final-layer MSE vs A4s, same 22 nets | **+0.040 %** [−0.026, +0.106] 95 % CI, B4 wins 8/22 — consistent with 0 (B vs A1 on VAL-100: −0.010 %) | same |
| bill assert | 308,282,626,669 F / 19,684 ops (not the handoff's 306.84 GF — see above) | verifier meter +0 |
| `verify_nominee` checkout/imports/static/rig/meter/project | **PASS** (#3680): client-only import, rig 1024×16 / 512×16 / 256×4, 2-process determinism, residual-eq 0.300 s, wall-eq 83.9 s; WARN `_pkk_fast_ok` literal handler (as every A1-family build) | `verify_b4/verify.md` |
| memory | by identity: B's flat planner (cap 5, no pad: native Linux 3,668 MB, −12.1 MB / 100) + A4s's tables (A3s 3.5e6 shape PASS 4,593 MB); plus the RLIMIT_AS rehearsal below | DESIGNATION B / A4s rows |
| **100-fresh-seed rehearsal** (L3 harness unchanged; real flopscope-server 0.12.3 + client-only venv; RLIMIT_AS 8 GiB; 2 BLAS threads; seeds 9,250,000 + [0, 100) = L6's A4s seeds; pod `sgp4emc34jh7n9` cpu3g 16 vCPU, 4 blocks × 25) | **PASS 100/100**: 0 errors, 0 non-finite, 0 shape faults; **one FLOP count 308,282,179,209 = predicted (served − 32,768) to the FLOP**; server VmPeak 2,740 MiB, client 452 MiB; server growth 11.3–22.2 MB/call (flopscope-server per-op bookkeeping, L3 §2.4; A4s 11.7–15.0) | `rehearsal/summary.json`, `rehearsal/runs/` |
| pod residual vs A4s, paired on the same 100 seeds (same pod flavour/layout, different host) | ratio of medians **0.804** (ops ratio 0.938); wall 0.697 | same |

**Residual / wall projection (served).** B's served rate on A1's chain was 13.69 µs/op, A4s's 13.82 ⇒ B4 median 0.270–0.272 s;
worst tail on record ×1.196 ⇒ **0.322–0.325 s** (pre-grade projection; superseded by the graded telemetry in §A2).
Wall: B served 53.5 s median / 61.1 s max at +0.47 % less F ⇒ B4 ≈ 54 / 62 s, worst-day ×1.242 ≈ 77 s ⇒ ≈ +55 % to 120 s.

### A1. Canary forecast (written 16:35 UTC, before submission)

**B4 graded = A4s graded 1.8800496e-9 × served-F ratio 308,282,211,977 / 293,789,651,431 (1.049330) × raw ratio 1.0000 (+0.040 %,
CI incl. 0) = 1.9728e-9** (= B #337462 × 0.9861, −1.39 %). Keep band ±1.5 % ⇒ **1.943–2.002e-9**: graded inside ⇒ nominee B = B4,
else B stays #337462. Served residual max ≈ 0.29 s, wall max ≈ 62 s, 0/100 failed. Slot: 7th of today's 10 (standing approval;
leaves the 3 reserved slots — any later canary today, e.g. A5s, needs Samuel's approval).

### A2. Canary outcome — #337562 GRADED (submitted 16:33:46Z; report `experiments/run_reports/canaryB4_L3b_337562_graded_report.json`)

| | forecast (written first) | graded #337562 | B #337462 | A4s #337511 |
|---|---|---|---|---|
| score | 1.9728e-9 (keep 1.943–2.002e-9) | **1.9724068470701697e-9** (−0.020 %) ✔ | 2.0005580e-9 (B4 = × 0.98593) | 1.8800496e-9 (B4 = × 1.0491) |
| raw (final-layer MSE) | = A4s × 1.0004 | 1.407e-8 | 1.4337e-8 | 1.4072e-8 |
| served FLOPs / MLP | 308,282,211,977 | **308,282,211,977** (to the FLOP, all 100) | 306,840,355,133 | 293,789,651,431 |
| served residual median / max | ≈ 0.27 / 0.29 s | 0.2674 / 0.2819 s (13.59 µs/op) | 0.2594 / 0.2760 s | 0.2900 / 0.3115 s |
| served wall median / max | ≈ 54 / 62 s | 54.8 / 59.6 s | 53.5 / 61.1 s | 71.4 / 80.2 s |
| failed | 0 | **0/100** | 0/100 | 0/100 |

**Worst-day margin** (L1's method and constants, `experiments/2026-10-21_L1/margin.json`: tail ×1.19631, day shift ×1.02872;
headroom = 1 − worst / cap): residual 0.320 / 0.329 s ⇒ **+20.0 % central / +17.7 % pessimistic** (B +22.4 / +20.2, A4s +13.3 / +10.8);
wall 68.0 / 70.2 s ⇒ +43 % / +42 %; first fail at a **1.419×** uniform slowdown of this graded run (B 1.449×, A4s 1.284×);
worst MLP of an N = 1,000 / 10,000 rerun on today's node 0.305 / 0.316 s. ≥ 10 % rule PASS.

**Verdict (handoff rule): graded inside ±1.5 % ⇒ nominee B = B4** (strategist / Samuel confirm; no designation made). B4 buys
−1.41 % score on the B slot for −2.4 pts of worst-day headroom (still ≥ 17.7 %). Trade-off to state plainly: **B4 shares A4s's fitted
tables** (B shared none), so the pair now differs in planner — op count, residual, memory path — but not in accuracy tables.
The failure modes the pair hedges (residual / wall cliff, memory) live in the planner, which still differs; a table-borne error would
hit both. DESIGNATION B4 table + row updated. Spend for §A: sweep pod `sgp4emc34jh7n9` cpu3g 16:02:41–16:21:33 (1,132 s × $0.64/h)
≈ **$0.20**, stopped.

### A3. 100-call native-Linux memory gate on B4 itself (strategist 17:15: finish B4's DESIGNATION row)

The brief asked for memory "by identity with B: fitted constants only". That identity does not hold: B4 vs B is 481 diff lines,
≈ 119 of them code (A3's NEW10 terms carried by A4/A4s); B4 vs A4s differs only in the 3 planner lines, and no gated build has
B4's code × planner combination. So the gate was run on B4, with B as a same-host control.

Pod `s0detmbqch7w8i` (`arc-whest-L3b-B4memgate`, cpu3g 16 vCPU Secure, $0.64/h; native Ubuntu 24.04, Python 3.12, whestbench 0.16.1 /
flopscope 0.12.3 / numpy 2.2.6), `local_sim/linux_memory_gate.py` from HEAD with L1's arguments (RLIMIT_AS 8,192 MB, 100 repeats,
f64 read-only, 2 BLAS threads, `--reset-accumulator`, seed 90000000101, `--lifecycle-control-mib 16`). Raw: `pod_mem/`.

| build | VmPeak | settled call 10 → 99 | block medians 10–30 / 30–55 / 55–80 / 80–100 (MiB) | finite / FLOPs | verdict |
|---|---:|---:|---|---|---|
| B (control, same host) 7a146f84 | 3,853.7 MB | +1.0 MiB | 2005 / 2005 / 2006 / 2006 | 100/100, one value | PASS |
| **B4 4af12b5e** | **3,848.2 MB** | **−30.2 MiB** | 2170 / 2163 / 2159 / 2141 | 100/100, 308,282,626,669 every call | **PASS** |

The gate's printed first → last line "FAILs" both (B4 +364 MB, B +198 MB): that is the call 0 → 10 allocator warm-up (B4 1777 → 2171
MiB) seen on every build including the 8be control (GOTCHAS 9; L1 §6). After warm-up neither grows. Incidents, both mine and harmless
to the reading: (1) first launch used the default `--lifecycle-control-mib 96`, whose deliberate-retention control cannot fit 100×
under 8 GiB ⇒ controls failed before any candidate ran (`*.badctl.json`); (2) the corrected run completed every check, then crashed
in the gate's incidental "same file as repo-root estimator.py" comparison (absent on the pod, as on Oct 20) before writing JSON
(`run1/`); rerun with a stand-in `estimator.py` wrote the reports above. Pod 17:18:15–19:19 (7,289 s) ≈ **$1.30**, stopped.

**All B4 nominee checks now PASS**: graded #337562 0/100, verify #3680, VAL22 parity, 100-seed rehearsal, 100-call native-Linux memory,
worst-day residual +20.0 % / +17.7 %.


## B. Cut-and-refit — does a refit pay back X2's exact row cut A?

Owner: L3b arm-B subagent (started 16:00 UTC). Not baked, not submitted; nothing designated.

### B0. Prediction and keep rule (copied from the handoff before any number)

Refit recovers about half of the cut's raw loss (+3 % raw at −3 % F ⇒ score-equiv ≈ 0); **P(score-equiv ≤ −1 %) ≈ 0.25**.
Keep: any visit with score-equivalent ≤ −1 % and its 95 % CI excluding 0 by 19:30 UTC ⇒ handoff note to L2 (bake with λ 3.5e6,
competes with A5s on VAL-paired score-equivalent). Otherwise the table goes to the write-up as the row-elasticity-with-refit price.

### B1. The cut (exactly X2's variant A)

X2's `src/vA_omax.py` row rule: `_K2_O[l] = max(R, D, H)[l]` for l = 0..14 (KAS keep counts; O must contain the D21/birth
columns mapped through `_oinv`), l = 15 stays 704. A4's `_KAS_K` is identical to 8be's, so the rule gives the same line;
`src/make_cut.py` derives it from A4's own tables and asserts it equals X2's vA line. Rows cut per layer
[0,0,0,32,0,32,32,32,32,64,32,64,32,64,32,0] (448 of the transported rows). One-line diff vs A4's export (67ed6090) ⇒
`src/a4cut_start.py` **76ee5c86…**; with A4s's planner line (`src/bake_lam35.py`, λ 1e6 → 3.5e6, the bake L2 would apply)
⇒ `src/a4cut_start_lam35.py` **fee86bf6…** (the same bake on the export reproduces A4s 7b0d6800 byte-for-byte).

**The cut changes only what it should** (pod VAL0, CUT vs uncut control, nets VAL900/901): predictions at layer indices
0, 1, 2 are byte-identical; the first difference is at layer index 3 (the first layer with a cut, 32 rows), on 52 / 47 of 1,024
neurons (max |Δ| 2.7e-4 / 1.4e-4); from layer index 4 on, the change has propagated to ≈ all neurons (as any upstream change does).

### B2. Bill — real flopscope 0.12.3 meter (`l3_bill.py` = `tools/_cost_worker.py` path with seed-regenerated full weights), VAL900 and VAL901

| source | planner λ | FLOPs (900 = 901) | ops | ΔF vs A4 at same λ | Δops |
|---|---|---|---|---|---|
| A4 export 67ed6090 | 1e6 | 288,310,814,515 | 22,976 | — | — |
| cut 76ee5c86 | 1e6 | 283,946,467,522 | 23,019 | **−1.514 %** | +43 |
| **A4s 7b0d6800** (bake) | 3.5e6 | **293,790,066,123** (= A4s's asserted bill) | 20,991 | — | — |
| **cut + λ 3.5e6 fee86bf6** | 3.5e6 | **289,126,365,310** | 21,054 | **−1.587 %** | +63 |

Both nets bill identically (shape-determined). X2 measured −1.521 % / +43 ops for the same cut on 8be, so the bill transfers.
The score-equivalents below use the baked ΔF **−1.587 %** (the cut as it would ship). Break-even raw is therefore +1.61 %;
the keep line (score-equiv ≤ −1 %) needs Δraw ≤ +0.597 % vs A4's export. Residual: +63 ops (+0.3 %) on A4s's 20,991.

### B3. Refit set-up

- **Pod** `zfxqmgmmvnyhuv` (`arc-whest-L3b-cutrefit`), RTX 5090 **Community $0.69/h** (Secure 5090 had no stock at 16:01),
  created 16:01:20 UTC, image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, host driver 570 (CUDA 12.8).
  No torch 2.13 wheel exists for cu128, so the run uses the image's **torch 2.8.0+cu128 / numpy 2.1.2** (+ flopscope 0.12.3,
  whestbench 0.16.1), as L4's J did.
- **Corpus `l3bext`** (`build_l3bext.py`, on the pod from HF `keenanpepper/arc-whestbench-p2-d8b-corpus-14k@cc3405a9`):
  the handoff's "shard_ext07 rows 0..2047" does not exist — each ext shard holds 1,500 nets. L7 uses ext03/04 (arm a) and ext05/06
  (arm b), so the fresh corpus is **shard_ext07 rows 0..1499 ∪ shard_ext00 rows 100..647** (FRESH-D owns ext00 rows 0..99) = 2,048 nets,
  A4's corpus size, so the roster has A4's exact shape (updates 1..256 all fresh, then TRAIN900, 4 TRAIN rows omitted).
  Recipe re-proved array-equal on d8b shard00; asserted disjoint from FRESH-D, d8bext2 (A4) and d8bext3 (L7 a).
  `l3bext_mean.npy` sha256 **59bde3da04b24521aab7138b6b0a7bab7f12a27cc8da109bf5fd770258ca325a** (idx json in `pod/art/`).
- **Label check** (L6's `labelcheck/mc_check.py`, our own N = 1e8 MC on the pod): net idx 2055 (ext07 row 0) mean |Δm1| **5.0e-5**,
  max 5.1e-4, layer-16 MSE 2.3e-9 ⇒ gate ≤ 2e-4 **PASS** (L6: 4.3e-5 on its nets).
  Weight generator checked at start against R3's d8b parity hashes (0/1/2/777/2047) and L6's pod-MC shas (2049/2050): PASS.
- **Trainer** `l3b_train.py` = A4's `a4_train.py` loop unchanged (NEW = U1 nine tables, corrector LR ×10, cosine 368, VAL 900..999
  at 0/92/184/276/368, earliest strict best, step 0 = fallback, per-visit export) on the pod layout of `2026-10-20_L3/gaingroups/l3_train.py`;
  only the corpus and the generator path differ.
- **Zero control (uncut A4 export, pod, torch 2.8)**: VAL0 **1.483259e-8** vs L6's A4 s368 VAL 1.483245e-8 ⇒ paired **+0.001 %**
  [−0.006, +0.008], 47/100, max |Δpred| 2.1e-6. **Not byte-identical** (L6 ran torch 2.13+cu130; the handoff's byte-identity check
  cannot hold across torch versions) — it is the cross-version f32 floor L4 also measured (+0.002 %), 1/600 of the effects below.
  Rows are paired against L6's recorded A4 VAL (the handoff's reference); against the pod's own control they differ by ≤ 0.001 pt.
- Speed ≈ 13–15 s/update alone on the GPU.

### B4. VAL table (paired vs A4's export VAL, L6 `a4_run/val_step368.json`; ΔF −1.587 % real meter; `val_table.py`)

| step | mean raw | paired Δraw vs A4 | wins/100 | 95 % CI | score-equiv [95 % CI] | export sha |
|---|---|---|---|---|---|---|
| 0 (cut, unrefit) | 1.575572e-08 | +6.145 % | 5 | [+5.271, +7.019] % | +4.460 % [+3.600, +5.320] | 76ee5c86 |
| 92 | 1.626087e-08 | +9.588 % | 5 | [+8.500, +10.675] % | +7.848 % [+6.778, +8.918] | f1923093 |
| 184 | 1.593010e-08 | +7.341 % | 5 | [+6.308, +8.375] % | +5.637 % [+4.620, +6.654] | de292cfb |
| 276 | 1.577493e-08 | +6.292 % | 6 | [+5.389, +7.194] % | +4.604 % [+3.716, +5.493] | f01248e3 |
| **368** (best, earliest strict) | **1.571107e-08** | **+5.832 %** | **6** | **[+4.948, +6.716] %** | **+4.152 % [+3.282, +5.022]** | 518e3e9e |

Against the pod's own uncut zero control the 368 row is +5.831 % (identical to 0.001 pt). Run complete 17:32 UTC (`complete.json`
passed, best = 368, peak CUDA 3.18 GB). Per-visit exports: `pod/art/runs/CUT/estimator_stepNNN.py`; VAL jsons (per-net vectors, fitted
tables) beside them; train log `pod/art/runs/CUT/train.jsonl`.

Unrefit, the cut costs **+6.145 % raw on A4** — X2 measured +6.149 % on 8be: the cut's raw price did not move across five refits.

### B5. Decision and what it says

**NO-GO.** Best visit (368) is score-equivalent **+4.15 % [+3.28, +5.02]** — worse than A4's export, CI far from the keep line
(≤ −1 %). No handoff note to L2; nothing baked or submitted.

- Prediction vs outcome: predicted the refit recovers half (+3 % raw) ⇒ score-equiv ≈ 0, P(≤ −1 %) ≈ 0.25. Measured: the refit recovered
  **0.31 pt of 6.15** (+6.145 % → +5.832 %), and part of even that is the fresh data, not the cut: A4's own epoch on 2,048 fresh nets gave
  −0.60 % vs its parent. Against an uncut refit on the same data (expected ≈ −0.5 % vs A4, L7's arms measure it) the cut's price
  *with refit* is ≈ +6.3 %, i.e. **recovery ≈ 0**. The trajectory has A4's shape (early harm at 92, then descent), so the optimiser
  worked; the nine NEW10 gain tables plus the corrector simply cannot replace the transported D3 on the most-negative-α rows.
- Row elasticity *with* refit (the number the ledger lacked): Δraw / |ΔF| ≈ 5.83 / 1.59 ≈ **3.7** (unrefit 3.9; X2 on 8be 4.0).
  Break-even is 1. Any exact-row cut in this representation costs ~4× its FLOP saving in raw MSE, refit or not. For the write-up:
  the transported rows are content, not slack, and a gain-table refit does not buy them back.
- Instrument notes: torch 2.8 on the pod (no cu128 wheel for 2.13) ⇒ VAL0 of the uncut parent is +0.001 % [−0.006, +0.008] vs L6's
  torch-2.13 VAL, not byte-identical; 600× below the effects measured here, and the cut rows are also paired against the same-pod control.

### B6. Spend

| pod | GPU | $/h | up | cost |
|---|---|---|---|---|
| zfxqmgmmvnyhuv `arc-whest-L3b-cutrefit` | RTX 5090 **Community** (Secure had no stock) | 0.69 | 16:01:20–17:32:40 UTC (uptime 5,473 s), **stopped** | **≈ $1.05** |

Budget ≤ $6. Only this pod was created/stopped by this arm. The pod's disk (stopped) keeps the bundle, `cache/l3bext` (2,048 labels) and
the HF shards ext07/ext00 for any post-deadline reuse; terminate it when no longer wanted.
