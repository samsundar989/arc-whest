# L6 — d8bext2: 2,048 fresh labelled nets, then A4 = A3's export + 368 updates on them

Handoff: `_claude_snap/handoff_2026-10-19/HANDOFF_2026-10-21_L6_FRESH_LABELS_A4.md`. Started 07:12 UTC Oct 21.

## 0. Prediction and keep rule (written before any A4 number)

- Prediction (handoff): A4 vs A3's export **−0.5 % raw** VAL (80 %: −0.1 … −0.9 %), 0 FLOPs.
- Keep: VAL ≤ −0.3 % vs A3's export (≥ 60/100, 95 % CI excl. 0) **and** FRESH-A100 vs A3 ≤ −0.25 % (CI excl. 0);
  then bake with `_SW_OP_FLOPS = 3000000` (λ line count == 1), parity vs trainer preds, real bill (2 nets),
  `tools/verify_nominee.py` all stages with memory on a pod, 100-fresh-seed sweep, canary ≤ 19:00 UTC (hard 20:00).
  Otherwise A stays A3s/A1s.

## 1. Labels — no pods needed: the corpus already had them

**Correction to the handoff premise.** `tools/fresh/fresh_d.py` is not a labeller; it converts keenanpepper's public
corpus `hf://datasets/keenanpepper/arc-whestbench-p2-d8b-corpus-14k` (H200 MC, N = 1e8, raw moments m1..m4). That corpus
holds **14,048** labelled nets: d8b = idx 0..2047 (16 small shards, our `cache/d8b`) **plus 12,000 more in
`shard_ext00..07`** (1,500 each, strided: shard_ext*s* holds idx 2048 + 8k + s). Nothing in the repo had trained on the
ext shards; FRESH-D owns shard_ext00 rows 0..99 only. (As L2 already noted, "d8bext2048" was d8b itself.) So the
2,048 fresh nets were a 756 MB download, not 23 GPU-hours, and they carry the **same labeller, generator and N as
d8b** — the closest possible match to the corpus A1/A3 trained on.

- **d8bext2** = shard_ext01 rows 0..1023 ∪ shard_ext02 rows 0..1023 → idx 2049 + 8k and 2050 + 8k, k < 1024
  (2049 … 10234; full list in `cache/d8bext2/d8bext2_idx.json`). No overlap with FRESH-D; all idx ≥ 2048.
  HF revision cc3405a9; both shards `n_done = 1500`, `w_seed0 = 770000`, `n_samples = 1e8`.
- Built by `build_d8bext2.py`: label = m1 (float32), the exact recipe that reproduces `cache/d8b/d8b_mean.npy`
  (asserted array-equal on shard00 before use); m1[-1] == final_mean to 1e-6 for every row.
  `cache/d8bext2/d8bext2_mean.npy` (2048, 16, 1024) f32, sha256 **9ce74dc3…** (full hash in the idx json).
  Raw shards kept beside it (m1..m4) for the write-up and post-deadline builds.
- **Independent verification (pod dvo6t3wij5b9vl, RTX 5090, 07:16–07:22 UTC)** — `labelcheck/mc_check.py`: weights rebuilt
  with Linux torch CPU `manual_seed(770000 + idx)`, our own N = 1e8 MC (fp32, TF32 off, f64 cross-chunk sums,
  `h @ W[l]` as `tools/fresh/mc.py`), vs the corpus m1:

  | net | role | weight sha256 | mean \|Δm1\| | max \|Δm1\| | layer-16 MSE |
  |---|---|---|---|---|---|
  | 0 | d8b control | 2fe17c4a… = d8b fixture ✔ | 4.25e-5 | 4.8e-4 | 1.19e-9 |
  | 2049 | d8bext2 row 0 | eb023fcc… | 4.30e-5 | 4.2e-4 | 1.19e-9 |
  | 2050 | d8bext2 row 1024 | e2a715d7… | 4.36e-5 | 4.7e-4 | 1.81e-9 |

  Gate mean |Δm1| ≤ 2e-4: **PASS ×3**. Layer-16 MSE ≈ 2 × the N = 1e8 label noise (≈ 7.7e-10) — pure MC noise; the
  new nets agree exactly as well as the d8b control. `labelcheck/labelcheck.json`.
- **Trainer weight source**: L6 copies of the WSL IPC generator (`ipc/helper.py`: index bound 2048 → 14048, nothing else;
  `ipc/client.py`: FAMILY pinned, verify() also probes 2049/2050 against the pod's shas). `ipc/smoke.py` PASS 07:2x UTC:
  d8b fixtures 0/1/2/777/2047 + d8bext2 2049/2050 all match ⇒ corpus labels, our MC and the trainer's weights are
  three-way consistent.

## 2. A4 refit

`a4_train.py` = A3's trainer (`2026-10-21_L2/src/a3_train.py`) with only the corpus changed: roster = d8bext2 shuffled,
then full 0..899 shuffled (seed 20260925), 368 × 8 — updates 1..256 are all fresh nets (asserted), 4 TRAIN rows omitted.
Corrector LR ×10, cosine 368, VAL 900–999 at 0/92/184/276/368, earliest strict best, step 0 = fallback.
`--parent-val` asserts VAL0 predictions and per-net MSE byte-identical to the parent's VAL.

**Parent = A3 step 368** (L2 #3645 complete 07:50 UTC; best_val step 368: 1.492264e-08, −1.418 % vs A1, 85/100,
CI [−1.665, −1.171] % — strictly beats 276) → `src/a3_export_s368.py` sha **49179dfc…**, its VAL
`src/a3_val_step368.npz` (74554388…) / `.json` (a69557de…). A4's paired table is vs this VAL (`val_table.py`).

**#3660** queued 07:53 UTC (16 GiB RAM / 8 GiB VRAM / 3 CPU, as A3's #3645: peak 7.1 GiB RAM, 3.3 GiB CUDA), behind
L4's J #3650 (declared 4 h 30 m, started 07:51) ⇒ expected run ≈ 12:20–16:00 UTC at A3's ≈ 33 s/update.
The job reads `ipc/` and `cache/d8bext2` from the live tree (absolute paths, as A3's trainer): frozen until it ends.

**Started 08:08 UTC** (L4 cancelled J #3650, freeing the GPU). Generator parity proof PASS incl. the d8bext2 fixtures
(0/1/2/777/2047/2049/2050/0). **VAL0 = 1.4922636593e-08 = A3 s368, predictions and per-net MSE byte-identical
(asserted in-run).** 35 s/update ⇒ step 368 ≈ 11:50 UTC.

### A4 VAL table (paired vs A3 s368; `python val_table.py a4_run src/a3_val_step368.json`)

| step | mean raw | mean ratio | paired Δraw vs A3 | wins/100 | 95 % CI |
|---|---|---|---|---|---|
| 0 | 1.492264e-08 | 1.00000 | +0.000 % | — | — |
| 92 | 1.533169e-08 | 1.02741 | +2.772 % | 16 | [+2.220, +3.324] % |
| 184 | 1.506249e-08 | 1.00937 | +0.966 % | 32 | [+0.598, +1.334] % |
| 276 | 1.489058e-08 | 0.99785 | −0.190 % | 58 | [−0.485, +0.104] % |
| **368** | **1.483245e-08** | **0.99396** | **−0.600 %** | **73** | **[−0.793, −0.407] %** |

#3660 complete 10:3x UTC (best_val = step 368, earliest strict best; peak CUDA 3.27 GB). **Gate 1 PASS** (≤ −0.3 %, ≥ 60/100,
CI excl. 0) — on the prediction (−0.5 %). Updates 1–256 were all d8bext2 nets; 276 → 368 (TRAIN900 tail, low LR) added −0.41 pts.
Export `src/a4_export_s368.py` sha **67ed6090…**; vs A3 s368 only fitted tables differ (35 names, values only).
Step 92 (08:56 UTC) = peak-LR early-harm phase, as A3 (+1.27 % at its 92) and R3 (+0.75 %); twice A3's, consistent with
restarting fresh Adam at LR ×10 from a parent that had itself annealed to LR 0.

## 3. Bake, gates, canary

**Planner correction.** The handoff's `_SW_OP_FLOPS = 3e6` assumed A1s's op count; A4 has A3's shape (NEW10, +741 ops), and L1
measured that shape at λ 3e6 = 21,371 ops, worst-day headroom **+8.9 % central — fails the ≥ 10 % rule** (L1 RESULT §6 table).
The cheapest rule-passing planner for this shape is **λ 3.5e6** (293,790,066,123 F, 20,991 ops, +10.5 % / +7.9 %). Bills are
shape-determined, so L1's A3s table is A4's. Both bakes are made; **λ 3.5e6 is the candidate**, λ 3e6 is kept for the handoff's record.

| bake | sha256 | `_SW_OP_FLOPS` | validate |
|---|---|---|---|
| **`cand_a4s368_lam35.py` (A4s)** | **7b0d6800…** | 3,500,000 | PASS |
| `cand_a4s368_lam30.py` | 30cc2466… | 3,000,000 | PASS (fails headroom rule; not gated) |

One-line diff vs the export, count asserted == 1. Memory: L1's pod 100-call gate on A3s λ 3.5e6 (7e1e1a2e) PASS is shape-only and
A4s has that exact shape (tables differ only) ⇒ carries over; the pod rehearsal below adds RLIMIT_AS 8 GiB evidence on A4s itself.

- **FRESH-A100 #3668 (11:12 UTC; exit 2 = ev.py "not accepted" under `--no-cost`, as A1's/A3's): Gate 2 PASS.** A4s raw
  1.513606e-8 vs A3 1.522384e-8 (`fresh_a_a4s368_lam35.json` vs `2026-10-21_L2/fresh_a_a3s368.json`, n = 100, 0 failed):
  ratio 0.994234, paired **−0.583 % [−0.787, −0.379]**, **72/100** (gate ≤ −0.25 %, CI excl. 0). FRESH agrees with VAL (−0.600 %).
- **`verify_nominee` #3669 (11:20 UTC): VERDICT PASS** (`verify_a4s_lam35/`) — bill **293,790,066,123 F (+0) / 20,991 ops**,
  determinism 2 processes × 3 nets bit-identical, rig 1024×16 / 512×16 / 256×4 finite, served-equivalent residual **0.3200 s**,
  wall 81.9 s; client-only venv import (no numpy). WARN: A1's `_pkk_fast_ok` literal handler (as A1/A3).
- **100-fresh-seed rehearsal: PASS** (L3 harness unchanged, real flopscope(-server) 0.12.3 + client-only venv, RLIMIT_AS 8 GiB,
  2 BLAS threads, seeds 9,250,000–099 = A1s's; pod 9aorp6isi1jl2q cpu3g 16 vCPU, 4 blocks × 25, 10:36–10:49 UTC; mint parity on
  9,250,000 = L3's whestbench-verified mint; `rehearsal/summary.json`, raw `rehearsal/runs/`):
  100/100 distinct nets, **0 errors / 0 non-finite / 0 shape faults**, client rc 0 and server alive in all blocks; **FLOPs
  293,789,618,663 on every call = 293,790,066,123 − 414,692 (A3's served offset) − 32,768 (output op)** ⇒ served F
  293,789,651,431, the A3s λ 3.5e6 bill to the FLOP; server VmPeak 3,673 MiB (45 % of the cap), client 452 MiB; server growth
  11.7–15.0 MB/call (flopscope-server bookkeeping, L3: 10.6–13.4). Pod residual/wall are host-local (different host from L3's A1s
  block) and are not used for any projection.

## 4. Spend

| pod | GPU | $/h | up | cost |
|---|---|---|---|---|
| dvo6t3wij5b9vl `arc-whest-L6-labelcheck` | RTX 5090 Secure | 1.19 | 07:15:43–~07:23 UTC (stopped) | ≈ $0.15 |
| 9aorp6isi1jl2q `arc-whest-L6-rehearsal` | cpu3g 16 vCPU Secure | 0.64 | 10:33:35 + 1,352 s (stopped) | ≈ $0.24 |

Budget was ≤ $35 for 3–4 label pods × 6 h; the corpus made them unnecessary. L1 brought up its own memory-gate pod
(`arc-whest-L1-memgate`, v0fza2b41jnbyw), so this lane does not host L1's gates.

### Canary forecast (written 11:2x UTC, before submission)

Score = A3 graded 1.8527052525e-9 (#337495) × FRESH raw ratio 0.994234 × served bill ratio 293,789,651,431 / 287,564,751,853
(1.021647; both = billed − A3's 414,692 served offset, A4s's confirmed to the FLOP by the pod rehearsal)
= **1.8819e-9** — vs A1s graded 1.8974e-9: **−0.82 %**; vs A3 (λ 6e5, fails headroom): +1.58 %.
Keep window ±1.5 %: **1.8537 – 1.9101e-9**. Expect 0/100 failed; served residual ≈ A3's 14.25 µs/op × 20,991 ⇒ median ≈ 0.299 s.
Worst-day headroom (L1 method, A3s λ 3.5e6 row): **+10.5 % central / +7.9 % pessimistic** (A1s +13.1 / +10.6) — passes the ≥ 10 %
rule centrally, not pessimistically; whether A4s replaces A1s as A-safe under D4 is the strategist's call.

### Canary A4s — AIcrowd #337511 (submitted 11:21:32Z, ledger 6/10)

**GRADED 1.8800496320794433e-9** (raw 1.407222086e-8, multiplier 0.1336001, 0/100 failed, completed 11:47:57Z;
`experiments/run_reports/a4s_L6_337511_graded_report.json`). vs forecast 1.8819e-9: **−0.10 %**, inside ±1.5 %.
**vs A1s #337481 (1.8973911e-9): −0.91 %.** vs A3 #337495: +1.47 % score, raw −0.67 % (FRESH predicted −0.58 %).
Served effective compute 293,789,651,431 F per MLP = the rehearsal's prediction to the FLOP.

| graded run | residual median / max | wall median / max | µs/op | worst-day residual (median × 1.196 / × 1.029) | headroom |
|---|---|---|---|---|---|
| **A4s #337511** | 0.2900 / 0.3115 s | 71.4 / 80.2 s | 13.82 | 0.347 / 0.357 s | **+15.3 % / +12.1 %** |
| A1s #337481 | 0.2905 / 0.3023 s | 70.2 / 74.0 s | 14.08 | 0.347 / 0.358 s | +15.1 % / +11.9 % |
| A3 #337495 | 0.3331 / 0.3447 s | 78.4 / 85.9 s | 14.25 | 0.398 / 0.410 s | +0.4 % / −2.4 % |

(Projection as L2 applied it to A3; L1's `margin.py` reads A1s at +13.1 / +10.6 % — same ordering, restate with it.) A4s served
1.8 % fewer µs/op than A1s; part of that may be node noise between grading runs, so treat the headroom as **≈ A1s**, not better.
The pre-grade estimate (+10.5 / +7.9 %, A3's 14.25 µs/op) was conservative.

**Outcome: A4s beats the current A-safe (A1s) by 0.91 % graded at the same worst-day headroom, and has every nominee check
(graded 0/100, FRESH, verify, memory by identity + RLIMIT_AS rehearsal, 100-fresh-seed rehearsal).** Whether A-safe becomes A4s
is the strategist's/Samuel's call (D4); this lane does not designate.

## 5. Summary

- Labels: 2,048 fresh labelled nets came free from keenanpepper's corpus ext shards (verified by our own MC on a pod), not from
  23 GPU-h of new MC. The remaining 9,852 ext nets (ext00 rows ≥ 100, ext01/02 rows ≥ 1024, ext03–07) are unused and are the
  obvious corpus for any post-deadline build.
- The corrector-data law held: one epoch on 2,048 new nets gave −0.60 % VAL / −0.58 % FRESH / −0.67 % graded raw (handoff predicted
  −0.5 %, 80 % −0.1…−0.9).
- Spend ≈ $0.39 of the $35 budget (two pods, both stopped).
