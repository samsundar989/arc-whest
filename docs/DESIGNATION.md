# DESIGNATION — current A and B (maintained by the worker; designating is Samuel's)

## Final nominee tables — A4s + B (D6, `_claude_snap/handoff_2026-10-19/DECISIONS_2026-10-20.md`; Lane L1, 2026-10-21 16:10 UTC)

**Update 17:15 UTC (strategist): B = B4 (#337562, 1.9724e-9 = B-young × 0.9859) — table "B4" below; B-young is the fallback B.**
**Strategist's nominees (D6, 16:00 UTC): A = A4s (#337511), B = B-young (#337462).** A3 is not nominated (residual cliff). A5s
(L7) replaces A4s only if it clears every gate and grades by ≈ 20:30 UTC; its row is below.
**Update 20:05 UTC (L2): A5s #337601 GRADED 1.8751e-9 = A4s × 0.9974, inside its forecast, every gate PASS ⇒ by Samuel's rule A = A5s** (row below; A4s is the fallback A). **Nothing here is a nomination or
designation: Samuel nominates by Oct 22 18:00 UTC (deadline 23:59).** Both entries are complete; nothing is owed on either.

**How to read the pair.** On a normal grader node A4s scores 6.0 % better than B. On a node 1.28× slower than the one that graded
A4s, A4s starts failing MLPs on the 0.4 s residual cliff and B does not (B lasts to 1.45×). A4s is the entry that scores; B is
the one that survives a slow node. They share no planner (cap 6 + pad vs cap 5, no pad) and no fitted tables (A4 chain vs A1 chain).
**With B = B4 (17:15)** the pair reads: A4s scores 4.7 % better than B4 (1.8800 vs 1.9724e-9); A4s fails at 1.284×, **B4 at 1.419×**
(L1 `nominee_margin.py` on #337562's 100 MLPs: residual 0.2674 / 0.2819 s, worst day 0.320 / 0.329 s = **+20.0 % / +17.7 %**, wall
+43 % / +41 %; = L3b's table). A4s and B4 now share the same fitted tables (both the A4 export) and differ only in the planner, so the
pair still insures against a slow node but no longer against a defect in A4's fitted values; B-young (A1 chain) remains the
fallback B that would.
**With A = A5s (20:05)** the pair reads: A5s scores 4.9 % better than B4 (1.8751 vs 1.9724e-9); A5s fails at **1.326×** (L1
`nominee_margin.py` on #337601's 100 MLPs: residual 0.2913 / 0.3017 s, worst day 0.349 / 0.359 s = +12.9 % / +10.4 %, wall +26 % / +24 %;
= L2's row), B4 at 1.419×. A5s's tables are a soup of two fresh-data epochs on A4's export, B4's are A4's export itself: related, not
identical, still the same chain lineage.

| field | **A = A4s** | **B = B-young** |
|---|---|---|
| what it is | A3 s368 export (A1 + NEW10, refit 368 on TRAIN900 ∪ d8b) + one epoch on 2,048 fresh corpus nets (d8bext2), best-VAL step 368; planner λ_op 3.5e6 (cap 6, pad) | A1 (8be + R3 refit, step 736) with planner only changed: cap 6→5, pad off, λ_op 5e6 |
| source / sha256 | `experiments/2026-10-21_L6/src/cand_a4s368_lam35.py` **7b0d6800** (committed 70ecd8f, clean) | `experiments/2026-10-20_L3/byoung_a1/estB_r3s736.py` **7a146f84** (committed, clean) |
| package / sha256 | `experiments/2026-10-21_L6/a4s-s368-lam35-7b0d6800.tar.gz` **c0fdf5f2** (inner estimator = 7b0d6800, re-checked 16:05) | `experiments/2026-10-20_L3/b-young-r3s736-7a146f84.tar.gz` **bfba72f7** |
| **graded** | **#337511 1.8800496320794433e-9**, raw 1.40722e-8, multiplier 0.13360, 0/100 failed, smoke PASS | **#337462 2.0005580379785525e-9**, raw 1.43373e-8, multiplier 0.13954, 0/100 failed, smoke PASS |
| forecast (written first) | 1.8819e-9 (−0.10 % ✔) | 2.0007e-9 (−0.01 % ✔) |
| vs each other / vs A1s / vs A1 | **B × 0.9398** / A1s × 0.9909 / A1 × 1.0077 | A4s × 1.0641 / A1s × 1.0544 / A1 × 1.0723 |
| real 0.12.3 bill / ops | 293,790,066,123 F / **20,991** ops (served 293,789,651,431) | 306,840,769,522 F / **18,943** ops |
| VAL-100 | −0.600 % raw vs A3 s368 [−0.793, −0.407], 73/100 (A3 vs A1 −1.418 %) | −0.010 % vs A1 [−0.048, +0.028], 47/100 |
| FRESH-A100 (N = 1e8 labels) | −0.583 % vs A3 [−0.787, −0.379], 72/100 (#3668) | −0.008 % vs A1 [−0.051, +0.035], 49/100 |
| graded served residual med / max | 0.2900 / 0.3115 s (13.82 µs/op) | 0.2594 / 0.2760 s (13.69 µs/op) |
| graded wall med / max | 71.4 / 80.2 s | 53.5 / 60.6 s |
| **worst-day residual** (central / pess.) | 0.347 / 0.357 s ⇒ **+13.3 % / +10.8 %** headroom to 0.4 s | 0.310 / 0.319 s ⇒ **+22.4 % / +20.2 %** |
| worst-day wall (central / pess.) | 88.7 / 91.4 s ⇒ +26 % / +24 % to 120 s | 66.4 / 68.5 s ⇒ +45 % / +43 % |
| uniform slowdown of graded run to first fail | **1.284×** (residual; wall 1.496×) | **1.449×** (residual; wall 1.979×) |
| … on top of the pessimistic worst day | 1.121× | 1.253× |
| worst MLP of an N-net rerun (today's node) | N = 1,000 0.331 s · N = 10,000 0.343 s | 0.296 s · 0.306 s |
| ≥ 10 % worst-day residual rule (L3) | PASS central and pessimistic | PASS, every method |
| memory, 100 calls, 8 GiB RLIMIT_AS | PASS by identity with the gated A3s λ 3.5e6 build (35 fitted constants differ, 0 code lines): VmPeak 4,593 MB, call 10→99 +1.0 MiB; + L6 RLIMIT rehearsal server VmPeak 3,673 MiB | PASS native Linux: VmPeak 3,668 MB, −12.1 MB (B-on-8be WSL +32 MiB) |
| fresh-seed rehearsal (real client↔server 0.12.3) | 100/100 (seeds 9,250,000–099), 0 errors / non-finite / shape faults, bill to the FLOP | 300/300 (seeds 9,250,000–299), same |
| verify_nominee | #3669 PASS (bill exact, 2-process determinism, rig 1024×16 / 512×16 / 256×4, client-only venv) | #3611 + #3625 PASS |
| known limitation (both) | IndexError at a global weight scale ×0.5 / ×2 (not He; § Provisional pair below) — a hard FAILED MLP, never a silent wrong answer | same |
| evidence | `experiments/2026-10-21_L6/RESULT.md` §2–3; `experiments/2026-10-21_L1/RESULT.md` §6 | § B nominee below; `experiments/2026-10-20_L3/RESULT.md` §3; `experiments/2026-10-21_L3/RESULT.md` §2 |

**B4 — nominee B (strategist, 17:15 UTC: "B4 graded 1.9724e-9 inside forecast ⇒ nominee B = B4"; L3b row, 2026-10-21).**
Replaces B-young (#337462) in the B column above; B-young stays the fallback B. Same planner as B, A4s's code and fitted tables.
Samuel designates; nothing here is a designation.

| field | **B4** |
|---|---|
| what it is | A4's selected export (67ed6090) + B's exact 3 planner lines (cap 6→5, pad off, λ_op 1e6→5e6) — the same diff as A1 → B |
| source / sha256 | `experiments/2026-10-21_L3b/src/estB4_a4s368.py` **4af12b5e** (committed 0a08bbf, clean) |
| package / sha256 | `experiments/2026-10-21_L3b/b4-a4s368-4af12b5e.tar.gz` **882541f0** |
| **graded** | **#337562 1.9724068470701697e-9**, raw 1.407e-8, multiplier 0.14019, 0/100 failed |
| forecast (written first) | 1.9728e-9 (−0.02 % ✔; keep band 1.943–2.002e-9) |
| vs B / vs A4s | **B × 0.98593 (−1.41 %)** / A4s × 1.0491 |
| real 0.12.3 bill / ops | 308,282,626,669 F / **19,684** ops (served 308,282,211,977 = forecast to the FLOP); +0.47 % F / +741 ops vs B |
| VAL parity vs A4s (22 nets) | raw +0.040 % [−0.026, +0.106], 8/22; max \|Δpred\| 1.03e-5 (planner-reorder floor) |
| graded served residual med / max | 0.2674 / 0.2819 s (13.59 µs/op) |
| graded wall med / max | 54.8 / 59.6 s |
| **worst-day residual** (central / pess.) | 0.320 / 0.329 s ⇒ **+20.0 % / +17.7 %** headroom to 0.4 s (B: +22.4 / +20.2) |
| worst-day wall (central / pess.) | 68.0 / 70.2 s ⇒ +43 % / +42 % |
| uniform slowdown of graded run to first fail | **1.419×** (residual; wall 2.01×); on top of the pessimistic worst day 1.215× |
| worst MLP of an N-net rerun (today's node) | N = 1,000 0.305 s · N = 10,000 0.316 s |
| ≥ 10 % worst-day residual rule | PASS central and pessimistic |
| memory | **No exact identity exists, so it is measured on B4 itself.** B4 vs B is *not* fitted-constants-only: 481 diff lines, ≈ 119 of them code (A3's NEW10 terms that A4s carries); B4 vs A4s differs only in the 3 planner lines, and B's flat-memory pass (native Linux 3,668 MB, −12.1 MB / 100) is on A1's code. **100-call native-Linux gate on B4 itself: PASS** (pod s0detmbqch7w8i, `linux_memory_gate.py`, L1's arguments: RLIMIT_AS 8,192 MB, f64 read-only, 2 BLAS threads, seed 90000000101, `--lifecycle-control-mib 16`; all controls PASS): 100/100 finite, one FLOP count, **VmPeak 3,848 MB** (2.1× under the cap), **settled call 10 → 99 −30.2 MiB**, block medians 2170 / 2163 / 2159 / 2141 MiB (no trend). Same-host B control: VmPeak 3,854 MB, +1.0 MiB, 2005 / 2005 / 2006 / 2006. The gate's printed first → last "FAIL" (B4 +364, B +198 MB) is the call 0 → 10 allocator warm-up every build shows (GOTCHAS 9). Plus the client/server RLIMIT rehearsal 100/100 (server VmPeak 2,740 MiB). Raw: `experiments/2026-10-21_L3b/pod_mem/` |
| fresh-seed rehearsal | 100/100 (seeds 9,250,000–099), 0 errors / non-finite / shape faults, bill to the FLOP |
| verify_nominee | #3680 PASS (bill exact, determinism, rig 1024×16 / 512×16 / 256×4, client-only venv) |
| diversity vs A4s | **shares A4s's fitted tables** (B shared none): the pair still differs in planner (cap 5 / no pad vs cap 6 / pad) and so in op count, residual and memory path — the failure modes the pair is built around — but no longer in accuracy tables |
| evidence | `experiments/2026-10-21_L3b/RESULT.md` §A; `experiments/run_reports/canaryB4_L3b_337562_graded_report.json` |

Method: as § Worst-day rerun margin (`experiments/2026-10-21_L1/margin.json` constants: tail residual 1.19631 #334487, wall 1.24193
#336083; day shift 1.02872 / 1.03132). The A4s row is recomputed from its own per-MLP telemetry
(`experiments/run_reports/a4s_L6_337511_graded_report.json`); headroom = 1 − worst/cap (L6's table states cap/worst − 1 = +15.3 %;
same numbers). A4s's 13.82 µs/op is 1.8 % below A1s's 14.08 on another day; part of that may be node noise, so read its headroom as
≈ A1s's (+13.1 / +10.6 %), not better. **Fallback A if A4s is set aside: A1s** (e823d62f, #337481 1.8974e-9, every check PASS).

| A5s (L7 last push) | build | sha | graded | headroom | memory | FRESH | verify |
|---|---|---|---|---|---|---|---|
| **GRADED — A = A5s by Samuel's rule (graded inside ±1.5 % of forecast and FRESH ≤ −0.25 % CI excl. 0); nominating is Samuel's** | soup(b,c) s368 = mean of the 35 fitted tables of L7 arms b (d8bext4, LR ×10) and c (d8bext3, LR ×3), both one fresh epoch on A4's export; VAL −0.454 % vs A4 [−0.599, −0.309], 74/100 (b −0.379, c −0.400); baked `_SW_OP_FLOPS = 3500000` as A4s, 0 code lines differ; `experiments/2026-10-21_L7/src/cand_a5s_soupbc_180e488b.py` (≡ `cand_a5s_soupbc368_180e488b.py`), package `a5s-soupbc368-lam35-180e488b.tar.gz` 2ee01ae9 | **180e488b** | **#337601 1.8751280753233397e-9** (forecast 1.8715e-9, +0.19 % ✔), raw 1.40354e-8, multiplier 0.13360, 0/100 failed; **A4s × 0.99738** | served 0.2913 / 0.3017 s (13.88 µs/op), wall 71.3 / 78.5 s ⇒ worst-day **+12.9 % / +10.4 %**, first fail **1.326×** (A4s 1.284×) | PASS by identity with A4s (fitted constants only) + RLIMIT_AS 8 GiB 100-seed sweep 100/100, VmPeak 3,651 MiB, bill to the FLOP | −0.531 % vs A4s [−0.665, −0.397], 77/100 (#3688) | #3689 PASS (293,790,066,123 F +0 / 20,991 ops, determinism, rig, client-only venv); parity PASS |

## Provisional pair (D4, `_claude_snap/handoff_2026-10-19/DECISIONS_2026-10-20.md`) — status, Lane L1, 2026-10-21 07:30 UTC (superseded by D6 above)

**Pair = A-safe + B.** A-safe = **A3s** (A3's selected export + `_SW_OP_FLOPS` 3e6) if A3 clears its gates, else **A1s**. **A1 is not
nominated.** No scale-crash change on the finals path. Nothing here is a nomination; Samuel nominates by Oct 22 18:00 UTC.
**L1 reading (11:55 UTC): A-safe = A4s** (graded 1.8800e-9, worst-day residual +13.3 % / +10.8 %, = A1s headroom at −0.91 % score); A1s is the fallback A-safe. (09:20 reading A1s over A3s stands for A3s.)

| role | build | sha | graded | worst-day residual headroom (central / pess.) | 100-call memory | FRESH-A100 | verify_nominee |
|---|---|---|---|---|---|---|---|
| A-safe (D4 preferred) | A3s = A3 s368 (5ddf3290; **A3 graded #337495 1.8527e-9 = A1 × 0.9931**, 14.25 µs/op) + λ_op 3e6–4.5e6 (L1 bills #3659/#3666) | not baked | projected from A3 graded × F ratio: 3e6 1.8837e-9 (fails rule); **3.5e6 1.8928e-9 = A1s − 0.24 %**; 4e6 1.8977e-9 = A1s; 4.5e6 1.9121e-9 | 3e6 +8.9/+6.3 % ✗; 3.5e6 +10.5/+8.0 %; 4e6 +11.3/+8.7 %; 4.5e6 +13.0/+10.5 % | λ 3e6 PASS (4,649 MB, −8 MiB); **λ 3.5e6 PASS** (4,593 MB, +1.0 MiB) | L2 A3 FRESH −1.625 % vs A1 | L2 #3665 PASS (λ 6e5) |
| **A-safe (L1 recommended)** | **A4s** = A4 s368 (A3 + one fresh epoch) + λ_op 3.5e6 `experiments/2026-10-21_L6/src/cand_a4s368_lam35.py` | **7b0d6800** | **#337511 1.8800e-9**, 0/100 (= A1s × 0.9909) | **+13.3 % / +10.8 %** (served 0.290 / 0.312 s, 13.82 µs/op; fails at 1.284×) | PASS by identity (gated A3s 3.5e6 code; VmPeak 4,593 MB, +1.0 MiB) + L6 rehearsal 100/100 | L6 #3668 −0.583 % vs A3 | L6 #3669 PASS |
| **A-safe (fallback, current)** | **A1s** `experiments/2026-10-21_L1/src/cand_r3s736_lam30.py` | **e823d62f** | **#337481 1.8974e-9**, 0/100 | **+13.1 % / +10.6 %** | **PASS** (pod v0fza2b41jnbyw, native Linux 8 GiB: VmPeak 4,543 MB, settled call 10→99 +33.7 MiB, plateau; 8be control +1.0) | **PASS** #3657: raw 1.547531e-8, Δ vs A1 −0.009 % (t95 [−6.69e-13, +3.86e-13]), 50/100 | PASS #3652 |
| **B** | B-young on A1 chain `experiments/2026-10-20_L3/byoung_a1/estB_r3s736.py` | **7a146f84** | **#337462 2.0006e-9**, 0/100 | **+22.4 % / +20.2 %** | PASS (native Linux 3,668 MB, −12.1 MB; B-on-8be WSL +32 MiB) | −0.008 % vs A1 [−0.051, +0.035] | PASS #3611 + #3625 |
| **B4** (L3b; graded inside ±1.5 % ⇒ B slot by the L3b rule) | B's 3 planner lines (cap 5, no pad, λ_op 5e6) on A4's export `experiments/2026-10-21_L3b/src/estB4_a4s368.py` | **4af12b5e** | **#337562 1.9724e-9**, 0/100 (= B × 0.9859) | **+20.0 % / +17.7 %** | by identity (B planner + A4s tables) + rehearsal 100/100 | VAL22 +0.040 % vs A4s [−0.026, +0.106] | PASS #3680 |
| not nominated | A1 | 7d381fc5 | #337456 1.8656e-9 | +2.5 % / −0.3 % | WSL PASS | — | PASS |
| not nominated (L2, base of A3s) | A3 = A1 export + NEW10 refit 368 (TRAIN900 ∪ d8b) s368 + λ 6e5 `experiments/2026-10-21_L2/src/cand_a3s368_lam06.py` | 5ddf3290 | **#337495 1.8527e-9**, 0/100 (forecast 1.8447e-9 +0.43 %); residual 0.333 / 0.345 s, wall 78.4 / 85.9 s | **+0.4 % / −2.4 %** ✗ (first fail 1.160×) | not run (A3s measured by L1) | −1.625 % vs A1 [−1.856, −1.394], 89/100 | PASS #3665 (287,565,166,545 F / 23,374 ops) |

**Rerun rule — why the pair is chosen on headroom.** The private rerun is a *fresh* suite of N MLPs, with N chosen "to reliably
distinguish" entries (likely ≫ 100), and one MLP over the 0.4 s residual or 120 s wall cliff zeroes that entry's prediction for that
net (≈ 1e-2 adjusted — out of the money). The worst MLP of a suite is a tail draw, and more MLPs mean deeper tail draws. Pooling every
per-MLP served residual of our 84 graded runs divided by its own run's median (8,393 draws; `experiments/2026-10-21_L1/telemetry.json`),
the expected worst-of-N ratio is **×1.095 at N = 100, ×1.140 at N = 1,000, ×1.181 at N = 10,000** (wall ×1.127 / ×1.205 / ×1.242).
On today's node rates that puts the worst MLP of a 1,000-net suite at **A1 0.372 s, A1s 0.331 s, B 0.296 s** (10,000 nets: 0.385 /
0.343 / 0.306 s), before any slow-node day effect. A1's public 0/100 is therefore weak evidence about its rerun; A1s and B keep their
margin at any plausible N. Ranking within the pair is by score; choosing the pair is by headroom.

**Known out-of-distribution limitation (not fixed, D4.4).** Every build since v87_B — 8be, A1, A1s, B, and A3s by construction —
raises `IndexError` in `_yp_contract` via `_dslice21` when the weights are scaled ×0.5 or ×2 from He (×0.9 and ×1.1 pass;
`experiments/2026-10-21_L1/RESULT.md` §1). It is a hard crash (a FAILED MLP, never a silent wrong answer). The rerun suite is drawn
from the same He distribution, and every graded run, including the organisers' smoke test, has passed.

## A1 = R3 refit of 8be + planner λ 0.6e6 — NOT NOMINATED (D4: residual headroom; Lane L2 record, 2026-10-21 04:00 UTC)

A1 = exact 8be (#336993) refit by the held R3 recipe (fresh Adam, corrector LR ×10, 736 updates on TRAIN900 + d8b2048, best-VAL
export = step 736), then `_SW_OP_FLOPS` 1e6 → 6e5 (planner only). No new code paths. Evidence: `experiments/2026-10-20_L2/RESULT.md`.

| field | A1 |
|---|---|
| file / sha | `experiments/2026-10-20_L2/src/cand_r3s736_lam06.py` / **`7d381fc5`** (export `val_step0736/estimator.py` 73321458 + λ line) |
| package | `experiments/2026-10-20_L2/a1-r3s736-lam06-7d381fc5.tar.gz` sha `4c5ca48e` |
| **graded** | **#337456, 1.8656192469864824e-9** (raw 1.4338e-8, multiplier 0.130114, 0/100 failed); forecast 1.863e-9 (interval 1.76–1.88e-9) ✔; −1.61 % vs #336993 |
| real 0.12.3 bill | 286,123,309,398 F / 22,633 ops (graded 286,122,895,009 = bill − 414,389, the same offset as 8be) |
| VAL-100 vs 8be (trainer) | −1.361 % raw [−1.588, −1.134], 89/100 |
| FRESH-A100 vs 8be (N = 1e8 labels) | −1.480 % raw, Δ 2.325e-11 t95 [1.89e-11, 2.76e-11], 87/100 |
| served residual (graded) | max 0.3349 s, median 0.3261 s (cap 0.4 s) |
| graded telemetry #337456 | per-MLP wall median 78.0 s / max 83.2 s (cap 120 s), residual median 0.3261 / max 0.3349 s, **0/100 failed** (`experiments/run_reports/a1_337456_graded_report.json`) |
| **rerun margin** (L1, worst day on record) | residual 0.390 s central (**+2.5 % headroom**, pess. −0.3 %), wall 96.8 s (+19 %); first fail at a uniform 1.194× slowdown, on residual. **Fails the ≥ 10 % residual-headroom rule**; B passes (+22.4 %). See § Worst-day rerun margin |
| served-equivalent wall | 81.6 s (verify_nominee project, ref #336993) |
| determinism | 2 processes × 3 DEV nets bit-identical |
| imports / static | flopscope/whestbench/stdlib only; client-only venv import (no numpy); contract checks PASS |
| memory, 100 calls, RLIMIT_AS 8 GiB | **WSL #3626 PASS: VmPeak 4,613 MiB, settled growth call 10→100 −27.0 MiB**, 100/100 clean, FLOPs identical (`experiments/2026-10-20_L2/memory_s736_lam06/`). The legacy abs(last−first) row flags a −287 MB *shrink* (GOTCHAS item 9). Native-Linux pod: +200.7 MB, but exact 8be +204.2 MB on the same host ⇒ host behaviour. (λ 1e6 twin #3627: 4,523 MiB, +0.0 MiB) |
| rig (client↔server 0.12.3, 3 shapes) | **#3633 PASS**: 1024×16 69.3 s, 512×16 22.9 s, 256×4 1.3 s, all finite / right shape (`experiments/2026-10-20_L2/verify_s736_rig/`). #3632 errored only because the rig reuses the `imports`-stage stub and was run without it |
| clean checkout / fresh client venv | committed file == working file == DESIGNATION sha; client-only venv import (no numpy) + rig predict through that venv: PASS |
| **verdict** | **all nominee checks PASS** (checkout, imports, static, rig, meter/determinism/bill, project, WSL memory, FRESH, graded). Designation is Samuel's. |

## B nominee — B-young on A1's chain (Lane L3, 2026-10-21 04:25 UTC; recommended B — designating is Samuel's)

B = A1's source with the Strassen planner only changed (`_SW_CAP` 6→5, `_SW_PAD` True→False, `_SW_OP_FLOPS`→5e6; builder
`experiments/2026-10-20_L3/byoung/build_byoung.py`). Same arithmetic in a shallower, unpadded order: equal accuracy, ~16 % fewer
op calls, no pad path — it fails differently from A on runtime, not on accuracy. Evidence: `experiments/2026-10-20_L3/RESULT.md` §3.

| field | B |
|---|---|
| file / sha | `experiments/2026-10-20_L3/byoung_a1/estB_r3s736.py` / **`7a146f84`** (from A1 `7d381fc5`; identical when built from `73321458`) |
| real 0.12.3 bill | 306,840,769,522 F (DEV0–7 one count) / 18,943 ops — +7.24 % F vs A1 |
| VAL-100 vs A1 (real chain) | −0.010 % raw [−0.048, +0.028], 47/100 |
| FRESH-A100 vs A1 (N = 1e8 labels) | −0.008 % raw [−0.051, +0.035], 49/100 |
| served-equivalent residual / wall | 0.289 s / 83.0 s (verify_nominee project, ref #336993) |
| memory, 100 calls, RLIMIT_AS 8 GiB | native Linux: VmPeak 3,668 MB, settled growth −12.1 MB (B-on-8be via WSL: 3,691 MiB, +32 MiB) |
| verify_nominee | checkout/imports/static/rig (1024×16, 512×16, 256×4)/meter (determinism, exact bill)/project PASS (#3611 + #3625) |
| canary | **#337462 GRADED 2.0005580379785525e-9** (forecast 2.0007e-9; = A1 × 1.07233), raw 1.43373e-8, 0/100 failed |
| served residual max / wall max | **0.276 s / 61.1 s** (A1: 0.335 s / 83.6 s); per-MLP served-residual corr with A1 0.25 |
| package | `experiments/2026-10-20_L3/b-young-r3s736-7a146f84.tar.gz` sha256 bfba72f7 |
| graded telemetry #337462 | residual median 0.2594 / max 0.2760 s, wall median 53.5 / max 61.1 s, 0/100 failed |
| **rerun margin** (L1, worst day on record) | residual 0.310 s central / 0.319 pess. (**+22.4 % / +20.2 % headroom**), wall 66.4 / 68.5 s (**+45 % / +43 %**); first fail at a uniform 1.449× slowdown (residual; wall 1.979×). **Passes the ≥ 10 % rule** |
| **private-rerun rehearsal** (L3, Oct 21) | **300/300 fresh nets** (REHEARSAL seeds 9,250,000–299, never used before), real 0.12.3 client↔server, one participant process per 100 calls, server under RLIMIT_AS 8 GiB: 0 errors, 0 non-finite, 0 shape faults; FLOPs 306,840,322,365 on every net (= graded − 32,768 output op); server VmPeak 3,550 MiB; server-side growth ≈ 11 MB/call is flopscope-server bookkeeping (0.57 KB/op, same per op as A1), not B (`experiments/2026-10-21_L3/RESULT.md` §2) |
| determinism, cross-host | bill identical on pod Linux and Windows native; pod client↔server ≡ in-process bit-for-bit; Linux vs Windows max \|Δ\| ≤ 5.0e-6 (≈ 2e-4 relative on score); WSL leg not run (refit guard) |
| **verdict** | **all B checks PASS** (checkout, imports, static, rig, meter/bill, project, native-Linux memory, graded, rehearsal, worst-day headroom). Designation is Samuel's. |

## Worst-day rerun margin — A1, λ 1e6 twin, B (Lane L1, 2026-10-21 05:00 UTC; no nomination/designation made)

From 82 graded runs on 23 UTC days (Sep 14 – Oct 21), per-MLP telemetry. Our bill is constant over MLPs, so the spread inside a run is
grader-node noise. **Central** = own graded median × worst tail on record (residual max/median 1.196, #334487; wall 1.242, #336083);
**pess.** = × worst same-code day shift (1.029 / 1.031, v50 ×3); **per-op** = ops × worst per-op rate on record (15.92 µs/op, #336147).
Evidence and per-day tables: `experiments/2026-10-21_L1/RESULT.md` §2 (`margin.json`, `telemetry.json`).

| nominee | ops | worst-day residual (central / pess. / per-op) | residual headroom to 0.4 s | worst-day wall (central / pess.) | wall headroom to 120 s | uniform slowdown of graded run to first fail |
|---|---:|---|---|---|---|---|
| **A1** `7d381fc5` (#337456) | 22,633 | 0.390 / 0.401 / 0.360 s | **+2.5 % / −0.3 % / +9.9 %** | 96.8 / 99.9 s | +19 % / +17 % | **1.194× (residual)**; wall 1.442× |
| A1 λ 1e6 twin `cand_r3s736_lam10` **73321458, graded #337470 1.8708071209909215e-9** (forecast 1.8706e-9; 0/100 failed; served residual 0.3171 / 0.3256 s, wall 75.9 / 79.2 s; package `08927fdc`, verify PASS #3649) | 22,235 | **0.379 / 0.390** / 0.354 s | **+5.1 % / +2.4 %** / +11.5 % | 94.3 / 97.2 s | +21 % / +19 % | 1.228× (residual) |
| **B** `7a146f84` (#337462) | 18,943 | 0.310 / 0.319 / 0.302 s | **+22.4 % / +20.2 % / +24.6 %** | 66.4 / 68.5 s | +45 % / +43 % | **1.449× (residual)**; wall 1.979× |

Against the L3 rule (≥ 10 % headroom on the worst day on record): **B passes; A1 fails on residual** (wall is fine); **the λ 1e6 twin
does not pass either** — its own canary (#337470) read 14.26 µs/op, the same node rate as A1 (14.41), so −1.8 % ops buys 2.6 points
for +0.28 % score. ≥ 10 % central headroom needs ≤ 20,889 ops at A1's per-op rate (≤ 21,719 at
8be's). Leaders (G §3, wall only): 8 of 10 top entries fail under a smaller uniform slowdown than A1 (J2W 1.014×, andrei_bulzan
1.032×, mliston 1.071×, oliver_chen 1.085×, AndreasHad04 1.099×, megan_park 1.120×, Luna 1.145–1.161×); marius 1.222×,
thegamers 1.218×; **B outlasts all of them** — a slow node is a route up only through B.

**Residual-safe A candidate (planner sweep, `experiments/2026-10-21_L1/planner_sweep.json`): A1s** =
`experiments/2026-10-21_L1/src/cand_r3s736_lam30.py` sha **e823d62f** = A1 with `_SW_OP_FLOPS` 6e5 → 3e6 only (cap 6, pad kept):
289.5…290.9 G F (**+1.68 % vs A1**), **20,630 ops**, output max |Δ| vs A1 7e-6 (B's reorder level); verify_nominee #3652 PASS (bill exact, 2-process determinism,
rig 3 shapes). **Graded #337481 1.897391135725812e-9** (forecast 1.8970e-9; A1 × 1.0170, **B × 0.9484**), 0/100 failed, served
residual **0.2905 / 0.3023 s**, wall 70.2 / 74.0 s ⇒ **worst-day residual 0.348 s central (+13.1 %) / 0.358 s pess. (+10.6 %), wall
87.2 s (+27 %); first fails at 1.323×** — the only A-chain build meeting the ≥ 10 % rule. Package `a1s-r3s736-lam30-e823d62f.tar.gz`
sha256 `14f94773`. **Owed before it can be nominated: 100-call WSL memory, FRESH-A100.** Sweep: λ_op 2e6 20,952 ops +1.17 % F;
5e6 19,328 ops +3.81 % F (`experiments/2026-10-21_L1/RESULT.md` §5).

Smoke shapes (real client/server 0.12.3): A1 and B PASS at 256×4, 512×16, 1024×16, **1024×32** (0.44–0.49 B; ≈ 165–220 s served, would
exceed a 120 s wall) and 2048×4, setup ≤ 0.003 s. **Both crash (IndexError in `_yp_contract` via `_dslice21`) at 1024×16 with a
global weight scale ×0.5 or ×2** — the known S8.4 limitation (`RESULTS.md:3331`), now reached by a plain scale change; 8be
(#336993) crashes identically, and it passed the organisers' smoke, so that smoke does not use this scale. He-init nets unaffected.

## Nominee verification — `tools/verify_nominee.py` (Lane L1, 2026-10-20 20:15 UTC; no nomination/designation made)

Current official best and fallback: **#336993**, `experiments/2026-10-20_x1/estimator_8be.py`, sha `8be40031`, graded
1.8962023807611765e-9 (raw 1.4535553773953325e-8, served wall max 81.4 s, residual max 0.339 s, 0 failed). Rerun with
`uv run python tools/verify_nominee.py <source> --stages checkout,imports,static,rig,meter,project[,memory]` (through workerq).

| check | #336993 `8be40031` | X1-B module on 8be `0937b9cb` (not submitted) |
|---|---|---|
| committed copy, sha matches | PASS | PASS |
| imports: AST flopscope/whestbench/stdlib only; client-only venv (no numpy) import | PASS | PASS |
| contract: no bare/constant-returning except, no `budget_summary`, no RNG seeding/clock | PASS | PASS |
| real `flopscope-client` ↔ local `flopscope-server` 0.12.3: 1024×16 / 512×16 / 256×4 finite | PASS (97 / 32 / 2 s) | PASS (94 / 29 / 2 s) |
| determinism: 2 processes × 3 DEV nets bit-identical | PASS | PASS |
| real 0.12.3 bill, one count on DEV 0–7 | 286,868,957,368 F / 22,235 ops (= canonical; graded + 414,389 offset) | 287,276,100,542 F / 22,970 ops |
| served-equivalent residual (ref 0.3389 s × ops ratio) ≤ 0.37 s | PASS 0.339 s | PASS 0.350 s |
| served-equivalent wall (ref 81.4 s × wall-law ratio) ≤ 100 s | PASS 81.4 s | PASS 82.1 s |
| WSL RLIMIT_AS 8 GiB, 100 calls, real 0.12.3 stack | PASS: VmPeak 4,570 MiB, settled growth +0.0 MiB | not run |
| verdict | **PASS** (`experiments/verify/8be40031{,_memory}/`) | PASS without memory (`experiments/verify/cand_8be_x1b_0937b9cb/`) |

The module row is a dry run of the A2 path only. The module missed its pre-registered VAL wins criterion (56/100 < 60) and
passed FRESH-A100 (−0.591 % raw, 66/100). Using it is the strategist's decision (`experiments/2026-10-20_L1/RESULT.md`).

## Recommendation — 2026-10-16 13:50 UTC (Lane P10; grader on flopscope 0.12.3; no nomination/designation made)

The grader moved to flopscope 0.12.3 on Oct 16 (top-k `svd(k=)` re-priced, +3.0 % F for svd builds). Scores below marked "0.12.3-eq" are
the graded score × the exact 0.12.3/0.12.1 F ratio (RESULTS § "flopscope 0.12.3"). If the private re-run uses 0.12.3, only no-svd builds keep their numbers.

| role | file | sha8 | submission | score | raw | F/B | served wall max | served residual max | VmPeak |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| **A (recommended if v97 grades clean)** | `estimator_k3_v97_nosvd_l3e6.py` (no `svd(k=)`) | `c66c498d` | #336508 (grading) | ≈ 2.172e-9 projected (0.12.3) | 1.6916e-8 FRESH-A | 0.137400 | — | ≈ 0.312 s est. | WSL 5,241 MB |
| A (best under 0.12.1; uses `svd(k=)`) | `estimator_k3_v96_k94b_cl.py` | `0859a6e5` | #336453 | 2.1557624259471184e-9 (0.12.1); **2.2221e-9 0.12.3-eq** | 1.5928e-8 | 0.135346 → 0.13951 | 80.5 s | 0.317 s | WSL 5,397 MB (12 reps) |
| A (same code as v96, graded on 0.12.3) | `estimator_k3_v96b_k94b_cl.py` | `27279f83` | #336470 | 2.205696194615543e-9 (0.12.3) | 1.5810e-8 | 0.13951 | 87.6 s | 0.348 s | WSL 5,368 MB |
| **B (recommended)** | `experiments/2026-10-16_s10/estB2e_v96b_l4e6.py` (no `svd(k=)`) | `bc9e0c99` | #336495 | **2.2740565377012537e-9** (0.12.3) | 1.5899e-8 | 0.143035 | 62.7 s | **0.282 s** | VmPeak@100 5,228 MB |
| B (previous; uses `svd(k=)`) | lane-s `estB_v92_opfix.py` | `bfc727bc` | #336358 | 2.3133299624563775e-9 (0.12.1); 2.3825e-9 0.12.3-eq | 1.6615e-8 | 0.13923 → 0.143401 | 64.8 s | 0.281 s | VmPeak@100 5,144 MB |
| B alternative (fails B residual gate: 0.3085 > 0.30 s; no `svd(k=)`) | `experiments/2026-10-16_s10/estB2b_nosvd_k92f_cl.py` | `900a7fa2` | #336438 | 2.2735668306448326e-9 | 1.6308e-8 | 0.139412 | 64.5 s | 0.3085 s | VmPeak@100 5,196 MB |

## (superseded) Recommendation — 2026-10-15 21:45 UTC (Lane P9)

| role | file | sha8 | submission | score | raw | F/B | served wall max | served residual max | VmPeak |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| A (was recommended) | `estimator_k3_v93r_l11e6.py` | `39ae70ff` | #336312 | 2.2426514503021637e-9 | 1.6614e-8 | 0.134989 | 81.3 s | 0.313 s | 5,202 MB |
| A (alternative: best public, over the 0.33 s residual bound) | `estimator_k3_v93_pad1024.py` | `f921c80a` | #336282 | 2.234892068273354e-9 | 1.6618e-8 | 0.134484 | 83.3 s | 0.337 s | 5,246 MB |
| **B (current graded B)** | lane-s `estB_v92_opfix.py` | `bfc727bc` | #336358 (Oct 16 01:14 UTC) | **2.3133299624563775e-9** | 1.6615e-8 | 0.13923 | 64.8 s served | ≈ 0.252 s est. (not shown) | VmPeak@100 5,144 MB |

Known limitation (all K-stack builds since v87_B): planted non-finite α crashes `_dslice21` (IndexError) — reaches the grader as
a FAILED MLP, never on in-distribution He nets (RESULTS § P8, S8.4). Evidence: `RESULTS.md` § P9, `docs/STRATEGIST_PACKET_2026-10-15.md`.
**Samuel designates.**

## Recommendation — 2026-10-11 (Lane P6, plan rev 23; no nomination/designation made)

| role | file | sha8 | submission | score | raw | F/B | wall evidence | VmPeak |
|---|---|---|---|---:|---:|---:|---|---|
| **A (recommended)** | `estimator_k3_v82c_r1.py` | `66f3ee4a` | #335253 | **3.0464210089351763e-9** | 1.8454e-8 | 0.16508 | same code as v82b (constants only): A/B 0.953 vs v82a; law 76.5 / 90.0 s | 6,024 MiB |
| **B (recommended)** | `estimator_k3_v82b_bundlews.py` | `7148e22a` | #335250 | **3.1432860023588976e-9** | 1.8605e-8 | 0.16895 | A/B 0.916 vs v81b_ws → ≈ 78 s noisy-implied | 6,017 MiB |

**Update 03:10 UTC: v82c_r2 graded 3.032324364499572e-9 (#335298) and is now the recommended A** (`a648ba07`,
constants only vs v82c_r1; VmPeak 6,027 MiB). **Update Oct 12 15:15 UTC: v84_age6 graded 2.849755454236018e-9 (#335562) and is now the recommended A** (`c1b098c0`,
VmPeak 6,768 MiB, ≈ 94.9 s noisy-implied). B: v83_r5ws #335461 (2.9952e-9); v83_r7ws gated for the Oct 13 A/B slot.
**Update Oct 12 05:30 UTC: v83_r7od16 graded 2.8941891502782363e-9 (#335484) and is now the recommended A** (`852ccf04`,
VmPeak 6,135 MiB, ≈ 101.9 s noisy-implied).
**Update Oct 12 02:30 UTC: v83_r5od16 graded 2.908247724225528e-9 (#335455) and is now the recommended A** (`1d19eb78`,
VmPeak 6,132 MiB, wall ≈ 101.9 s noisy-implied). **B: v83_r5ws graded 2.9952404444999084e-9 (#335461) and is now the recommended B** (`b9e64076`, VmPeak 6,127 MiB,
≈ 78 s noisy-implied).
**Update Oct 11 19:05 UTC: v83_r4od16 graded 2.91912543907686e-9 (#335407) and is now the recommended A** (`e20cf029`,
VmPeak 6,129 MiB; wall ≈ 101.9 s noisy-implied, inside A ≤ 105 s but the thinnest margin we have shipped — B covers it).
**Update 13:20 UTC: v83_fold graded 2.9483303409388216e-9 (#335365) and is now the recommended A** (`c1adefbd`,
VmPeak 6,148 MiB, wall 1.012× v82c_r2 ≈ 91 s noisy-implied). Its ws twin is not yet built; B stays v82c_r2ws.
**B: v82c_r2ws graded 3.1017860563938384e-9 (#335319) and is now the recommended B** (`1c48dfb4`, VmPeak 6,022 MiB,
≈ 78 s noisy-implied). In flight: v82c_r2 (R2 refit, VAL −0.46 %
vs R1, constants only) on FRESH-A paired vs v82c_r1. **Samuel designates.**

## Recommendation — 2026-10-10 evening (Lane P5; no nomination/designation made)

| role | file | sha8 | submission | score | raw | F/B | wall evidence | VmPeak |
|---|---|---|---|---:|---:|---:|---|---|
| **A (recommended)** | `estimator_k3_v82b_bundle.py` | `a358fb45` | #335227 | **3.0716825837413753e-9** | 1.8607e-8 | 0.16508 | local A/B 0.953 vs v82a (0.934 vs v81d) → below v81d's ~101 s noisy-implied; law 76.5 / 90.0 s | 6,026 MiB |
| **B (recommended)** | `estimator_k3_v82b_bundlews.py` | `7148e22a` | #335250 | **3.1432860023588976e-9** | 1.8605e-8 | 0.16895 | A/B 0.916 vs v81b_ws → ≈ 78 s noisy-implied | 6,017 MiB |

**Update: v82c_r1 graded 3.0464210089351763e-9 (#335253) and is now the recommended A** (same code as v82b, refit
constants only; VmPeak 6,024 MiB; wall = v82b's). Its ws twin `estimator_k3_v82c_r1ws.py` is built but not gated.
**Samuel designates.**

## Recommendation — 2026-10-10 (Lane P5, plan rev 22; no nomination/designation made)

| role | file | sha8 | submission | score | raw | F/B | wall evidence | VmPeak |
|---|---|---|---|---:|---:|---:|---|---|
| **A (recommended)** | `estimator_k3_v81d_starall2.py` | `b7efb0f5` | #334948 | **3.3200317352723394e-9** | 1.883284e-8 | 0.17629 | same code as v81c (A/B 0.95 vs v81b; v81 ≈ v80's ~101 s noisy-implied); server wall max 90.0 s (Oct 10 board read) | 6,144 MiB |
| **B (recommended)** | `estimator_k3_v81d_starall2ws.py` once graded (Oct 10 A/B slot); until then `estimator_k3_v81b_star2ws.py` | `1a586019` / `512c33ec` | — / #334918 | — / 3.4711270541038904e-9 | — / 1.9265e-8 | 0.18030 / 0.18018 | block-40 twins (~85 s noisy-implied) | 6,166 MiB (v81b_ws) |

A = best graded score; B = its wall- and memory-safe twin (Strassen block 40). v82a (#335165,
bit-identical to v81d at −1.22 % F) replaces A here if it grades as projected (3.2795e-9).
Final A/B are decided on FRESH-LOCK (≤ 3 reads) plus wall/residual margins on Oct 16–17.
**Samuel designates.**

## Evidence update — 2026-10-06 night (no nomination/designation made)

| role | file | sha8 | submission | score | raw | F/B | wall evidence | VmPeak |
|---|---|---|---|---:|---:|---:|---|---|
| **A (candidate)** | `estimator_k3_v77_kash.py` | `4ff8018f` | #334487 | **3.959571663492616e-9** | 2.202053984490249e-8 | 0.17981 | local A/B vs v73 = 0.897 → ≈ 74.7 s median, **≈ 92.7 s noisy-implied** (A ≤ 105 s); law 83.5 / 98.1 s | 6,116 MiB |
| **B (graded)** | `estimator_k3_v76_kasws.py` | `fe05c0a4` | #334474 | 4.4952209455824486e-9 | 2.2018e-8 | 0.20416 | ≈ 62.6 s noisy-implied | 4,885 MiB |
| B (built, gated, not submitted) | `estimator_k3_v78_kashws.py` | `6b3db2a5` | — | projected 4.2260e-9 | DEV 2.2733e-8 | 0.19296 | local A/B vs v74_ws = 0.890 → ≈ 57.4 s noisy-implied | 4,687 MiB |

v78 = v77 at Strassen block 40, the matching wall-safe B; queued for the Oct 7 A/B slot. All
walls are measured-parent projections, not server reads. Read the server walls of v77 and
v76/v78 from the result pages before designation.

## Evidence update — 2026-10-06 evening (Lane P P2; no nomination/designation made)

| role | file | sha8 | submission | score | raw | F/B | wall evidence | VmPeak |
|---|---|---|---|---:|---:|---:|---|---|
| **A (candidate)** | `estimator_k3_v75_kas.py` | `d4d7885c` | #334464 | **4.1871242928610786e-9** | 2.201440164384394e-8 | 0.19020 | NEW code path: local A/B vs v73 = 0.951 → ≈ 79.2 s median, **≈ 98.2 s noisy-implied** (A ≤ 105 s); law 87.3 / 102.6 s | 6,388 MiB |
| **B (candidate)** | `estimator_k3_v76_kasws.py` | `fe05c0a4` | #334474 | **4.4952209455824486e-9** | 2.201782972832689e-8 | 0.20416 | local A/B vs v74_ws = 0.971 → ≈ 50.5 s median, **≈ 62.6 s noisy-implied** (B ≤ 95 s) | 4,885 MiB |

v75 = v73 + decided-neuron pruning (`tools/build_kas.py`); v76 = v75 at Strassen block 40. Both
walls are measured-parent projections (local ratio × a parent's measured server median), not
server reads. **v75's A margin is ~7 s on a noisy day**, so read its server wall from the
result page before it is designated. Previous pair (v73 #332893 / v74 #334434) below.

## Evidence update — 2026-10-06 (PLAN rev 18, Lane P P0; no nomination/designation made)

| role | file | sha8 | submission | score | raw | F/B | wall evidence | VmPeak |
|---|---|---|---|---:|---:|---:|---|---|
| **A (candidate)** | `estimator_k3_v73_gb.py` | `8fc42e77` | #332893 | **4.6198973531323855e-9** | 2.1514587515980564e-8 | 0.21473 | projected noisy 103.3 s (v72 measured median 83.30 s + 0.005 s); A policy ≤ 105 s | 6,724 MiB |
| **B (candidate)** | `estimator_k3_v74_ws.py` | `5aa60322` | #334434 | **4.976093977527745e-9** | 2.1515782080427925e-8 | 0.23128 | projected quiet 58.1 s / noisy 68.3 s from the wall law; B policy ≤ 95 s | 5,073 MiB |

v74_ws is v73_gb with `_STRASSEN_MIN_BLOCK = 64` and nothing else. DEV raw 2.227566e-8 vs v73's
2.227946e-8 (−0.017 %); 19,667 ops; residual 0.218 s; packaged estimator = tested source;
validate + validate-package OK. It replaces v68_ws #332544 (5.0664e-9) as the wall-safe B once
graded. Under PLAN rev 18 §2, designation is decided on FRESH-LOCK plus wall/residual margins
(Oct 14–16); this table is evidence only. **Samuel designates.**

## Evidence update — September 28 review (no nomination/designation made)

The best graded source is now **v72_gt**, SHA8 `3b6cc397`, submission **#332766**,
score **4.620737758949154e-9**. Live Chrome confirms100 completed/0failed; measured
wall median83.30295s, worst86.2848s, implied noisy worst103.295658s. Exact source/package
and WSL memory hashes match; VmPeak6702.67MiB. Meter472,202,879,721FLOPs,22,992ops;
projected residual from ops is approximately0.255s, not a new measured server residual.
v70 #332737 and v71 #332753 preceded it at4.665633e-9 and4.636794e-9.

The last graded wall-safe fallback is still **v68_ws #332544**, score5.0664e-9;
it has older accuracy and is not a v72 twin. No refreshed B or final nomination was
performed by this review. PLAN rev16 governs further work and all acceptance gates,
including105s noisy for a new A code path and95s for B. The older standing table below
is retained as historical evidence and is superseded by this update.

Full review and source attribution: `docs/research/2026-09-28_worker_evidence_review_codex.md`
and the September28 `RESULTS.md` addendum.

*Policy — PLAN rev 13, Samuel's decision 2 (2026-09-25):*
**A** = the best-scoring entry. Strassen leaf 32 is allowed; implied **noisy-day worst ≤ ~115 s**.
**B** = its wall-safe twin (leaf 64), implied **noisy-day worst ≤ 95 s**.
Re-run each designate on one quiet and one noisy day before Oct 10. Never spend
B's wall margin for score. **Agents never designate**; this file only keeps the
evidence current.

"Noisy day" = a run whose worst/median ratio is ≥ 1.15 (Sep 23 after ~08:00 UTC:
1.197–1.235). Quiet runs sit at 1.03–1.055. Implied noisy worst = graded median ×
1.24 (RESULTS.md § Calibration, wall refit).

## Why the pair is a free option — the rule, quoted

Live Rules v14 §7, read in Chrome 2026-09-25 (PLAN rev 13 decision 2):

> *"Where a Participant has more than one submission eligible for score-based prize
> consideration in a Phase, only that Participant's best-scoring such submission is
> ranked, so that each Participant occupies at most one position in that Phase's
> score-based ranking."*

**Fallback reading.** A net that exceeds the 120 s wall (or the 0.4 s residual) is
not a disqualification: it gets the zero-prediction fallback (§5.5), which scores
that net at roughly **0.9** instead of ~5e-9 — ruinous for *that entry's* mean,
but it touches only that entry. The other designate is ranked on its own, and
only the better of the two counts. So A may carry wall risk that B does not;
B exists so that a bad private-run day on A costs us A's edge, not the entry.

## Standing — 2026-09-27 00:40 UTC (v68_ws graded)

| role | file | sha8 | graded | score | worst wall (median) | implied noisy worst | residual | VmPeak | meets policy |
|---|---|---|---|---:|---|---:|---:|---:|---|
| **A (recommended)** | `estimator_k3_v68_wino.py` | — | #332478 | **4.7039e-09** | **93.86 (84.00)**, 1.117 | 104.2 s | 0.2521 s (1.59x) | 6,736 MiB | **yes, A (≤ 115 s)** — by ~11 s |
| A fallback | `estimator_k3_v67_s3.py` | — | #332440 | **4.7404e-09** | **94.44 (90.74)**, 1.041 | 112.5 s | 0.2566 s (1.56x) | v59 path | **yes, A (≤ 115 s)** — by 2.5 s |
| retired | `estimator_k3_v66_s2t2.py` | `5766053c` | #332409 | **4.7854e-09** | **101.10 (89.24)**, 1.133 | 110.7 s | 0.2635 s (1.52x) | v59 path | **yes, A (≤ 115 s)** — by 4.3 s |
| retired | `estimator_k3_v65_s2t.py` | `29a65bd2` | #332393 | **4.8134e-09** | **94.34 (89.27)**, 1.057 | 110.7 s | 0.2606 s (1.53x) | v59 path | **yes, A (≤ 115 s)** — by 4.3 s |
| retired | `estimator_k3_v64_s2.py` | `e92f2e7e` | #332350 | **4.9124e-09** | **94.47 (91.11)**, 1.037 | 113.0 s | 0.2531 s (1.58x) | v59 path | **yes, A (≤ 115 s)** — by 2 s |
| retired | `estimator_k3_v62_s1.py` | `d1d722db` | #332313 | **4.9347e-09** | **102.61 (90.27)**, 1.137 | 111.9 s | 0.2713 s (1.47x) | v59 path | **yes, A (≤ 115 s)** — by 3.1 s |
| retired | `estimator_k3_v59_corr400.py` | `f419f765` | #332232 | **4.9756e-09** | **94.42 (89.84)**, 1.051 | 111.4 s | 0.2511 s (1.59x) | 6,434 MiB (v57 path) | **yes, A (≤ 115 s)** — by 3.6 s |
| retired | `estimator_k3_v57_corr.py` | `253c7b19` | #332129 | **5.0443e-09** | 92.81 (90.43), 1.026 | 112.1 s | 0.2507 s | 6,434 MiB | yes, A |
| **B (recommended)** | `estimator_k3_v68_ws.py` | `0db9afba` | #332544 | **5.0664e-09** | **55.67 (51.89)**, 1.073 quiet | 64.3 s | 0.2063 s (1.94x) | 5,085 MiB | **yes, A and B** — by 31 s (B) |
| B fallback | `estimator_k3_v64_ws.py` | leaf64 of e92f2e7e | #332364 | **5.2699e-09** | **63.55 (55.27)**, 1.150 noisy | 68.5 s | 0.2252 s (1.78x) | 4,962 MiB | **yes, A and B** — measured on a noisy day |
| retired | `estimator_k3_v60_ws400.py` | `e9344107` | #332223 | **5.3387e-09** | 57.59 (55.73), 1.033 | 69.1 s | 0.2121 s (1.89x) | 4,941 MiB | yes, A and B, by 46 / 26 s |
| retired | `estimator_k3_v56b_wallsafe.py` | `b401ad75` | #332063 | 5.7076e-09 | 60.7 (54.5), 1.114 | ~68 s | 0.2188 s | 4,908 MiB | dominated by v60_ws400 |
| retired | `...v56c_mix__tr64.py` | `52d83de7` | #332088 | 5.4341e-09 | 80.30 (76.90) | 95.4 s | 0.2413 s | 6,468 MiB | superseded |

v59 is v60_ws400 with `_STRASSEN_MIN_BLOCK = 32` instead of 64 — same corrector
weights, same accuracy (v60's graded raw 2.301125e-08), v57's cost and code path.
Its derived score was 2.301125e-08 × 0.21625525 = 4.9763e-09; **#332232 graded
4.9756e-09** (raw 2.300780e-08), so A is now MEASURED. v57 #332129 stays the
fallback A (same code path, 1.4% worse).

**A's thin margin.** v57/v59's implied noisy worst (~112 s) clears the ~115 s A
bound by about 3 s and the 120 s cliff by about 8 s. That is the accepted risk
of decision 2, and it is exactly why B is v60_ws400 (51 s to the cliff).
The anchor is real: v55 (leaf 32, median 92.8 s, 2.5 s above v57's) read
114.5 / 113.8 s worst on the Sep 23 noisy day, with 0 failures.

## Re-run log

| date (UTC) | file | submission | day | worst | median |
|---|---|---|---|---:|---:|
| 2026-09-22 | v50 | #331863 | quiet | 88.7 | 84.4 |
| 2026-09-23 03:06 | v50 | #331939 | quiet | 87.7 | 85.0 |
| 2026-09-23 08:27 | v55 | #331967 | noisy | 114.5 | 92.8 |
| 2026-09-23 09:18 | v55 | #331980 | noisy | 113.8 | 92.5 |
| 2026-09-23 10:13 | v50 | #331989 | noisy | 104.1 | 87.0 |
| 2026-09-23 19:56 | v56b | #332063 | moderate (1.114) | 60.7 | 54.5 |
| 2026-09-24 00:11 | tr64 | #332088 | quiet (1.044) | 80.30 | 76.90 |
| 2026-09-24 08:30 | v57 | #332129 | quiet (1.026) | 92.81 | 90.43 |
| 2026-09-25 01:05 | v60_ws400 | #332223 | quiet (1.033) | 57.59 | 55.73 |
| 2026-09-25 04:05 | v59 | #332232 | quiet (1.051) | 94.42 | 89.84 |
| 2026-09-25 | v62_s1 | #332313 | moderate (1.137) | 102.61 | 90.27 |
| 2026-09-25 22:30 | v64_s2 | #332350 | quiet (1.037) | 94.47 | 91.11 |
| 2026-09-26 00:03 | v64_ws | #332364 | noisy (1.150) | 63.55 | 55.27 |
| 2026-09-26 04:50 | v65_s2t | #332393 | quiet (1.057) | 94.34 | 89.27 |
| 2026-09-26 07:10 | v66_s2t2 | #332409 | moderate (1.133) | 101.10 | 89.24 |
| 2026-09-26 09:20 | v67_s3 | #332440 | quiet (1.041) | 94.44 | 90.74 |
| 2026-09-26 13:20 | v68_wino | #332478 | moderate (1.117) | 93.86 | 84.00 |
