# BENCHMARK.md — where we actually stand

**PRIVATE. The census in §3 is a screen, not an accusation. Names never leave
this repo; nothing here goes on the forum. Cf. RESEARCH_NOTES §2g rule 2.**

*Updated every session. Last update: 2026-08-06.*

Rank on the public board is not the prize — the prize is a private re-run on
fresh networks — but the board is the only external calibration we have, and it
is worth exactly as much as our ability to read it honestly. This file keeps
three things current: what our own instruments say (§1), what the field's
anchors are (§2), and how many teams above us plausibly got there by estimating
(§3).

---

## 1. Our numbers

| instrument | tier | value | date | notes |
|---|---|---|---|---|
| **D14 decision, Sprint-5 frozen carrier** | DECIDE | paired `V·C` **1.1216×** [1.1188, 1.1244] vs 1.15× gate | 2026-08-06 | **FAIL — packed lane killed.** Block spent; see §7 |
| public mini, **v2.5 packed candidate** | CONFIRM | **4.11e-7** adj / 1.48e-6 MSE / mult 0.284 / 0 failed | 2026-08-06 | 26% WORSE than v2.4.1 on the local backend. FLOPs 3.86e10/MLP (−28%) but residual 387.6 ms/MLP vs 70.8. Not promoted |
| public mini, v2.4.1 control (same day) | CONFIRM | 3.26e-7 adj / 1.49e-6 MSE / mult 0.222 / 0 failed | 2026-08-06 | re-measured alongside v2.5 |
| vault, 500 sealed MLPs | VAULT | median **2.4651e-7**, mean **3.2669e-7**, p99 **2.0535e-6** | 2026-08-06 | v2.4.1, touch #1; max 3.0469e-6; 0 non-finite; GT noise 0.09% of raw MSE |
| fresh-mint distribution, 1200 MLPs | DECIDE | median 2.19e-7, mean 3.01e-7, p99 1.65e-6, max/median 12.9× | 2026-08-05 | pre-provenance-fix mint (block L3) |
| public mini split (100) | CONFIRM | v2.4.1 **3.172e-7** adj / 1.4895e-6 MSE / mult 0.216 / 0 failed | 2026-08-06 | v2.4 was 3.400e-7 / 1.6079e-6 |
| public mini, N=65,536 candidate | CONFIRM | **3.15e-7** adj / 4.10e-7 MSE / mult 0.783 / 0 failed | 2026-08-06 | only ~0.7% better after real local residual billing; not adopted |
| **graded (submission 327743, v2.4.1 — THE SHIPPED FILE)** | graded | **2.6314e-7** adj / 1.3323e-6 MSE / mult 0.19896 / residual 10.022 ms/MLP / 0 of 100 failed | 2026-08-10 | **equivalent to v2.4, not better** — paired geo-mean 0.9665× [0.7969, 1.1721], 22/50, p = 0.839. Quote as a draw |
| graded (submission 324420, v2.4) | graded | 2.7194e-7 adj / 1.3748e-6 MSE / mult 0.1990 / residual 9.845 ms/MLP | 2026-08-05 | rank #90 of 200 at 2026-08-06 |
| graded (324094, v1.1) | graded | 3.893e-7 | 2026-08-05 | the diversified second selection |

**Private/public effective-compute ratio, the write-up's honest cost-divergence
claim, now has a fifth point:** 327743 reads **0.99198**, beside v2.4's 0.99157.

Reminders that keep these readable:

* The graded number is the mean over **50 fixed** MLPs (indices 0–49), so
  submission-to-submission comparisons are *paired re-measurements*, never
  population evidence, and the level carries a fixed 7–20 % offset vs the local
  100.
* The mini split is **CONFIRM tier** — read-only. It is quoted here, never
  optimised against.
* A single 100-MLP run cannot resolve a <15 % question: two realisations of the
  same estimator differ ~18 % in the mean. v2.4.1's 6.7 % edge over v2.4 on the
  mini split is luck, not method — the 32-draw paired measurement puts them at
  0.9959×.

---

## 2. Anchor table

| anchor | adjusted | what it is |
|---|---|---|
| GT noise floor (1e9-sample truth) | ~2e-10 | the physical floor of the metric |
| **reconstruction zone** | **< 1e-7** | below the best published honest method by >1.5×; see §3 |
| best *published* honest method (18106) | 1.551e-7 | structure-aware dead/on/kink hybrid |
| published RQMC + Rao-Blackwell (18053) | 4.10e-7 | |
| published variance/bias budget (18085) | 4.47e-7 | |
| **us (v2.4, graded)** | **2.719e-7** | |
| our declared honest sampling frontier | ~2.7e-7 | GAME_PLAN rev 6 |
| `sampling_mse` on the graded suite | 6.4695e-7 | identical to 17 digits across all our submissions; best reading is a reference full-budget plain-MC sampler |
| plain MC at full budget | ~2.8e-6 | |
| calibrated covariance propagation (our v0) | 2.4927e-6 | graded |

Two cautions carried forward. The `sampling_mse` **label-noise bound is
withdrawn** — 89 of 200 board entries sit below it, so it cannot be additive
reference noise. And 18106's 1.551e-7 is attributed to a family that our Sprint 3
measured end-to-end at ~1.02×, so either the method differs from its description
or the number reflects the board; trust local measurement.

---

## 3. Board-hygiene census — round 1428, snapshot 2026-08-06T10:03Z

Produced by `local_sim/board_census.py` from the archived snapshot. Criteria are
fixed in advance and stated so they can be argued with:

* **reconstruction-capable** — ≥257 entries, the number that over-determines the
  50 fixed references by RESEARCH_NOTES §2g D1.
* **late step-change** — a ≥3× improvement between consecutive personal bests
  *after* already being below 1e-6. Real methods compound in small steps once in
  the regime.
* **sub-frontier** — below 1e-7, i.e. beating the best published method by >1.5×.

A team is flagged **only** when sub-frontier **and** (capable **or** stepped).
Sub-frontier alone is not enough; capacity alone is not enough either.

| | count |
|---|---|
| entries on the board | 200 |
| probing-signature | **8** |
| plausible-honest | **192** |
| **our rank among plausible-honest** | **#82 of 192** (raw #90 of 200) |

Full census of every team above us (`write-up` means a method-bearing public
write-up was known at snapshot time):

| raw rank | participant | entries | largest consecutive-best jump | write-up | signature |
|---|---|---|---|---|---|
| 1 | joe_wanza | 942 | 8.40× | n | probing |
| 2 | dpskv5 | 276 | 8.76× | n | probing |
| 3 | ednacob | 111 | 2.56× | n | plausible-honest |
| 4 | dstepanov | 411 | 3.41× | n | probing |
| 5 | ely2sh | 245 | 2.92× | n | plausible-honest |
| 6 | huang_chung_yi | 425 | 1.53× | n | probing |
| 7 | thylinao | 82 | 4.02× | n | probing |
| 8 | SKIBIDI_TOILET | 365 | 1.29× | n | probing |
| 9 | abhinav_gorrepati | 250 | 1.25× | n | plausible-honest |
| 10 | jtel | 221 | 1.28× | n | plausible-honest |
| 11 | mliston | 460 | 1.88× | y | probing |
| 12 | (anon) | 500 | 2.77× | n | probing |
| 13 | (anon) | 156 | 1.77× | n | plausible-honest |
| 14 | (anon) | 651 | 2.08× | n | plausible-honest |
| 15 | kaileh57 | 326 | 2.20× | n | plausible-honest |
| 16 | (anon) | 182 | 1.99× | n | plausible-honest |
| 17 | andrei_bulzan | 144 | 2.24× | n | plausible-honest |
| 18 | andrew_epstein | 45 | 2.24× | n | plausible-honest |
| 19 | jamespayor | 120 | 2.90× | n | plausible-honest |
| 20 | sweaty_dog | 95 | 1.86× | n | plausible-honest |
| 21 | yanggan_gu | 29 | 1.26× | n | plausible-honest |
| 22 | alexander_luz | 59 | 2.02× | n | plausible-honest |
| 23 | fklassen | 165 | 1.43× | n | plausible-honest |
| 24 | burhan_ul_tayyab | 42 | 1.78× | n | plausible-honest |
| 25 | josu | 53 | 1.56× | n | plausible-honest |
| 26 | neilarmstrong4 | 7 | 1.08× | n | plausible-honest |
| 27 | (anon) | 89 | 1.44× | n | plausible-honest |
| 28 | ndrew1337 | 53 | 1.30× | n | plausible-honest |
| 29 | williawa | 186 | 1.22× | n | plausible-honest |
| 29 | (anon) | 54 | 1.38× | n | plausible-honest |
| 31 | konstantin_baltsat | 22 | 1.18× | n | plausible-honest |
| 32 | isaac_wooden | 12 | 1.27× | n | plausible-honest |
| 33 | jainam2 | 120 | 2.00× | n | plausible-honest |
| 34 | (anon) | 65 | 1.21× | n | plausible-honest |
| 35 | bgrubbs1984 | 198 | 1.76× | n | plausible-honest |
| 36 | ArtemVoronov | 46 | 1.29× | n | plausible-honest |
| 37 | frankie_c | 17 | 1.22× | n | plausible-honest |
| 38 | (anon) | 69 | 2.51× | n | plausible-honest |
| 39 | skye_nygaard | 9 | 1.23× | n | plausible-honest |
| 40 | ymym_tymblack | 7 | 1.60× | n | plausible-honest |
| 41 | (anon) | 162 | 1.59× | n | plausible-honest |
| 42 | ionel_chiosa | 37 | 2.00× | n | plausible-honest |
| 43 | (anon) | 101 | 2.61× | n | plausible-honest |
| 44 | shiv_m | 51 | 1.95× | n | plausible-honest |
| 45 | cc23 | 56 | 2.46× | n | plausible-honest |
| 46 | d0rb | 18 | 1.82× | n | plausible-honest |
| 47 | adrianleb | 36 | 1.17× | n | plausible-honest |
| 48 | pluto | 114 | 1.28× | n | plausible-honest |
| 49 | jeremy43 | 69 | 2.52× | n | plausible-honest |
| 50 | (anon) | 78 | 1.19× | n | plausible-honest |
| 51 | bin_yong_bong | 4 | 1.26× | n | plausible-honest |
| 52 | trim_qewas | 40 | 1.38× | n | plausible-honest |
| 53 | wizood | 8 | 1.90× | n | plausible-honest |
| 54 | mainguyenanhvu | 92 | 1.52× | n | plausible-honest |
| 55 | neilarmstrong | 2 | 1.11× | n | plausible-honest |
| 56 | jlacombe | 37 | 1.19× | n | plausible-honest |
| 57 | (anon) | 204 | 1.76× | n | plausible-honest |
| 58 | MichaelIbrahim | 25 | 1.18× | n | plausible-honest |
| 59 | (anon) | 49 | 1.52× | n | plausible-honest |
| 60 | guangtao_zeng | 22 | 3.89× | n | plausible-honest |
| 61 | mythos10086 | 108 | 1.10× | n | plausible-honest |
| 62 | ngcferreira | 161 | 1.35× | n | plausible-honest |
| 63 | amalgonim | 11 | 1.25× | n | plausible-honest |
| 64 | artist | 18 | 1.12× | n | plausible-honest |
| 65 | suliman_tadros | 76 | 1.56× | n | plausible-honest |
| 66 | The_Alchemist | 74 | 2.02× | n | plausible-honest |
| 67 | lode_dockx | 66 | 1.15× | n | plausible-honest |
| 68 | qi_zhang5 | 12 | 1.45× | n | plausible-honest |
| 69 | hmyuuu | 22 | 1.33× | n | plausible-honest |
| 70 | (anon) | 82 | 1.99× | n | plausible-honest |
| 71 | LorenzoTomaz | 3 | 1.01× | n | plausible-honest |
| 72 | (anon) | 31 | 1.16× | n | plausible-honest |
| 73 | na_ML | 49 | 1.34× | n | plausible-honest |
| 74 | seckin_adali | 9 | 1.39× | n | plausible-honest |
| 75 | il.miscusi | 33 | 1.52× | n | plausible-honest |
| 76 | bek_reko | 8 | 1.44× | n | plausible-honest |
| 77 | jason_matthew | 21 | 1.09× | n | plausible-honest |
| 78 | ilpo | 33 | 2.42× | n | plausible-honest |
| 79 | nkosi_ndwandwe | 49 | 1.96× | n | plausible-honest |
| 80 | (anon) | 61 | 1.74× | n | plausible-honest |
| 81 | al_jannico | 76 | 1.25× | n | plausible-honest |
| 82 | malone | 20 | 2.62× | n | plausible-honest |
| 83 | (anon) | 226 | 1.85× | n | plausible-honest |
| 84 | (anon) | 43 | 1.51× | n | plausible-honest |
| 85 | deeptanush_kapaka | 8 | 1.48× | n | plausible-honest |
| 86 | frederik_hytting_jorgensen | 36 | 1.21× | n | plausible-honest |
| 86 | mxa1811 | 165 | 1.54× | n | plausible-honest |
| 88 | german_alfaro | 43 | 1.42× | n | plausible-honest |
| 89 | CharliC | 76 | 2.46× | n | plausible-honest |

Our own row: **2.719e-7 on 5 submissions**, largest late jump 1.74×.
The submission count remains by far the lowest of any team near us.

Flagged rows (rank / entries / largest late jump):

| rk | entries | late jump | why flagged |
|---|---|---|---|
| 1 | 942 | 8.40× | capable and stepped, 4.10e-9 |
| 2 | 276 | 8.76× | capable and stepped, 5.80e-9 |
| 4 | 411 | 3.41× | capable and stepped, 5.90e-8 |
| 6 | 425 | 1.53× | capable, 7.50e-8 |
| 7 | 82 | 4.02× | stepped, 8.45e-8 |
| 8 | 365 | 1.29× | capable, 8.56e-8 |
| 11 | 460 | 1.88× | capable, 9.15e-8 |
| 12 | 500 | 2.77× | capable, 9.25e-8 |

**The screen is deliberately conservative, so #82/192 is an upper bound on how
many real methods beat us, not an estimate.** The full machine-readable record is
`experiments/board_snapshots/2026-08-06T1003Z_2026-08-06_phase1_resume_census.json`.
---

## 4. Phase-2 opening protocol (days 1–4)

The first days of round 1429 are the only quasi-honest window on a fresh suite:
reconstruction needs roughly 257 submissions against a *new* scored set, which at
even an aggressive cadence is several days. So:

* **Snapshot the board daily** for the first four days
  (`board_snapshot.py --label phase2-dayN`, then `board_census.py --round 1429`).
* Log our rank each day in the table below.
* **Flag the first probing signature** — the first sub-1e-7 entry, or the first
  ≥3× step-change below 1e-6. That timestamp marks the window closing and goes in
  the write-up's private forensics appendix.

| day | date | board top | our rank | our honest rank | first probing signature |
|---|---|---|---|---|---|
| 1 | — | — | — | — | — |
| 2 | — | — | — | — | — |
| 3 | — | — | — | — | — |
| 4 | — | — | — | — | — |

---

## 5. Competitor V·c decomposition (from write-ups, ~Aug 17)

Above the 0.1 multiplier floor, `adjusted = V·c/B` — per-sample variance × per-sample
cost. That makes `V·c = adjusted·B` the only cross-team comparable quantity, and
it is the thing to extract from any Phase-1 write-up that publishes enough detail.
**Only teams in the 1e-7 – 6.5e-7 band with a stated method are worth the
extraction**; below that the number is not a method's number.

| team | adjusted | V·c | method | source |
|---|---|---|---|---|
| us (v2.4) | 2.719e-7 | 7.4e4 | pruned + rotated Sobol RQMC | this repo |
| 18106 | 1.551e-7 | 4.2e4 | structure-aware dead/on/kink | forum |
| 319341 (regraded 0.10.0) | 1.599e-7 | 4.4e4 | packed activation propagation | forum |
| *(pending Aug 17 write-ups)* | | | | |

For reference, board #1's implied V·c is **1.5e3** — 28× below anything published
and 49× below us, which is the quantitative form of the §3 flag.

---

## 6. Sprint 4 — we reproduced 319341's arithmetic, independently

Submission 319341 is the only competitor number in the honest band whose mechanism
is described in enough detail to rebuild. Sprint 4 rebuilt it from first
principles — column ordering by firing rate, a cold prefix, row buckets on prefix
support, exact zero-rectangle skipping, gathered corrections, Strassen dense
blocks — and priced it under real flopscope 0.10.0.

| quantity | v2.4.1 (ours) | Sprint-4 candidate | 319341 (published) |
|---|---|---|---|
| counted FLOPs/sample | 3.253e6 | **2.359e6** | **~2.144e6** |
| effective compute/sample | depends on residual | **~2.762e6 preregistered; 4.72e6 local** | **2.374e6 graded** |
| `MSE × N` | **2.2525e-2 graded** (2.61e-2 D11) | 2.59e-2 D11 | 1.852e-2 graded |
| operations / MLP | 1,416 | 10,263 | unknown |

**The packing mechanism is reachable; the published effective-cost target is not
yet reached.** The earlier 2.359e6-versus-2.374e6 comparison mixed our counted
FLOPs with their effective compute. The open gaps are:

1. **`MSE × N`.** The like-for-like graded comparison is our 2.2525e-2 against
   their 1.852e-2, a **1.216×** per-sample variance advantage. The 1.40× figure
   comes from D11/local levels and crosses suites, so it is diagnostic only.
   Packing is exact and cannot explain the gap; antithetic measured 1.009×.
2. **Residual.** Packing needs 7.2× the operations and ~5× the data-dependent host
   round-trips, and flopscope bills `1e11 × uninstrumented seconds`. Whether
   319341's 2.374e6 is an *effective* figure: its published telemetry implies
   about 2.144e6 counted plus 0.219e6 residual-equivalent per sample. Before
   spending a graded run, reduce our operation count and sweep N so fixed costs
   are comparably amortised.

---

## 7. Sprint 5 — the arithmetic target is **reached**, counted-to-counted. The lane is still killed.

Sprint 5 did §6's homework: compressed the operation count, re-derived the bucket
design, and reopened N. Seed blocks D13 (design) and D14 (locked, single-use,
now spent). Full record in `experiments/LOG.md`.

**The counted-cost comparison, finally in matching units:**

| quantity | v2.4.1 | Sprint-5 frozen carrier | 319341 (published) |
|---|---|---|---|
| counted FLOPs/sample @ our N=16,384 | 3.2304e6 | **2.4597e6** | — |
| fixed FLOPs `F0` (N-independent) | 6.854e9 | 6.860e9 | — |
| marginal counted FLOPs/sample | 2.822e6 | **2.056e6** | — |
| counted/sample extrapolated to **N = 84,992** | — | **2.137e6** | **2.144e6** |
| `MSE × N` | 0.02554 (D14) / **0.022525 graded** | 0.02554 (D14) | **0.018520 graded** |
| operations / MLP | 1,417 | 10,230 | unknown |

**Within 0.4% of their counted arithmetic, at their sample count.** Sprint 4's
"2.359e6 vs 2.374e6" compared our counted number to their effective one; this row
is counted-to-counted and it lands. So the packing arithmetic is not the gap.

**D14 verdict: 1.1216× [1.1188, 1.1244] paired `V·C`, against a 1.15× gate. FAIL.**
Control and candidate MSE agreed to five significant figures (1.5586e-06), which
retires D12's phantom 1.2079× variance advantage for good.

**What every Sprint-5 lever was worth:**

| lever | what it bought | what it cost | net |
|---|---|---|---|
| batched depth-2 Strassen | 10,263 → 4,823 operations (−53%), bitwise identical | +1.07% counted FLOPs, **+8% residual** (3 of 3 sample counts) | **shipped off** |
| one-sort/segment partitioner | −630 operations, 210 shape reads → 180 value reads | ±0% residual | wash |
| one/two-bucket layouts | fewer operations | **+7% counted FLOPs**, up to 8× peak memory | worse |
| slot-loop gather | 122 MB → 32 MB peak array | +136 ms residual | memory tool only |
| N = 32,768 / 65,536 | fixed cost amortised 2–4× | `MSE × N` flat; residual 96% marginal in N | 1.11× ± 0.11 local, **1.002× CONFIRM** |

**The structural reason, and it is the write-up's result.** flopscope prices a
gather at 4 FLOPs/element. Packing's whole saving is substituting gathers for
dense multiply-adds. A random-access gather costs far more *wall time* per element
than a BLAS multiply-add, and flopscope charges wall time back at λ = 1e11. The
lane trades an operation that is cheap in the price table for one that is
expensive in physics — and the price table is not what gets billed twice.

**Where 319341's advantage actually lives:** not in arithmetic (we match it to
0.4%) and not in packing (packing is exact and moves no variance). It is
`MSE × N` — 0.018520 against our graded 0.022525, **1.216×** — a per-sample
variance mechanism we still cannot name. That is the open question worth the next
sprint, and it is a variance question, not a cost one.
