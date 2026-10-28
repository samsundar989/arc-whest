# L2 — A3 (NEW10, 368 updates from A1's export) → nominee A verification (Oct 21)

Handoff: `_claude_snap/handoff_2026-10-19/HANDOFF_2026-10-21_L2_A3_SECOND_STAGE_368.md` (rev 37).

## 0. Prediction and keep rule (written 04:50 UTC, before launch)

| | prediction | 80 % |
|---|---|---|
| VAL-100 paired raw vs A1's export (1.513612e-8) at step 368 | **−0.5 %** | −0.1 … −1.0 % |
| F vs A1 | +0.15 % (NEW10 ≈ +60 ops) | — |
| score vs A1 | ≈ −0.35 % ⇒ **A3 ≈ 1.859e-9** on A1 graded 1.8656e-9 | — |

**Keep A3 only if all four hold**, else A stays A1 and nothing is submitted:
1. VAL-100 paired vs A1's export at the selected visit (earliest strict best of 92/184/276/368; step 0 = fallback)
   ≤ −0.5 % raw, ≥ 60/100 wins, 95 % t-CI excluding 0;
2. FRESH-A100 paired vs A1 (#3607/#3624 predictions) ≤ −0.4 % raw, CI excluding 0;
3. served-equivalent residual with the real op count ≤ 0.37 s;
4. `tools/verify_nominee.py` passes.
Then canary A3 ≤ 12:00 UTC; graded within ±1.5 % of forecast ⇒ A = A3, else A stays A1.

## 1. Run

**Correction to the handoff premise.** "d8bext2048" is the d8b set: 2,048 labelled nets, seeds 770000+i,
labels `cache/d8b/d8b_mean.npy`, weights from the authenticated WSL generator. No other extension corpus exists.
**R3 (A1's parent fit) already trained on it**: its `train.jsonl` draws 4,091 d8b + 1,797 full = 5,888 =
736 × 8, i.e. two epochs of the same 2,944-net roster. So A3's 368 updates are a third pass over A1's own corpus,
not fresh nets. What is new relative to last night's null 112-update screen is the length (368) and the mix (that screen
used TRAIN900 only). Run as specified; the keep gates are unchanged.

- Parent `src/a3_parent.py` sha **da779190…** = 3-way merge (base exact 8be `gaingroups/merged.py`, ours A1's export
  `r3_local2_full_owner_binding/full_fit_v1/val_step0736/estimator.py` sha 73321458 → `src/a1_export_s736.py`,
  theirs `gaingroups/est_8be_p2.py` 52dc31e8). Clean merge: A1's 52 table lines + the 429 P2 lines; nine G1 tables at 0.
- Trainer `src/a3_train.py`: L3's `l3_train.setup` unchanged (parser-repair adapter, NEW = the nine G1 tables, rate 1e-2,
  `_U1_TRAIN`/`_FZ_TRAIN` on; no `_RHO`, no rank change) + the pair stack's `roster(20260925)` (full 0–899 ∪ d8b 0–2047,
  368 × 8, 4 omitted) and `Data` (d8b generator verified at start). Fresh Adam, corrector LR ×10, cosine 368,
  VAL full 900–999 at 0/92/184/276/368, earliest strict best, step 0 = fallback. Exact resume from a per-update checkpoint.
- VAL0 must reproduce A1's VAL (1.513612e-8, per-net) or the run is void.
- **#3638 (04:51 UTC): VAL0 = A1 PASS** — predictions byte-identical to R3 `val_step0736/predictions.npz` (max|Δ| 0),
  per-net MSE identical, mean 1.5136123e-8. Then FAILED at the first d8b request: the helper exited `lease_expired`
  (stale `lease.tmp` ⇒ the client heartbeat thread died on a transient Windows replace while VAL0 ran). Fix (commit after 8d95a7b):
  the lease write retries; a failed d8b request reopens and re-verifies the generator (stateless, sha-checked payloads).
  **Resumed from the step-0 checkpoint as #3645** (the resume path re-ran VAL0: same 1.5136123e-8). ≈ 33 s/update, no generator restart.

### A3 VAL table (paired vs A1's export; `python src/val_table.py a3_run`)

| step | mean raw | mean ratio | paired Δraw vs A1 | wins/100 | 95 % CI |
|---|---|---|---|---|---|
| 0 | 1.513612e-08 | 1.00000 | +0.000 % | — | — |
| 92 | 1.532615e-08 | 1.01255 | +1.274 % | 35 | [+0.702, +1.845] % |
| 184 | 1.513273e-08 | 0.99978 | +0.001 % | 47 | [−0.477, +0.478] % |
| 276 | 1.499746e-08 | 0.99084 | **−0.918 %** | **70** | [−1.283, −0.554] % |

**Step 276 clears gate 1** (≤ −0.5 %, ≥ 60/100, CI excl. 0). **Provisional A3 (07:05 UTC)** = `estimator_step276.py`
(sha f92ab76b → `src/a3_export_s276.py`) + λ line ⇒ `src/cand_a3s276_lam06.py` sha **7b3d722b…** (one-line diff, count asserted = 1),
`whest validate` PASS, commit 57cc543. Gates started early on CPU: FRESH-A100 paired vs A1 **#3653**, verify_nominee
checkout/imports/static/rig/meter/project with A1's graded report as the projection reference **#3656**. Selection stays VAL-only:
if 368 strictly beats 276, rebake from 368 and rerun the gates.

**Verifier #3658 on 7b3d722b (07:45 UTC): VERDICT PASS** (`verify_a3s276/verify.md`) — checkout/imports/static (one WARN, the same
`_pkk_fast_ok` literal handler as A1), rig 1024×16 111 s / 512×16 36 s / 256×4 2 s finite, determinism 2 processes × 3 nets
bit-identical, one bill **287,565,166,545 F / 23,374 ops**, served-equivalent residual **0.3459 s** (≤ 0.37 ✔), wall 84.2 s.
(#3656 was cancelled: it had defaulted to a GPU request behind the trainer; resubmitted CPU-only.)

**The price is ~8× the handoff's estimate:** F **+0.504 %** vs A1 (predicted +0.15 %), ops **+741** (+3.3 %; predicted ≈ +60).
So at VAL −0.918 % raw the score-equivalent is ≈ **−0.42 %**, not −0.77 %. On L1's worst-day method (graded median 0.3261 s × ops
ratio × worst tail 1.196) A3's central worst-day residual is **0.403 s, headroom −0.7 %** (pess. 0.414 s); a uniform slowdown of
**1.157×** on A1's graded node fails it (A1: 1.194×, B: 1.449×). Gate 3 as written (≤ 0.37 s) passes; this is a margin finding
for the strategist, not a gate failure.

| 368 | 1.492264e-08 | 0.98590 | **−1.418 %** | **85** | [−1.665, −1.171] % |

**#3645 complete (07:52 UTC)**: 368 updates, `best_val` = step 368 (1.4922637e-8), no generator restart, peak CUDA 3.27 GB.
Step 368 strictly beats 276 ⇒ **A3 rebaked from step 368**: `estimator_step368.py` sha 49179dfc → `src/a3_export_s368.py`, + λ line ⇒
`src/cand_a3s368_lam06.py` sha **5ddf3290…** (count asserted = 1), `whest validate` PASS. Prediction was −0.5 % (80 %: −0.1 … −1.0);
**outcome −1.418 % is outside the interval, better.** Gate 1 PASS (≤ −0.5 %, 85/100, CI excl. 0).
s276 FRESH #3653 cancelled before it started (it too defaulted to a GPU request). Gates on 5ddf3290: FRESH-A100 paired vs A1 **#3661**,
verify_nominee **#3665** (CPU-only; #3665 needed a local `.git/info/exclude` for the d8b generators' `lease.json` — L4's live lease
broke workerq's `git add` snapshot).

**Verifier #3665 on 5ddf3290 (08:10 UTC): VERDICT PASS** (`verify_a3s368/`) — same bill as s276 (tables only): **287,565,166,545 F /
23,374 ops** (+0.504 % F vs A1), determinism bit-identical, rig 98 / 31 / 2 s finite, served-equivalent residual **0.3459 s**, wall 84.2 s.
WARNs: DESIGNATION listing (until graded) and A1's `_pkk_fast_ok` literal handler. ⇒ gates 3 and 4 PASS. Package
`a3-s368-lam06-5ddf3290.tar.gz` sha256 924130df…, validate-package PASS (release dir commit 2c125e3). FRESH #3661 running.

**FRESH-A100 #3661 (08:43 UTC; exit 2 = ev.py's "not accepted" under `--no-cost`, as A1's #3607):** A3 raw **1.522384e-8** vs A1
1.547390e-8 (`fresh_a_a3s368.json` vs `2026-10-20_L2/fresh_a_s736.json`, both n = 100, 0 failed): mean ratio 0.98384, paired
**−1.625 % [−1.856, −1.394]**, Δ 2.501e-11 t95 [2.137e-11, 2.864e-11], **89/100**. ⇒ **Gate 2 PASS** (≤ −0.4 %, CI excl. 0).

**All four keep gates PASS ⇒ A3 is kept; canary.**

### Canary forecast (written 08:50 UTC, before submission)
Score = A1 graded 1.8656192e-9 × FRESH raw ratio 0.983840 × bill ratio 287,565,166,545 / 286,123,309,398 (1.005039)
= **1.8447e-9** (−1.12 % vs A1). Keep window ±1.5 %: **1.8170 – 1.8724e-9** ⇒ A = A3; outside ⇒ A stays A1.
Expect 0/100 failed, served residual max ≈ 0.346 s.
Margin caveat for the strategist (above): A3's worst-day central residual 0.403 s (A1 0.390 s, B 0.310 s).

### Canary A3 — AIcrowd #337495 (submitted 08:45:29Z, ledger 5/10)

**GRADED 1.8527052525413123e-9** (raw 1.416773756e-8, multiplier 0.1307693, 0/100 failed; `experiments/run_reports/a3_337495_graded_report.json`).
vs forecast 1.8447e-9: **+0.43 %, inside ±1.5 % ⇒ A = A3.** vs A1 #337456: **−0.69 %** score, −1.19 % raw, +0.50 % F.
FRESH predicted −1.625 % raw; graded raw is −1.19 % (public panel, one draw).

| | A1 #337456 | A3 #337495 |
|---|---|---|
| served F per MLP | 286,122,895,009 | 287,564,751,853 (= bill − 414,692) |
| residual median / max | 0.3261 / 0.3349 s | 0.3331 / 0.3447 s (14.25 µs/op) |
| wall median / max | 78.0 / 83.2 s | 78.4 / 85.9 s |
| worst-day residual (L1 method, own median × 1.196 / × 1.029) | 0.390 / 0.401 s | **0.398 / 0.410 s** (+0.4 % / −2.4 % headroom) |
| uniform slowdown to first fail | 1.194× | **1.160×** |

A3's graded per-op rate (14.25 µs) matches the λ 1e6 twin's (14.26), so the pre-grade 0.403 s used A1's slower node; on its own grade the
central worst day is 0.398 s — still short of L1's ≥ 10 % rule, like A1. **The nominee-pair choice (A3 vs A1 vs A1s, against B) is the
strategist's**; this lane only reports that A3 is the best graded score and the thinnest residual margin of the A candidates.

**Superseded by D4 (read 09:25 UTC, `docs/DESIGNATION.md` § Provisional pair, `DECISIONS_2026-10-20.md`).** The pair is chosen on
worst-day headroom: A-safe + B, A1 not nominated. A3 at λ 6e5 has *less* headroom than A1, so under D4 **A3 is not a nominee** —
the handoff's "A = A3" keep step is overtaken. A3's role is the base of **A3s** (A3 s368 + `_SW_OP_FLOPS` 3e6–4.5e6), which L1 bills
and memory-tests on the pod; L1's 09:20 reading keeps **A-safe = A1s** (A3s 3.5e6 only 0.24 % better, 2.6 points less headroom).
A3's row is in DESIGNATION as "not nominated (base of A3s)". The local WSL memory gate #3667 on λ 6e5 was terminated a few minutes after it started (no reading)
(redundant with L1's A3s pod pair; avoids WSL contention with L6's #3660).

**Residual note (06:00 UTC, from L1 §2).** A1 has only +2.5 % central worst-day residual headroom (fails L1's ≥ 10 % rule). NEW10
adds ≈ +60 ops (+0.3 %) ⇒ ≈ +0.001 s: passes gate 3 (≤ 0.37 s) but trims that margin further. If A3 is kept, the strategist should
also see an A3 bake at λ 1e6 (−398 ops, the twin L1 is canarying) beside the λ 6e5 bake; the gate rule here stays as written.

Step 92 (05:42 UTC) is the early-harm phase at peak LR (R3 itself read +0.75 % at its step 92; the null 112 screen never left this phase).
