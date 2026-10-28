# L4 — joint content arm J (Oct 21)

Handoff: `_claude_snap/handoff_2026-10-19/HANDOFF_2026-10-21_L4_JOINT_CONTENT_ARM.md`. All times UTC.

## 0. Prediction and keep rule (strategist's, copied before any J number exists, 04:50)

- Prediction: −3.5 % raw at +5.4 % F ⇒ score-equivalent ≈ +1.9 % (80 %: −2 … +4 %). P(clears ≤ −1 % score-equiv) ≈ 0.15.
- **Keep rule:** score-equivalent Δ = Δraw (VAL-100 paired vs A1 export, best visit) + ΔF (real meter) ≤ −1.0 %,
  raw CI excluding 0, ≥ 65/100 wins ⇒ bake + λ, FRESH-A100, verifier (100-call memory on the pod), 100-fresh-seed sweep, canary ≤ 16:00.
  Otherwise write the table and stop. No partial compositions.

## 1. Composition (`src/`, 3-way merges with `git merge-file`, 0 conflicts each)

| file | sha256 (8) | what |
|---|---|---|
| `a1_export.py` | 73321458 | parent: R3 `full_fit_v1/val_step0736/estimator.py` (A1 without the λ line) |
| `base_8be.py` | 8be40031 | merge-1 base (exact 8be) |
| `est_8be_p2.py` | 52dc31e8 | merge-1 theirs: L3's shared gain source (U1 P2 hunks + nine G1 tables + `_RHO` face, all off at 0) |
| `step1.py` | da779190 | A1 + NEW10. Differs from est_8be_p2 only by A1's 52 fitted-table lines |
| `base_P2.py` | 6354caac | merge-2 base: `claude_u2b_math_review/P2_u1_forum.py` (P3's actual parent) |
| `theirs_P3.py` | f4851ac7 | merge-2 theirs: P3 = P2 + U2b nested-Dx (job 3444's source) |
| `J0_zero_r288.py` (= `step2.py`) | **ba62d253** | A1 + NEW10 + U2b, all new gains 0, rank 288: **the zero control** |
| `J_start_r320.py` | **7d62cff0** | J's start: `J0` with `_SHARED_RANK` 288 → 320 (one line). `whest validate` PASS |

step2 − step1 is exactly the 188 changed lines of P2 → P3 (the U2b block, `_U2B_LMIN = 9`, `_U2B_REPLACES_MF = True`,
four 16×2 tables at 0, `_U2B_TRAIN = False`). `_RHO` stays in the file at 0 and is not trained (it is not part of NEW10).

## 2. Corpus finding (matters for L2's A3 too)

"d8bext2048" is `cache/d8b/d8b_mean.npy` (2048 nets, seeds 770000+i; label sha 7c2904da). It is **not unused**:
R3's recipe (`b636_r3_multiepoch/recipe.py`: roster = full 0–899 + d8b 0–2047, 2 epochs × 368) trained A1 on it twice.
No other labelled extension corpus exists in `cache/`. So the 368 updates on TRAIN900 ∪ d8b2048 are a **third pass over
A1's own training nets**, not fresh nets. The handoff names that corpus, so J uses it unchanged, with a new order seed
(20261021) so the batches are not R3's epoch order.

d8b weights: R3's generator was WSL torch 2.8.0+cpu. Windows torch 2.13 does **not** reproduce its bytes (fixture 0/1 mismatch).
`l4_train.py` therefore checks every run against R3's generator parity hashes (nets 0, 1, 2, 777, 2047) and can generate via
an interpreter with torch 2.8 (`L4_D8B_PYTHON`, e.g. the pod image's system python).

## 3. Trainer

`l4_train.py` = L3's `l3_train.py` + the four U2b tables (16×2, rate 1e-3 as job 3444) next to NEW10 (rate 1e-2 as job 3472),
`_U1_TRAIN`/`_FZ_TRAIN`/`_U2B_TRAIN` on during training only, `_U2B_D4` cleared in the per-forward reset, the d8b corpus,
VAL 900–999 at 0/92/184/276/368 with per-net, per-layer and fitted-table logs. `--no-new --updates 0` gives the twin zero control.

## 4. Zero control and real bill

**Zero control: PASS.** `J0_zero_r288.py` (all new gains 0, rank 288) vs `a1_export.py` on the real numpy chain,
same process, 2 BLAS threads (workerq #3640): prediction sha256 identical on both nets
(900 `bae95fd4…`, 901 `46d8f080…`), and identical bills (286,868,957,368 F / 22,235 ops). Re-read on flopscope 0.12.3 (#3648): same hashes.

**Real bill, flopscope 0.12.3 / numpy 2.2.6, f64 read-only, nets 900 and 901 identical (#3648, `bills/bill_0123.jsonl`):**

| source | FLOPs | ΔF vs A1 export | ops | residual proj. (×15.24 µs, verifier's rate) | (×14.80 µs, A1 graded max) |
|---|---:|---:|---:|---:|---:|
| A1 export (λ 1e6) | 286,868,957,368 | — | 22,235 | 0.339 s | 0.329 s |
| J start (rank 320, new gains 0) | 292,214,932,424 | +1.864 % | 22,235 | 0.339 s | 0.329 s |
| **J live (`J_probe_live.py`: every new gain 1e-3, so every NEW10 and U2b path is on)** | **306,317,429,884** | **+6.779 %** | **24,580** | **0.375 s** | **0.364 s** |
| J live + λ 6e5 bake (A1's +398 ops, −0.260 % F) | ≈ +6.8 % vs A1 | | ≈ 24,978 | **≈ 0.381 s** | ≈ 0.370 s |

The zero-gain bill under-prices J: NEW10 and U2b are gated off at 0 and only bill once fitted. A fit moves every coefficient
(U1 3472: 130/144 moved; U2b 3444: all 48 gradients nonzero), so the live probe is J's real price.

**What this means before any fit:**
- Score-equivalent ≤ −1.0 % needs **Δraw ≤ −7.8 %** vs A1. The strategist's prediction is −3.5 % raw at +5.4 % F; at the measured +6.8 % F
  that prediction is **+3.3 % score-equivalent**. The best measured parts (rank −1.18 % at step 0, U2b −1.24 %, NEW10 −0.9 %) add to ≈ −3.3 %.
  Clearing needs a coupling effect more than twice the sum of the parts.
- Residual: ≈ 24,978 ops projects to 0.381 s at the verifier's rate (> the handoff's 0.37 s stop line) and 0.370 s at A1's graded rate (on it).
  §3 says "if over, stop", so even a raw clear would fail the bake gate unless the graded rate holds.

## 5. Compute route (05:15)

- Pod zrdhrgb88enc0w: `start` refused 04:42, 05:05, 05:07 ("not enough free GPUs on the host"); the other stopped 5090 pods
  (ex9g481cmcnch4, fvowdhk563zl6h, v0wocjeb1llckg) refused the same way. Creating a new pod was refused by the session's
  permission layer (billable action, awaiting Samuel). **Pod spend: $0.**
- Route taken: the local 5090 through workerq, behind L2's A3 (#3645, ETA ≈ 09:15). **J = workerq #3650**
  (`j_train.py` = L2's tested `a3_train.py` + the four U2b tables at rate 1e-3, `_U2B_TRAIN` edit, `_U2B_D4` reset;
  same roster(20260925), batches and VAL as A3, so J and A3 are a matched pair; d8b weights from the authenticated WSL
  generator; exact `--resume`). Expected: start ≈ 09:15, step 368 ≈ 13:00–13:30. That misses the 10:00 decision time,
  so the J canary window (≤ 16:00) holds only if J clears by a wide margin. In every case the per-visit table goes to the write-up.
- `l4_train.py` (seed-regenerated d8b, pod layout) stays as the pod route if a pod becomes available.

## 6. Pod run (06:08–)

Pod zrdhrgb88enc0w started at 06:08 (host freed); its container disk had been wiped, so it was reprovisioned:
bundle of all worktree sources (py, small json), `tools/twin`, `cache/twin/full_{meta.json,truth.npy}`, `cache/d8b/d8b_mean.npy`
(sha 7c2904da re-checked on the pod), full-net weights regenerated from seeds (fixture hashes nets 0/900 from `full_w.f32`).
PyPI timed out on this host for torch 2.13, so the run uses the **pod image's torch 2.8.0+cu128 / numpy 2.1.2** plus
whestbench 0.16.1 / flopscope 0.12.3. d8b weights come from the same torch 2.8 (R3's generator was torch 2.8.0+cpu) and match
R3's parity hashes at nets 0/1/2/777/2047 (checked at start). Launched 07:40: J0 (zero control, VAL0 only) and J (368 updates)
side by side. ≈ 27 s/update ⇒ step 368 ≈ 10:30. Local backup #3650 cancelled once J passed step 50.

**Twin zero control (VAL-100, pod torch 2.8):** J0 VAL0 = 1.513653e-8 vs A1's recorded VAL 1.513612e-8: paired +0.002 %,
48/100, max |Δpred| 2.4e-6. That is not byte-identical (it was on L3's torch 2.13 pod). The gap is the cross-torch-version f32
floor (≈ 1/700 of the smallest effect in this lane). The real-chain zero control in §4 is byte-identical. J's rows are paired
against A1's recorded VAL as the handoff asks; the offset +0.002 % is noted.

**J step 0 (rank 320, all new gains 0):** 1.495211e-8 ⇒ **−1.195 % raw vs A1, 85/100** (L3's ALL-R step 0: −1.18 %, 86/100).

### J per-visit table (VAL-100 paired vs A1's export VAL 1.513612e-8; `val_table.py pod_runs/J`; score-equiv = Δraw + 6.779 % F, the live real-meter bill)

| step | mean raw | mean ratio | paired Δraw vs A1 | wins/100 | 95 % CI | score-equiv |
|---|---|---|---|---|---|---|
| 0 | 1.495211e-08 | 0.98784 | −1.195 % | 85 | [−1.449, −0.940] % | +5.584 % |
| 92 | 1.506197e-08 | 0.99510 | −0.457 % | 56 | [−1.107, +0.193] % | +6.322 % |
| 184 | 1.479854e-08 | 0.97770 | −2.186 % | 79 | [−2.731, −1.641] % | +4.593 % |
| 276 | 1.462410e-08 | 0.96617 | −3.366 % | 96 | [−3.798, −2.934] % | +3.413 % |
| **368** | **1.454358e-08** | **0.96085** | **−3.901 %** | **99** | **[−4.255, −3.547] %** | **+2.878 %** |

Fitted-gain movement at 92 (max |g − init|): NEW10 `_YP_TPQ` 0.50, `_CL_C3` 0.43, `_TMET` 0.26, `_CL_CQ` 0.22, `_MARG2` 0.18,
`_MARG2_C2` 0.15, `_YP_TP` 0.09, `_D4_LAW_SCALE` 0.018, `_K4T1_CV` 0.006; U2b `_U2B_D4` 0.081, `_U2B_Q31` 0.081, `_U2B_TPQ` 0.061,
`_U2B_TP` 0.010. All tables have moved away from 0, so the live-path bill applies. Step 92 is the expected early-harm visit (R3 +0.75 % at 92).

Fitted-gain movement at 184 (max |g − init|): `_YP_TPQ` 0.79, `_CL_C3` 0.55, `_TMET` 0.39, `_MARG2` 0.21, `_MARG2_C2` 0.20, `_YP_TP` 0.19,
`_CL_CQ` 0.16, `_D4_LAW_SCALE` 0.038, `_K4T1_CV` 0.004; U2b `_U2B_D4` 0.14, `_U2B_Q31` 0.13, `_U2B_TPQ` 0.10, `_U2B_TP` 0.019 (still growing).
Per-layer VAL mean at 184: L8 4.341e-9, L12 9.024e-9, L16 1.480e-8.

Fitted-gain movement at 368 (max |g − init|): `_YP_TPQ` 0.84, `_CL_C3` 0.63, `_TMET` 0.44, `_YP_TP` 0.25, `_MARG2_C2` 0.21, `_CL_CQ` 0.19,
`_MARG2` 0.19, `_D4_LAW_SCALE` 0.045, `_K4T1_CV` 0.004; U2b `_U2B_D4` 0.17, `_U2B_Q31` 0.16, `_U2B_TPQ` 0.12, `_U2B_TP` 0.027.
Per-layer VAL mean, step 0 → 368: L8 4.292e-9 → 4.213e-9 (−1.8 %), L12 9.036e-9 → 8.800e-9 (−2.6 %), L16 1.4952e-8 → 1.4544e-8 (−2.7 %).
Run complete (`complete.json` passed, best = step 368, earliest strict best; peak CUDA 3.37 GB). Export `pod_runs/J/estimator_step368.py` sha f4a7c8cd.

## 7. Decision (10:50 UTC): NO-GO — J is not baked, no canary

Best visit = step 368: **Δraw −3.901 % vs A1 (99/100, CI [−4.255, −3.547] %)**, at **+6.779 % F** ⇒ **score-equivalent +2.88 %** (worse than A1).
Keep rule needs ≤ −1.0 %. The raw part clears its CI and wins tests easily; the price does not. Residual (≈ 24,980 ops ⇒ 0.37–0.38 s) would
also have sat on or over the 0.37 s line. Per the handoff: write the table and stop; no partial compositions.

Prediction vs outcome: predicted −3.5 % raw at +5.4 % F ⇒ +1.9 % score-equiv (80 %: −2 … +4 %). Measured −3.90 % raw at +6.78 % F ⇒ +2.88 %:
inside the interval; raw slightly better than predicted, price higher than assumed.

**Is the κ3/κ4 coupling real at the slice level? Mostly no — the parts add up.** Matched control L2 A3 (NEW10 alone on A1's export, same
roster(20260925) batches, same VAL, 368 updates) reads −1.418 % at 368. Attribution of J's −3.90 %:

| part | measured | source |
|---|---|---|
| rank 288 → 320 (no fit) | −1.195 % | J step 0 (this run) |
| NEW10 fitted | −1.418 % | L2 A3 step 368, matched recipe |
| U2b nested-Dx fitted | −1.24 % | job 3444 (alone, older parent/recipe) |
| **sum of parts** | **≈ −3.85 %** | |
| **J joint** | **−3.90 %** | this run |

The joint fit beats the sum by ≈ 0.05 pp, within noise. The large κ3/κ4 compensation seen at the exact-cumulant level (O9.5: both exact →
0.076×) does not show up as extra gain when the fitted readers are refit together at this budget. J's own fit (vs its step 0) is −2.73 %,
about A3 + U2b. For the write-up: the young-κ4 readers and the extra band rank are **additive, accurate (99/100) and FLOP-negative**.

## 8. Spend

Pod zrdhrgb88enc0w (RTX 5090 Secure, $1.19/h): started 06:08:10, stopped ≈ 10:49 (uptime 16,824 s = 4.67 h) ⇒ **≈ $5.56**.
No other pod was created or started successfully (four start attempts on other stopped pods were refused for capacity, $0).
Local: workerq #3640 / #3648 (CPU bills), #3643 cancelled, #3650 (local J backup) cancelled at start.

**Price split (real 0.12.3 bills):** rank 320 +1.864 % (§4), NEW10 live ≈ +0.50 % (A3's bill, 287,565,166,545 F incl. λ) ⇒ U2b live ≈ +4.4 %
(+≈ 2,000 ops). Score-equivalent per part ≈ rank +0.67 %, NEW10 −0.92 %, U2b ≈ +3.2 %. NEW10 is the only part that pays for itself here,
and it already shipped as A3 (#337495). U2b's nested-Dx reader buys −1.24 % raw for ≈ +4.4 % F at this planner.
