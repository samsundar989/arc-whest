# HANDOFF — Lane P6 (session 1, branch main), 2026-10-11: v83 = fold + kink-active H, the H frontier, spend items, hardening

**Repo and ownership.** Repo `C:\Users\samsu\Documents\arc-whest`, branch `main`. You own main, the estimator files, RESULTS, the ledger, `docs/DESIGNATION.md` (recommendations only; Samuel designates), CURRENT_STATE and the pod queue.

**Read first.** `docs/research/2026-10-10_evening_review.md` §0, §2, §5 and §6. This handoff follows P5, which is complete.

**Standing rules, unchanged:**
- PLAN rev 18 §5 hard gates:
  - ops ≤ 26,000;
  - projected residual ≤ 0.33 s;
  - wall ≤ 105 s for A and ≤ 95 s for B, noisy-implied;
  - setup < 2 s;
  - WSL 8 GiB PASS;
  - parity ≤ 1e-3;
  - hash = tested source.
- At most 5 canaries per UTC day plus 1 A/B slot.
- Record your prediction and the strategist's before every run.
- FRESH-A paired for any change that is not bit-identical.
- No fitting or selection on FRESH panels; FRESH-LOCK ≤ 3 reads, for designation only.
- No external messages. No `out=`, `where=` or symmetry tags.
- **No write-up work** until we are in the top 10.

## Why

v82c_r1 (#335253) scores 3.0464e-9 (raw ≈ 1.845e-8 × F/B 0.16508), rank 26. On the Oct 10 23:40 UTC board the cuts were 2.65e-9 for the top 20, 2.35e-9 for the top 15 and 2.05e-9 for the top 10, and they drift down daily.

The two Lane M5 items give about −7 % together. Exact pricing tonight found three more levers on the same column set:
- an asymmetric H (drop dead columns below α = −2.0, keep always-on columns up to α = 3.0). It keeps the hub-carrying columns and is 1.4 % cheaper than the planned τ2.5 cut;
- a deeper H cut (τ 2.0) worth −11.8 % F if its accuracy holds;
- per-shape Strassen depth, which now fits under the 26,000-op gate.

**Targets:** v83 ≈ 2.8e-9 by Oct 12; ≤ 2.65e-9 by Oct 16.

## Tool: exact prices for every lever

`docs/research/2026-10-11_p6price/` holds the strategist's tool. It contains `price.py`, plus the flopmodel interpreter with `linalg.svd` (top-k) and `linalg.eigh` added as flopscope 0.12.1 bills them.

**Usage:**
- `python price.py <variant>[,<variant>…]` writes `variants/est_<name>.py` (patched from `estimator_k3_v82c_r1.py`) and meters it in a fresh process.
- `--write-only` just writes the files, e.g. for a `twin.py eval --raw` screen.
- The base reproduces v82c_r1's real meter to the FLOP: 363,022,283,745 FLOPs, F/B 0.16508, 22,375 ops.
- Every anchor is checked to occur once. The table is in the folder's README.

**Key prices (F/B, ops):**

| variant | F/B | ΔF vs v82c_r1 | ops |
|---|---:|---:|---:|
| v82c_r1 | 0.16508 | — | 22,375 |
| fold | 0.16554 | +0.27 % | 22,618 |
| fold + H sym τ2.5 count (`fold+Habs`) | 0.15577 | −5.64 % | 23,097 |
| fold + H asym −2.0/+3.0 (`fold+Hasym2030`) | 0.15362 | −6.95 % | 23,386 |
| fold + H asym −2.0/+2.5 (`fold+Hasym2025`) | 0.15062 | −8.76 % | 23,386 |
| fold + H sym τ2.0 (`fold+Habs20`) | 0.14555 | −11.83 % | 23,360 |

**On fold+Habs (0.15577):**

| variant | F/B | ΔF vs fold+Habs | ops |
|---|---:|---:|---:|
| + per-shape depth, leaf ≥ 12 | 0.15371 | −1.32 % | 25,473 |
| + per-shape depth, leaf ≥ 16 | 0.15467 | −0.71 % | 25,167 |
| + age 6 | 0.16337 | +4.88 % | 22,592 |
| + exact SVD join | 0.15746 | +1.09 % | 22,233 |
| + J = 3 | 0.15614 | +0.24 % | 22,806 |
| + J = 4 | 0.15746 | +1.09 % | 22,515 |

## P6.0 — today (Oct 11)

a. **Commit the strategist's files unchanged, in one commit:**
   - `docs/research/2026-10-10_evening_review.md`
   - `docs/research/2026-10-11_p6price/` (its `.gitignore` excludes `variants/`)
   - `docs/research/2026-10-10_evening_tests/` (the D21-rank, cubature and complex64-billing scripts)
   - `docs/HANDOFF_2026-10-11_LANE_P6.md` and `docs/HANDOFF_2026-10-11_LANE_M6.md`
   - In CURRENT_STATE, mark P5/M5 complete and P6/M6 active.
b. **Merge lane-m into main** (a merge commit, as in 1a84c67).
   - **What it brings:** about 50 lane-m commits since 6acb6d6. They include the M2/M3/M5 experiments, packets and reports, `docs/RUNPOD_ACCESS.md`, `tools/twin/k4x_arm.py`, the fold/rad options in `tools/twin/twin.py` and `experiments/2026-10-10_m5/`.
   - **`twin.py` is the only file both branches changed.** Keep all of main's P5 code: the mask options `D2`, `D2c`, `Dleg`, `rho`, `J`, and the `--mask-stats` `e_all`/`e_kept` hook. Also keep M5's `--fold`/`--rad`.
   - **If the hook is lost, `compute_rho.py` silently writes ρ = 1.0.** After the merge, rerun it on FIT and check that it reproduces `experiments/2026-10-10_p5/d2/rho.json`.
   - Lane M then merges main back.
c. **A/B slot.** Gate and submit `estimator_k3_v82c_r1ws.py` as B insurance (the ws twin of A). Expected ≈ 3.12e-9.
d. **R2** (2 more epochs of the v82b arm, running on the pod since Oct 10 evening):
   - If its VAL best beats R1's by ≥ 0.3 %: bake v82c_r2, run FRESH-A paired vs v82c_r1, then canary.
   - Either way, start P6.1's refits from whichever of R1/R2 is better on VAL.
e. **`docs/DESIGNATION.md`** (recommendation): A = v82c_r1 #335253; B = v82b_ws #335250 until v82c_r1ws grades.

## P6.1 — v83 = v82c_r1 + fold + kink-active H by count (Oct 11–12, critical path)

**Step 1 — twin support** (main's twin after the merge):
- (i) **H ranking for the K-table path.** In the mask's `Ktab` block (`twin.py` ≈ 343–356) add `Hrank=abs` (score α²) and `Hrank=asym:a:b` (score max(−α/a, α/b)). Keep the K_H smallest scores. Today the K-table ranks by α only, so it can drop dead columns but never always-on ones.
- (ii) **`Hfull=1`** sets `_mSl = _mH`, so the P2 residual rows and S3 are dropped on the dropped columns too. **This is what the estimator ships:** its `_kh` gather removes `Lb` and `_sb` along with A_L, w2b, hubs and e_b. Lane M5's H2.5-literal arm kept the residual and S3 on always-on columns (`_mSl` = the dead-only mask), so its +1.35 % is not the shipped quantity.
- (iii) **Torch shim** (`tools/twin/shim/flopscope/numpy.py`, class `linalg`): add
  - `svd(a, full_matrices=True, compute_uv=True, hermitian=False, k=None)`: torch economy SVD, sliced to k;
  - `eigh`.
  - Lane M6.2 needs these. Detach the SVD basis in refits.

**Step 2 — unrefit screen, VAL-100** (`hold100`).
- Fastest route: `twin.py eval --raw --cand docs/research/2026-10-11_p6price/variants/est_<name>.py`. These are the exact files, with v82c_r1's R1 constants and g★ = `_STAR_GAIN`.
- Variants: `base`, `fold`, `fold+Habs`, `fold+Hasym2030`, `fold+Hasym2025`, `fold+Habs20`.
- Then the same arms through the twin mask (v82b arm + `--fold tau=2,mode=rec` + `H=…,Hrank=…,Hfull=1` with a K json = `aggr_q50.json` whose H row is the variant's table). This checks arm-vs-file parity before refitting.

**Step 3 — pod:** joint refit for the control (v82b arm), `fold+Habs`, and the best of the asymmetric/τ2.0 arms by the unrefit marginal rule.
- Recipe: corrector + gains + sg + fg, cosine schedule, lr ≤ 1e-4, 2 epochs each, matched.
- If ρ (d21e ratio, `experiments/2026-10-10_p5/d2/compute_rho.py`) moves > 2 % on the new chain, recalibrate it first.

**Step 4 — build v83** from the winning arm:
- The `fold` patch: `experiments/2026-10-10_m5/build_fold.py` with its third anchor changed to v82b's one-line `rec = _mm(fnp.concatenate([LA[:, c:], LP[:, c:]], axis=1), _R.T)`. `price.py`'s `fold` shows the exact replacement. Bake `_FOLD_GAIN` from `gain__fg` as `build_fold.py` does.
- The `habs`/`hasym` patch (H row of `_KAS_K` plus the ranking block at est ≈ 4115–4118).
- The refit leaves.
- Checks:
  - parity ≤ 1e-3 vs the twin arm (DEV-100, `--raw`);
  - the interpreter must reproduce the table's F/B and ops to the FLOP.

**Step 5:** FRESH-A paired vs v82c_r1, then canary.

**Predictions (strategist):**
- v83 with sym τ2.5: Δraw −1.0 to −2.0 % vs v82c_r1 (fold+H joint −1.3 %, two refit epochs −0.5 %, count + full drop +0 to +0.5 %); F/B 0.15577; **score 2.82–2.85e-9**.
- Asym −2.0/+3.0: 0.3–1 % better raw than sym τ2.5 **and** 1.38 % cheaper. It keeps the always-on hubs, which carried most of H's cost (Hk2.5 +1.37 % vs Hkh2.5 +0.09 %, unrefit), and drops more dead columns, which carry almost nothing. **Score 2.76–2.80e-9** if its raw lands at −1.3 to −2.5 %.
- Asym −2.0/+2.5: +0.5 to +2 % raw vs sym τ2.5 for −3.3 % F.
- τ2.0: +2 to +5 % raw for −6.6 % F.

**Accept** each on the marginal rule vs v82c_r1: −Δraw > ΔF/(1+ΔF).

## P6.2 — spend items from Lane M6.2 (Oct 12–14)

Integrate in the order the twin evidence arrives:

| item | ΔF on fold+H | break-even Δraw | evidence |
|---|---:|---:|---|
| age 6 | +4.88 % | −4.65 % | v53 measured −5.53 % |
| exact SVD join | +1.09 % | −1.08 % | unmeasured |
| J = 3 | +0.24 % | −0.24 % | unmeasured; J = 2 gave −1.14 % |
| J = 4, only if J = 3 clears its break-even by ≥ 0.5 % | +1.09 % | −1.08 % | unmeasured |

For each: price, twin parity, refit vs matched control, FRESH-A, canary.

**Measurements:**
- **Age 6:** VmPeak and wall. It added ~700 MiB on v53; B's projection must stay ≤ 7,800 MiB at 250 predicts.
- **SVD join:** wall. flopscope bills the top-k SVD at 4mnk, but numpy runs the full economy SVD, about 0.2 s per join on the server and 8 joins per net.

## P6.3 — per-shape depth (Oct 14–15)

Price `optdepth(12)` and `optdepth(16)` on the final stack. On fold+H:
- leaf ≥ 12 is −1.32 % F at 25,473 ops;
- leaf ≥ 16 is −0.71 % at 25,167 ops.

**No gate change is needed.**

**Build it only if:**
- wall A ≤ 105 s noisy-implied on DEV net 3 (×2, alternating);
- DEV raw stays within ±0.2 %, since deeper Strassen grows float32 error.

Not for B (block 40).

## P6.4 — refits (continuous, pod)

Queue, in order:
- **R3:** the P6.1 step 3 batch.
- **R4:** v84 with the accepted spend items.
- **R5:** a final polish at low lr.

Rules as in P5.4:
- every comparison uses a same-recipe control with matched epochs;
- select on VAL-100; accept only on FRESH-A paired;
- stop a recipe after two consecutive restarts that gain < 0.3 %.

Lane M's research refits fill gaps in the queue.

## P6.5 — adaptive keep counts (Variant B), only if time remains

About −1 % raw at ~0 F. Do not build "spend more on hard nets": per-net MSE CV is 12 % on FRESH-A, and the worst 10 nets hold 12 % of the error, so per-network budget allocation has little leverage.

## P6.6 — hardening and designation prep (Oct 16–17)

- FRESH-LOCK (≤ 3 reads in total) on the A and B finalists.
- B = the ws build of the final A (block 40). Projected VmPeak ≤ 7,800 MiB at 250 predicts in one WSL process under RLIMIT_AS 8,192 MiB.
- Package hashes = tested source.
- `docs/DESIGNATION.md` recommendation: A = best score; B = its wall- and memory-safe twin.
- **Freeze A at Oct 17 12:00 UTC.** Keep one slot for B insurance.

## Do not

- **Reopen closed items:**
  - κ₄-core loop, gain-mode κ₄ templates, exact dressing correction;
  - always-on D21 rows, alive-block covariance, compact-history carriers, learned pair closures;
  - leg rows on the D set, K16 = 800, old rebuild on D rows, covariance input pruning;
  - hub move, radial Sym(μ⊗κ₃), paid κ₄ pieces;
  - low-rank D21 readouts and spherical cubature (both tested Oct 10, §4);
  - rank 320 (+6.26 % F on v82c_r1; v67 net +2 %), P2 rank 8 (+0.93 %; v53 −0.12 % raw), `_SHARED_PASSES = 2` (+3.65 %; the SVD join gets an exact basis for +1.09 %).
  - Rank 288 is neutral, not closed: +3.47 % F against v67's −3.45 % raw nets about −0.1 %, inside the noise. Retest it only on top of an accepted SVD join.
- Merge the `_dslice21` call sites (+28 % wall).
- **Trap:** `_layer` assigns a local named `_sb`. Route new products through `_mm`.

## Report

- A daily paragraph in `docs/STRATEGIST_PACKET_2026-10-1X.md`.
- RESULTS rows with both predictions.
- The ledger of cumulative ΔF, Δraw and ops against v81d.
- A canary graded > 2 % worse than projected stops the line until it is diagnosed.
- Flag immediately if the selection UI opens, a flopscope release or re-evaluation lands, or an organizer post appears.
