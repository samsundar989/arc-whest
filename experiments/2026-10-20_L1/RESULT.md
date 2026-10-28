# L1 — X1 arm B as a post-hoc module on the served chain + nominee verifier

Spec: `_claude_snap/handoff_2026-10-19/HANDOFF_2026-10-20_L1_X1B_POSTHOC_VERIFY.md`. X6 cancelled (J_cov_faces.md), not built.

## 0. Predictions on record (written before any read)

| read | prediction | keep if |
|---|---|---|
| VAL-100 paired, real chain, corrector-on, module on 8be | −0.4 % raw (80 %: −0.2 … −0.6 %) | score-equiv (raw ⊕ real bill) ≤ −0.2 %, CI excludes 0, ≥ 60/100 wins |
| FRESH-A100 paired | −0.35 % raw | CI excludes 0 |
| module re-trained on A1's chain | within ±0.15 pp of the 8be number | same gates; else drop A2 |

Stop line: VAL score-equiv > −0.2 % ⇒ stop the module lane, go to the verifier. No tuning.

## 1. Retrain and seed selection (job 3545, 10 epochs, TRAIN 0–849, hold 850–899)
Same trainer (`experiments/2026-10-20_x1/train.py --arms B --seeds 0,1`), same dumps (`cache/x1/train`, the
real 8be chain via the cupy shim), 6,530 params. Selection on the hold slice only:

| seed | best hold Δ (rows 850–899) | selected |
|---|---|---|
| B0 | −0.233 % | |
| **B1** | **−0.366 %** | ✔ |

Export: `trainB/B1.pt` + `trainB/norm.pt` → base64 float32 constants (`_X1B_B64`, `_X1B_R`), constants sha
**`ebee87ffc98d71479d9f5855a9d845015ab92889e5f5e71208a7eb6496729602`**. Builder `build_x1b.py`.

## 2. Integration (`src/cand_8be_x1b.py`, sha `0937b9cb…`; zero control `src/cand_8be_x1b_s0.py`, sha `790146d2…`)
- Recorder at the existing `_CORR_HOOK` site captures, per layer, exactly what `dump.py` recorded (mu_pre, σ, D3,
  D4, **uncorrected** mu_out, ‖W[:,i]‖, D21 energy); after the chain's two output guards, `_x1b_apply` runs the
  recurrence in `fnp` float32 and changes **only the scored final row**: max(0, b₁₆ + s·u₁₆). Non-finite
  entries fall back to the chain's own row; any shape other than 1024×16 returns the chain unchanged.
- `_X1B_S = 0.0` issues **no** operation (the recorder is never armed, the tail is the parent's) — the zero
  control is the parent's code path, not a numerically small perturbation.
- One deviation from the trained module: GELU in its tanh form (|err| ≤ 3e-4 vs x·Φ(x)). The exact form via
  `stats.norm.cdf` billed 504 MF/net (203 MF of it the cdf); the tanh form bills 342 MF. Parity of the served fnp
  module vs the torch module on the same dumped trajectories (VAL900/901): **max |Δ final| 1.2e-7** against
  corrections of ≤ 5.5e-5; per-net MSE agrees to 1e-5 relative. Rows 0–14 bit-identical to the chain.
- `whest validate`: PASS.

### Zero control and whole-file real bill (job 3560, `zero_and_bill.py`, `zero_and_bill.json`)
Real flopscope 0.12.3, float64 read-only weights, parent (8be40031) vs s = 0 vs s = 1, each its own BudgetContext.

| net | s=0 vs parent max \|Δ\| | s=0 bill == parent | s=1 rows 0–14 identical | s=1 max \|Δ final\| | parent F / ops | s=1 F / ops |
|---|---|---|---|---|---|---|
| VAL900 | **0** | ✔ | ✔ | 5.48e-5 | 286,868,924,600 / 22,234 | 287,276,067,774 / 22,969 |
| VAL901 | **0** | ✔ | ✔ | 5.09e-5 | same | same |
| FRESH-A 0 | **0** | ✔ | ✔ | 5.77e-5 | same | same |
| FRESH-A 1 | **0** | ✔ | ✔ | 3.92e-5 | same | same |

**Zero control PASS.** Whole-file Δ: **+407,143,174 F (+0.1419 %)**, **+735 ops** (module 610 + recorder 125). The
harness here skips the canonical `materialise_predictions` op, so its parent reads 32,768 F / 1 op under the
official canonical 286,868,957,368 F / 22,235 ops; the delta is unaffected. Served-equivalent residual
0.338924 s × 22,970/22,235 = **0.350 s** (≤ 0.37 ✔). Memory: + ~16×7×n floats of state, negligible.

## 3. Reads
### VAL-100 paired, served module (real flopscope 0.12.3 arithmetic) on the real numpy-chain trajectories
`read_module.py`, `val_served_B1.json`. Chain alone mean raw MSE 1.53509e-8 (L16 label floor ≈ 7.7e-11; paired, so it
cancels to first order).

| read | Δraw (ratio of means) | paired t95 | bootstrap 95 % | wins | ΔF (module only, real meter) | score-equiv |
|---|---|---|---|---|---|---|
| VAL-100, B1, s = 1 | **−0.437 %** | [−0.755, −0.119] % | [−0.748, −0.126] % | **56/100** | +0.142 % (whole file, real meter) | **−0.296 %** |

Gate: score-equiv ≤ −0.2 % ✔ · CI excludes 0 ✔ · **≥ 60/100 wins ✘ (56)**. The stop line (score-equiv > −0.2 %)
is not hit, so the FRESH-A read proceeds; the wins criterion is recorded as **missed**, not reinterpreted.

### FRESH-A100 paired (job 3572, `fresh_a_served_B1.json`) — the acceptance panel, never fitted or selected on
Same served module on the real numpy-chain trajectories of 8be on FRESH-A 0–99 (dumps 3537/3538 + 3552/3553/3568/3569).
Labels N = 1e8 (L16 floor ≈ 7.5e-10, ≈ 5 % of raw; paired, so it cancels to first order).

| read | chain raw MSE | Δraw | paired t95 | bootstrap 95 % | wins | ΔF (real) | score-equiv |
|---|---|---|---|---|---|---|---|
| FRESH-A100, B1, s = 1 | 1.57023e-8 | **−0.591 %** | [−0.901, −0.281] % | [−0.889, −0.292] % | **66/100** | +0.142 % | **−0.450 %** |

Gate: CI excludes 0 ✔. Prediction −0.35 % raw ⇒ outcome −0.59 % (inside the VAL 80 % band, better than the point).

## 5. Nominee verifier (`tools/verify_nominee.py`)
Stages: committed-copy checkout + sha · AST import scan + isolated import in a flopscope-client-only venv (no numpy) ·
grader-contract AST checks · real client ↔ local flopscope-server 0.12.3 at 1024×16 / 512×16 / 256×4 (immutability and
shape-generic by construction) · real 0.12.3 meter in-process, two processes, DEV 0–7 bill + determinism · served-equivalent
residual/wall projection from #336993's graded telemetry · opt-in WSL RLIMIT_AS 100-call memory.
**#336993 (8be40031): PASS on every stage** (jobs 3562, 3574): bill 286,868,957,368 F / 22,235 ops = canonical (graded
+414,389), bit-deterministic, VmPeak 4,570 MiB, growth +0.0 MiB, residual-eq 0.339 s, wall-eq 81.4 s. Module candidate
0937b9cb: PASS (no memory stage), job 3575. Table in `docs/DESIGNATION.md`. Bugs found while building it: none in the
candidates; the tool's own Windows temp-dir cleanup (fixed).

## 4. Verdict on 8be (prediction vs outcome)
| read | predicted | outcome | gate |
|---|---|---|---|
| VAL-100 | −0.4 % raw (80 %: −0.2 … −0.6) | −0.437 % raw, −0.296 % score-equiv, 56/100 | score-equiv ✔, CI ✔, **wins ✘ (56 < 60)** |
| FRESH-A100 | −0.35 % raw | −0.591 % raw, −0.450 % score-equiv, 66/100 | CI excludes 0 ✔ |

**Status: pre-registered VAL keep rule missed on one of three criteria (wins); FRESH acceptance passed.** The handoff's
stop line (score-equiv > −0.2 %) is not triggered, so the lane continues to the A1 re-dump. Whether a module that
missed the VAL wins criterion may ride in A2 is the strategist's call, not this lane's (flagged in RESULTS.md).
Size of the effect for scale: −0.45 % of score is below the 1 % claimability line of VALIDATION_PROTOCOL §1.3,
so it can only ever ride inside a composition, never justify its own canary.

## 6. Lean module (op diet; `build_x1b.py` now builds this form, so the A1 pipeline uses it)
Same trained B1 weights; algebraically identical rearrangement: GELU computed as 2·GELU in 7 ops (the ½ folded exactly into W2/W3, a
power-of-two scale), W∘W formed once in the recorder and reused by the recurrence, the per-layer sqrt / 1/σ⁴ of the recorded channels
batched over layers. Constants sha `7bef89d2…`; `lean/cand_8be_x1b_lean.py` (f96c853f), zero control `lean/cand_8be_x1b_lean_s0.py`.

| check | original (0937b9cb) | lean (f96c853f) |
|---|---|---|
| fnp vs torch, VAL900/901 max \|Δ final\| | 1.2e-7 | 1.3e-7 |
| VAL-100 served read | −0.43698 %, 56/100, t95 [−0.755, −0.119] | −0.43698 %, 56/100, t95 [−0.755, −0.119] |
| zero control (s=0 vs 8be) | exact, same bill | exact, same bill (VAL900, FRESH-A 0; job 3604) |
| whole-file Δ bill (real 0.12.3) | +407,143,174 F (+0.1419 %), +735 ops | **+386,451,206 F (+0.1347 %), +612 ops** |
| served-equivalent residual on 8be / on A1 (22,633 ops) | 0.350 s / 0.356 s | **0.348 s / 0.354 s** |

## 7. Re-train on A1's served chain (strategist D1) — **NO-GO, A2 not canaried**
`a1_pipeline.py experiments/2026-10-20_L2/src/cand_r3s736_lam06.py` (A1 = R3 step 736 + λ 6e5, sha 7d381fc5), unattended, 00:36 → 03:20 UTC:
dumps on A1 (TRAIN cupy 3610/3612; VAL/FRESH-A numpy real chain 3613–3620), arm B seeds 0/1 (3628), lean build, zero control + bill (3629),
served reads (3630/3631). Outputs `a1/` (`SUMMARY.md`, `pipeline.json`, `cand_x1b.py`).

| item | value |
|---|---|
| seed selection on TRAIN 850–899 | B0 −0.017 %, B1 −0.002 % → B0 (on 8be: −0.233 / −0.366 %) |
| constants sha | `5c00b624…` |
| zero control (s=0 vs A1, 4 nets) | exact, same bill |
| whole-file Δ bill | +0.135 % F, 23,245 ops ⇒ residual-eq 0.354 s |
| VAL-100 vs A1 (chain 1.51370e-8) | −0.015 % raw, t95 [−0.065, +0.035], 50/100, score-equiv **+0.120 %** |
| FRESH-A100 vs A1 (chain 1.54726e-8) | −0.034 % raw, t95 [−0.094, +0.025], 51/100, score-equiv **+0.101 %** |

Prediction (D1): −0.3 … −0.5 % raw (80 %: −0.1 … −0.6 %). Outcome −0.03 % raw: below the interval. Gates: FRESH CI excludes 0 ✘,
score-equiv ≤ −0.2 % ✘ ⇒ **NO-GO; A1 stands alone.** Reading: R3 refit the in-loop corrector at LR ×10 on the same TRAIN nets, and it
absorbed the locally predictable mean error the post-hoc module was taking on 8be (seed hold-out gain fell 20×). The module's memory gate
is moot and was not run.
