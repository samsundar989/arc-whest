# PLAN — ARC White-Box Estimation Challenge, Phase 2 (rev 15, 2026-09-27)

**This is the only plan.** It supersedes rev 14 (Sep 27 morning; in git history) and incorporates Samuel's Sep 27 direction (turn the Sep 27 review — `docs/research/2026-09-27_{competition_review,architecture_audit,evidence_audit,literature_audit}_codex.md` — into measured progress and the next graded submission) plus `docs/STRATEGIST_PACKET_2026-09-27.md`. Labels: **[M]** measured, **[I]** inference, **[?]** unknown. Phase 2 closes **Oct 17 23:59 UTC**; team freeze Oct 2; write-up Oct 24.

Standing decisions carried forward: one Claude Code session drives (subagents for bounded tasks, one integrator); designation policy **A = best score (leaf 32), B = its wall-safe leaf-64 twin** (Rules v14 §7 ranks only the better designate); Samuel designates, agents never do; external MC data (keenanpepper corpora) approved; every substantial run has a written hypothesis, expected effect, resource estimate and stop condition *before* it runs. Decision point **Oct 5**; write-up starts **Oct 8** regardless; hardening/designation Oct 12–16.

---

## 1. Where we stand [M]

| role | file | graded | score | raw | F/B | wall median / worst |
|---|---|---|---:|---:|---:|---|
| A | `estimator_k3_v68_wino.py` (`98e40c83`) | #332478 | **4.7039e-9** | 2.1906e-8 | 0.21473 | 84.0 / 93.9 s |
| B | `estimator_k3_v68_ws.py` (`0db9afba`) | #332544 | 5.0664e-9 | 2.1906e-8 | 0.23127 | 51.9 / 55.7 s |

Board Sep 27 (Insights / result pages): J2W #332617 **1.761e-9** (raw 1.615e-8, 0.109 B), Luna #332577 1.907e-9 (raw 1.422e-8, 0.134 B); us #33. On the 50 common public nets J2W beats v68 on every one (≈ −26% raw, −49% FLOPs, paired, rounded values). Arithmetic: at our raw MSE the 0.1 floor gives 2.19e-9 (still behind J2W); at our cost we would need raw ≈ 8.2e-9 (−63%); a free, lossless old tier alone gives ≈ 3.38e-9. **Both axes must move; no knob moves them.**

## 2. Closed by measurement — do not rerun unchanged [M]

Rev 14's three steps finished early (`RESULTS.md` § Step 1, § Step 2, § Step 3.1–3.3):

| rev-14 step | result | status |
|---|---|---|
| 1. Harmonic (radial r = 1) old tier | metric-C projection ill-posed (cond up to 1e15); best well-posed radial arm **×5.93 raw** on DEV-100; drop-tier ×17.8 at −28% FLOPs | **closed** — the old tier's value is in the traceless part of κ3 |
| 2. Refit ceilings (exact channel + refit, FIT 80 → FIT 20) | κ3 **53%**, cov 41%, κ4 28%, all-but-μ 91% | **done** — diagnostic ceilings on nets used in base training, not attainable gains; not additive |
| 3.1–3.2 Closure bench + regression | transported var residual = observed drift; residual is a smooth nonlinear state function (MLP R² 0.63–0.87); κ3 drift **not** visible in marginal/(2,1) outputs | **done** |
| 3.3 Learned marginal/pair closure (S5) in the loop | **−0.75% raw at +1.68% FLOPs** → score-negative; per-neuron-only S5u = the no-net control (−0.17%) | **closed** |

Also closed (rev 13/14 §4, unchanged): shared rank ≠ 256 (both directions score-negative), birth-layer fold age, adaptive age/λ, tiered ranks, σ corrector, cov_out from corrected pK1, recursive corrector feature, hidden 64, memoryless old-tier regeneration, per-source dense κ4 legs, sampling of any kind, κ4 matrix core with memory (ceiling 28% < 30% bar). **Leaf 16 is not assumed to help**: it measured *more* billed FLOPs (0.22014 vs 0.21618 B, 26,773 vs 23,645 ops).

Forecasts contradicted by these results (radial fold within +10%, closure calibration ≥ 5–20%, κ4 ceiling ≥ 30%) are withdrawn and not repeated.

## 3. Evaluation semantics (repaired Sep 27) [M]

| name | membership | role | may be fitted on? |
|---|---|---|---|
| **DEV** | mini rows 0–99 | candidate scoring, paired vs incumbent; the canary gate | **never** (it is also repeatedly used for selection — treat as a design panel, not an untouched test) |
| **TRAIN** | full rows 0–899 (= FIT 0–99 + former holdout 100–899, authorised HANDOFF 09-25) | twin fitting | yes |
| **VAL** | full rows 900–999 (`twin_train_split.json` `holdout_100`; the twin logs call it "HOLDOUT-100") | checkpoint selection / early stopping | no — but repeatedly read, so it is a **validation** set, not a test |
| **d8b** | keenanpepper d8b corpus rows 0–2047 (MC final means) | extra TRAIN data (S2t3) | yes |
| **HOLDOUT-LEGACY** | full rows 100–999 (`holdout.json`, Sep 21 definition) | historical rows only | **contains 800 TRAIN nets** |

There is no untouched local test any more; **the organizer grading run is the only unseen evaluation**, and the private re-run the only one that counts. `tools/panels.py` pins these, checks disjointness (`--check`), and refuses `--panel holdout` (use `val`, or `holdout_legacy` explicitly to reproduce a historical row, which is labelled as such). FIT-20 (full 80–99) ceiling reads are withheld from the 80-net refit only; they were in base training.

## 4. The plan

### Lane A — incremental, near-term (submission track)
A1. **S2t3** (S3 parameter set continued on TRAIN 900 + d8b 2,048, lr 1.5e-4, 736-step cosine; worker-q 2229, preemptible, low priority). Its step-368 checkpoint (VAL −0.79% vs S3) is exported as `estimator_k3_v70_s2t3.py` (data-only diff from v68_wino). Take it through §5; canary if it passes. Attribution control: worker-q 2236 (S3 init, TRAIN 900 only, same flags, read at step 368) — attribution only, it does not gate the canary. If S2t3's later checkpoints beat step 368 on VAL, export again and re-run §5; a new canary needs a DEV-paired gain over the then-best graded row.
A2. Gate/closure candidates from Lane B reach this lane only through §5.

### Lane B — structural research: all-distinct third-order information
Hypothesis [I]: the κ3 error that the refit ceiling prices at 53% sits in the all-distinct (three different neurons) part of the carried third cumulant — how ReLU births it, how it is transported, and how it is compressed — which the marginal/pair closure bench cannot see. Not established; this lane tests it.

B1. **Projected all-distinct diagnostic** (`tools/k111_diag.py`, `experiments/2026-09-27_k111/`). With t = post K3, S = post K21 (diagonal zero), W the next weights: repeated D3 = Σ_a W³ t + 3 Σ_a W² (S W); AD = D3_next − repeated (and the D21 analogue). Compare model vs label AD on FIT nets, in free-running and exact-input modes; noise from the rep1 re-bake (K21 is N = 1e8); downstream effect by substituting only the AD part vs only the repeated part. Verified first on a tiny dense symmetric tensor. Stop: AD discrepancy below its noise floor, or AD-only substitution moves final MSE < 5% → the lane's premise fails; record and go to B3.
B2. **State-dependent factor gates** (only if B1 supports it): bounded, identity-initialised per-neuron gates g = 1 + s·tanh(f(α, γ3, γ4)) on (i) the new all-distinct birth factors (`fac1`, `w2b`), (ii) the inherited Wick transport (`_wick_tower`), tested separately; (iii) feed hubs only if (i)/(ii) show signal. Factor count unchanged; `p111_3`/`p111_21` recomputed from the gated factors (checked algebraically at full P2 rank, then the secondary error at shipped rank 4 measured). Identity parity before any fit; train through the twin from the newest parameter set; **equal-step no-gate control** from the same init. Price FLOPs/ops before training. Kill an arm when its VAL gain vs the control does not pay for its FLOPs, or disappears against the control. Do not grow the gate network to rescue an arm.
B3. **Conditional alternatives**, chosen by B1's measured discrepancy: (a) compact nonradial old-tier remainder — first the D21 singular-spectrum screen *with the diagonal restored*, r ∈ {8, 16, 32, 64}, across ages and transports; reject implausibly small ranks before any online compressor; (b) selected omitted connected terms (ARC TODO: factorized AUGMENT drops non-hypertree diagrams) — oracle-contracted effect and full carried cost first; never restore wholesale.
B4. **Layout** (no-concatenate blocks, earlier temp release, merged thin calls): runtime/memory headroom only; a score gain only if it lowers counted FLOPs or enables a better estimator. Leaf-16 A work must meet the 105 s noisy bound (§5).

### Lane C — write-up (from Oct 8; material accrues now)
Mechanism results already write-up grade: channel-drift anatomy, refit ceilings, closure bench/regression and its in-loop negative, the harmonic negative, the elasticity map; B1's measurement whichever way it lands.

## 5. Submission acceptance — every item, for the exact final source hash

Exit code 0 or an "all gates pass" line is **not** acceptance; `tools/ev.py` now exits non-zero unless every hard gate below that it can check has run and passed. Evidence recorded in `RESULTS.md` against the source sha8:

1. `whest validate` passes (official contract).
2. Numerical parity: the exported file reproduces the fitted twin computation (VAL or DEV panel mean within 1e-4 relative; data-only diffs checked by diff).
3. **Full DEV-100** evaluation with `tools/ev.py --baseline <incumbent>`: 100/100, **zero failed nets**, paired statistics reported.
4. **Projected public score** (DEV ÷ current `PUBLIC_FACTOR`, 1.038) **better than the best graded row**.
5. FLOPs from the real f64 read-only flopscope meter (`tools/_cost_worker.py`), never the counting shim.
6. Op calls **≤ 26,000**; projected server residual (ops × 11.09 µs) **≤ 0.33 s**; setup **< 2 s**; **0** scalar materializations.
7. **WSL 8 GiB RLIMIT_AS gate PASS** (`--mem`) on the final hash — mandatory, not optional.
8. Wall: a **data-only** candidate inherits its code path's measured wall (A policy: implied noisy worst = graded median × 1.24 ≤ ~115 s, rev 13). A **new code path** for A (layout, leaf 16, new ops) must project ≤ **105 s** noisy (measured parent median + wall-law delta, × 1.24). **B** keeps **≤ 95 s** noisy. Measured and projected walls are labelled as such.
9. `whest package` → `whest validate-package`; the packaged `estimator.py` sha256 equals the tested source's.
10. Ledger: `experiments/submissions.jsonl` checked first — **≤ 5 non-designated canaries per UTC day, total** (no separate backup slot assumed).

Then `whest submit --watch` from PowerShell. After grading: RESULTS row, per-net telemetry, calibration check (recalibrate if graded deviates from projected by > 2%), DESIGNATION evidence update. Never designate, contact organizers, or post.

## 6. Not doing
Everything in §2; re-running closed experiments unchanged; another generic closure MLP; nearby rank/age sweeps; sampling; fitting on DEV or expanding TRAIN into VAL; claiming VAL checkpoint selection or training loss as a graded gain.

## 7. Evidence
`RESULTS.md` (all rows and sections), `docs/research/2026-09-27_*_codex.md` (review, audits, public pairs CSV), `docs/research/2026-09-27_G_closure_learning_lit.md`, `docs/STRATEGIST_PACKET_2026-09-27.md`, `docs/DESIGNATION.md`.
