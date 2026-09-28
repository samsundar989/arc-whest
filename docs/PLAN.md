# PLAN — ARC White-Box Estimation Challenge, Phase 2 (rev 16, 2026-09-28)

**This is the only plan.** It supersedes rev 15 (in git history), incorporates the completed worker results and Samuel's September 28 instruction to review them and seek a substantial score improvement. The September 28 audits and experiments below inform this plan; they do not replace it. Labels: **[M]** measured, **[I]** inference, **[?]** unknown. Phase 2 closes **Oct 17 23:59 UTC**; team freeze Oct 2; write-up Oct 24.

Standing decisions carried forward: one integrating session drives (bounded subagents, no concurrent edits of the same files); designation policy **A = best score (leaf 32), B = its wall-safe leaf-64 twin** (Rules v14 §7 ranks only the better designate); Samuel designates, agents never do; external MC data (keenanpepper corpora) approved; every substantial run has a written hypothesis, expected effect, resource estimate and stop condition *before* it runs. Decision point **Oct 5**; write-up starts **Oct 8** regardless; hardening/designation Oct 12–16. Research probes/training use workerq and conservative declared resources; ordinary candidate evaluation uses the established direct evaluator path.

---

## 1. Where we stand [M]

| role | file | graded | score | raw | F/B | wall median / worst |
|---|---|---|---:|---:|---:|---|
| A | `estimator_k3_v72_gt.py` (`3b6cc397`) | #332766 | **4.620737758949154e-9** | 2.1518525734620654e-8 | 0.21473 | 83.303 / 86.285 s |
| graded wall-safe fallback (older accuracy) | `estimator_k3_v68_ws.py` (`0db9afba`) | #332544 | 5.0664e-9 | 2.1906e-8 | 0.23127 | 51.9 / 55.7 s |

Chrome, September 28: us **#36**, J2W #332700 public score **1.748e-9**, raw **1.616e-8**, mean cost **0.1082 B** (use the result page; the board rounds this to 1.70e-9). On all 50 matched public nets the leader beats v72: mean paired leader/ours ratios are **0.37975 score, 0.75376 raw, 0.50424 FLOPs**. Rounded per-net data and attribution: `docs/research/2026-09-28_public_pairs_332766_vs_332700.csv`. Final-layer reference noise floor approximately **7.7e-11** for all current-shape MSEs in this plan.

Arithmetic, not a forecast: at our raw MSE the 0.1 cost floor gives **2.1519e-9**, still behind. At our current cost matching 1.748e-9 needs raw approximately **8.14e-9**. Removing the entire old tier costs 339.757G rather than 472.199G FLOPs, a measured **28.05%** saving, but its measured prediction becomes unusable. Even a hypothetical lossless free old tier would still require roughly **47% less raw error** to match the leader. A winning cost route must eventually also replace/cheapen recent dense history, or be paired with a large accuracy improvement. The goal for the next structural candidate is **at least 15% lower paired DEV score**, preferably 20–30%; this is a research target, not an added canary gate or a promised result.

**Current exact cost census [M]:** job2309, `tools/cost_site_probe.py`, unchanged v72 and real flopscope0.12.1/f64 read-only grader weights. Exclusive costs sum exactly to **472,202,879,721 FLOPs**, 22,992 ops, zero materializations. Recent-history transport **30.181%** plus D21 readout **31.192%** = **61.373%**; resolved old-history work **27.086%**; covariance transport **7.690%**; closure correction **1.137%**; mean corrector **0.033%**. Mixed work is separately recorded, not arbitrarily assigned. `experiments/2026-09-28_cost_sites/report.md`. This directly identifies recent dense history as the largest cost target. Resolved old-site attribution differs from the28.05% deletion delta because the latter also changes mixed work.

## 2. Closed by measurement — do not rerun unchanged [M]

Rev 14's three steps finished early (`RESULTS.md` § Step 1, § Step 2, § Step 3.1–3.3):

| rev-14 step | result | status |
|---|---|---|
| 1. Harmonic (radial r = 1) old tier | metric-C projection ill-posed (cond up to 1e15); best well-posed radial arm **×5.93 raw** on DEV-100; drop-tier ×17.8 at −28% FLOPs | **closed** — the old tier's value is in the traceless part of κ3 |
| 2. Refit ceilings (exact channel + refit, FIT 80 → FIT 20) | κ3 **53%**, cov 41%, κ4 28%, all-but-μ 91% | **done** — label-intervention results on base-training nets, not attainable gains; not additive. September 28 found a carried-slice compensation issue in substitution semantics; do not call them clean realizable ceilings. |
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
A1. **Completed:** S2t3 step368 → v70 #332737; step736 → v71 #332753; transport gates → v72 #332766. Preserve v72 as the incumbent. Its measured median implies 103.30s noisy worst, below the new-code 105s rule. The older v68_ws is the last graded B; it is not a v72-accuracy twin. Refresh B only after the next useful A candidate is selected.
A2. **Optional small follow-through:** G1bg birth gates finished; best checkpoint **step112**, VAL 2.213009e-8, not final step196. Export/DEV it once if this does not delay structural work; no further gate-network growth or long gate sweep. S2t3's TRAIN-only control used a different cosine horizon and failed before completion, so it does not prove d8b can never help. No new corpus or lengthy continuation is justified by the present marginal gain.
A3. Structural candidates reach submission only through §5. Do not fit on DEV, consume canaries to tune mechanisms, or call an oracle projection a submission improvement.

### Lane B — structural research: all-distinct third-order information
**Measured premise:** the free-running all-distinct discrepancy grows to roughly25% at depth, far above moment-label uncertainty. The worker's rank screen shows a compressible *consumed matrix*, not a demonstrated compact symmetric tensor or online update. A scalar gate has little remaining value. **Unknown:** whether useful missing information is predictable from inexpensive state, and how much of it can replace expensive carried history.

B1. **Repair interpretation, completed first.** `tools/k111_consistent_probe.py` saves actual advanced D3/D21 for compensation while allowing oracle inputs to affect closure/mean. Tiny full-rank CPU identities pass; unmodified baseline is bit-identical. On matched FIT0–1, corrected AD intervention reduces final MSE by median **34.86%**, legacy37.63%; rank32 plus FULL D3 gives22.59% versus23.06%; full input replacement47.75% under either. This is a two-net sanity check on the older v68-equivalent baseline, not a deployable gain or a replacement20-net estimate. `AD_orth` removes only one model-scaling direction, not all possible gate functions.

B2. **Observable basis screen, completed.** `tools/k111_feature_probe.py`, workerq2292/2300, FIT0–7, v71-equivalent state, no state changes. Leave-one-network-out fitting of16 local/global matrix features explains only6–8% of orthogonal error energy at deep layers. Current covariance's top32 **right** directions capture58–68%, but the same32 **left** directions capture only~3%. Ordered quadratic left features improve this, without establishing a symmetric core. **Do not launch a large generic feature MLP from this result.** The next test is the stronger quadratic/symmetric representation in B3. These FIT nets were seen in baseline training; LONO only holds out the new coefficient fit.

B3. **Completed falsification: symmetric residual core.** The test used a label-free covariance basis U with q=16/32 and fitted an *oracle diagnostic* symmetric G for E21=(G contracted with U_i,U_i)U_j, with separate weight/measurement for D3. It included the full quadratic-left/unrestricted-core upper control and verified the matrix-free solver against tiny dense least squares, followed by a slice-consistent downstream intervention. Stop threshold was <10% raw benefit before training. Never infer tensor rank from the prior D21 SVD alone. Initial implementation details and sources: `docs/research/2026-09-28_frontier_options_codex.md` §6–7.

**B3 outcome [M], September 28:** FIT4 downstream test `tools/k111_core_oracle.py`, job2305, failed its stop rule. q32 moment fit makes paired final MSE **4.71% worse** (0/4 wins); weighting D3 by1024 makes it **5.81% worse** (1/4). Unrestricted orthogonal AD correction still helps **23.24%** (4/4). Strict baseline/consumer bit-parity and finite checks pass. The projected core explains moment energy but does not improve predictions. **Close this direct correction design; do not train it from the energy screen.** This is not a test of replacing the actual carried history with a compact, goal-trained state.

B4. **Completed mechanism screen: one omitted covariance triangle.** The selected distinct-index Gaussian term is f2_i f2_j f2_k Cij Cjk Cki. With PSD covariance approximation C≈UUᵀ, carry H_i=f2_i u_i u_iᵀ and read tr(H_i H_j H_k). The packed q=4/8 carrier has10/36 columns. Tiny CPU algebra verifies exact repeated-slice cancellation and reuse of P2 Z for H transport. Label alignment and theoretical coefficient1 were tested separately from an oracle fitted scalar. A later candidate would have needed full cancellation, readout, range-finding and source bookkeeping priced; the first screen did not justify it.

**B4 outcome [M], September 28:** FIT2 q4/8 source2/5/8/11/14 alignment screen, job2304, fails its signal threshold. Deep-layer D21 orthogonal energy capture remains **below0.01%** even with oracle scalar; theoretical D3 correction generally harms moment accuracy. **Close this selected next-layer Gaussian-triangle route.** No final-score gain measured; this narrow result does not exclude other connected diagrams.

B5. **Primary new hypothesis: compact history trained for the final output.** Goal-oriented compression/distillation must target recent dense history as well as the old tier: exact old-tier removal alone saves only28.05%. Measure which complete source removals dominate *paired score* and suffix sensitivity, then train a compact replacement state to reproduce the useful D3/D21 and final means. Keep zero-removal, unweighted-fit and equal-step controls. This is a new trained representation, not an unchanged rank/age sweep. Milestone targets (not claims): raw≤1.5e-8 at cost≤0.16B for a substantial intermediate gain; raw≤1.5e-8 at cost≤0.11B would beat the current observed leader. Meter the complete candidate before committing long training; a held-out coefficient fit or oracle compression is not sufficient evidence.

**B5 is now the first implementation lane.** Ordered experiment:

1. **Completed: exact real-meter cost census**, job2309, reconciled to472,202,879,721 FLOPs. Recent dense transport/readout accounts for61.373%, resolved old work27.086%. The replacement targets these operations. Charge its basis construction/folding/readout too; never use the old shape-counting shim to choose a cost target.
2. Build one **replacement student**: keep two or three recent exact source lineages and summarize the rest by one symmetric core G and thin basis U. Start with q32; price q64 only as the second predeclared capacity point. Derive new cores analytically from projected incumbent source factors, transport with the actual weights/Wick map, and transform cores consistently when U is re-orthogonalized. Preserve the existing covariance path and mean corrector initially to isolate this change. A compact core which is simply added to the current tower does not meet this cost goal.
3. Validate tiny dense tensor projection/transport, diagonal and (2,1) bookkeeping, zero-change controls, deterministic seeds and immutable-array portability. Build the **shipping computation** before a large fit: no per-network fitted parameters, labels, dense teacher, future teacher suffix, raw uninstrumented math, or dense eigendecomposition assumed free. A fixed-step thin range finder may replace diagnostic full eigenvectors only after its approximation/cost is measured. Screen complete cost near **0.12B or lower** and op/wall/memory bounds. If it cannot offer substantial cost reduction, stop before training.
4. Short controlled fit: same TRAIN networks/seeds/steps for analytic projection control, student with small shared fold/birth gates, and continued incumbent control. Train to the final output first, using teacher means/readouts as stabilization and organizer TRAIN means as the target; do not optimize unweighted tensor energy alone. Existing source-column features (norms and inner products of factors, age, birth weights) may condition shared gates; the per-net core itself must remain a computed state. An offline adjoint/per-net oracle is only a diagnostic: thousands of free core coefficients can overfit a single teacher output, so that success is not evidence of online learnability.
5. Select on VAL by **MSE × counted cost**, including controlled raw-error increases if cost pays for them. Expand to TRAIN900 only when a short run supports at least15% score improvement over v72 and beats the matched controls. Then DEV100, all §5 gates, and one canary. Intermediate target20–30% score improvement is useful even if it does not immediately win. A convincing leader-level path needs both raw accuracy and cost near the targets above.

B6. **One selected candidate at a time.** B3/B4 have failed their current tests; proceed to B5, not a larger version of the same correction. If the replacement student also fails its price or cross-network score gate, use the exact census to choose the next operator to compress and record the failure before changing mechanisms. Layout changes may create wall/memory headroom but only lower counted FLOPs or enabled accuracy gains improve score. No experiment above establishes a winning architecture yet.

### Lane C — write-up (from Oct 8; material accrues now)
Mechanism results already write-up grade: channel-drift anatomy, refit ceilings, closure bench/regression and its in-loop negative, the harmonic negative, the elasticity map; B1's measurement whichever way it lands.

## 5. Submission acceptance — every item, for the exact final source hash

Exit code 0 or an "all gates pass" line is **not** acceptance; `tools/ev.py` now exits non-zero unless every hard gate below that it can check has run and passed. Evidence recorded in `RESULTS.md` against the source sha8:

1. `whest validate` passes (official contract).
2. Numerical parity: the exported file reproduces the fitted twin computation — DEV panel mean within **1e-3 relative** (the twin's acceptance bar, RESULTS § Step 2 twin; two float32 backends differ ~1e-3 per net), and the build pipeline reproduces a known export byte-exact; data-only diffs checked by diff. *(Rev 15 first said 1e-4; corrected Sep 28 after v71 read 1.9e-4 — the 1e-4 figure had no basis in the backend noise; prior exports read 1.4e-5–6.3e-5.)*
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
