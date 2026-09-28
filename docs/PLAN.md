# PLAN — ARC White-Box Estimation Challenge, Phase 2 (rev 17, 2026-09-28)

**This is the only plan.** Rev 17 turns rev 16's reviewed research direction into the implementation, training and submission queue requested by Samuel on September 28. The compact student is still **unbuilt, unpriced and untrained**. `docs/HANDOFF_2026-09-28_COMPACT_HISTORY.md` delegates execution of this plan; it does not add a second strategy. Older handoffs are historical. Labels: **[M]** measured, **[I]** inference, **[?]** unknown. Phase 2 closes **Oct 17 23:59 UTC**; team freeze Oct 2; write-up Oct 24.

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

B5. **Primary new hypothesis: compact history trained for the final output.** Goal-oriented compression/distillation must target recent dense history as well as the old tier: exact old-tier removal alone saves only28.05%. Train a compact replacement state to preserve useful D3/D21 and final means, following the controls and ordered tests in E0–E6. This is a new trained representation, not an unchanged rank/age sweep. Longer-range targets (not claims): raw≤1.5e-8 at cost≤0.16B for a substantial gain; raw≤1.5e-8 at cost≤0.11B would beat the current observed leader. These are not pilot admission requirements. Meter the complete candidate before committing long training; a held-out coefficient fit or oracle compression is not sufficient evidence.

**B5 is now the first implementation lane.** Execute stages E0–E6 below in order. This replaces the looser rev 16 queue. No new experiment or result is claimed by this revision.

**B5 outcome [M], September 28 (after rev17):** closed at the E2 retention screen, before the E1 build. v72 with age3 plus shared rank q upper-bounds the analytic age3/q core's retention (same all-mode projection, P2/hub kept exact). On TRAIN0–7: r64 ×15.7 raw, r32 ×26.6, r128 ×6.8, r256 ×1.86 (metered 0.18725B, −62.5% paired score); 3 power passes do not help. The E5 fallbacks also fail at the price screen: age3/r256 as above, and a new youngest-m D21 read (m=4, 0.18573B, ×15.6 raw). No pilot, fit or candidate. `experiments/2026-09-28_compact_history/report.md`, RESULTS § "PLAN rev17 B5". Missing evidence for a ≥15% route: a κ3 representation whose dense (2,1) readout is materially cheaper than one n×(lineage columns)×n contraction per lineage-layer while retaining ≳256-dim content.

#### E0 — pin the comparison and the actual teacher

- Re-read current RESULTS and the submission ledger; preserve all existing dirty logs, checkpoints and archives. If a genuinely better graded incumbent has appeared, refresh the baseline explicitly before starting a comparison. Otherwise freeze v72 SHA256 `3b6cc39790295d3be60a88f9dd9053915f695bdf0a0da2cf2e99587cce692bac` as teacher and control.
- Reuse valid, hash-matched v72 predictions and cost evidence. Its actual parameters include S2t3 step736 and G1tg step84. The older S3/v68 or v71 diagnostic initializations are not this teacher. Prove that the training/export path reproduces v72 before attributing any gain to compression. `tools/twin/twin.py` currently trains through an unbaked source; simply passing the baked v72 file to its existing `train` command is not a supported replacement for this check.
- Use `experiments/2026-09-28_compact_history/` for the new evidence, with distinct subdirectories for every arm. Record source/parameter hashes, panel membership, exact commands, loss, optimizer/schedule, resource review, measurements and decision in a concise run note; record candidate rows in RESULTS. This is normal reproducibility, not a revival of retired pre-registration machinery.
- Use bounded agents for independent algebra, implementation, and measurement review. One integrator owns the carrier/layer integration and canonical documents; assign disjoint files. No per-experiment Skeptic process.

#### E1 — implement one complete replacement, with two capacities

**Exactly age3/q32 and age3/q64 initially.** Here age3 means that, after each nonfinal layer's birth, exactly `min(3, births_so_far)` young source lineages remain. Each expired source is folded once into one symmetric core `G(q,q,q)` and basis `U(width,q)`, then its obsolete dense A/M/Z/L/Y and associated metadata are removed. Audit every consumer before dropping a field: any information still needed by another channel must have an explicit replacement or be retained and priced. Retaining the old tower behind a new core is not this experiment. Clamp rank for narrow widths; keep the code shape-generic.

The core represents `T[i,j,k] = sum_abc G[a,b,c] U[i,a] U[j,b] U[k,c]`. Implement:

1. Actual linear transport (`W.T @ U` in the current source convention) and Wick row scaling of U. Preserve the v72 transport gate consistently in both transport and compensation.
2. QR gauge change: if `U = Q R`, apply R to all three modes of G. Rebasing into a new orthonormal Q uses `Q.T @ U_old` on all three modes. No full width-cubed tensors in production.
3. Analytic joining from the **entire** projected source: D-block, separable/two-leg terms, factored P2 residual, and active hubs. Derive each symmetrization constant from our own reference algebra. Projecting only A_L/M omits carried information.
4. Readout `B[i,c] = sum_ab G[a,b,c] U[i,a] U[i,b]`, then unmasked `D21 = B @ U.T` and `D3 = diag(D21)`. Respect the existing off-diagonal convention at consumers; test the identity before masking.
5. One frozen, label-free basis algorithm for both capacities. Start from a matrix-free **weighted history-factor Gram** range finder, using the existing join's pattern and including the old core's mode Gram plus all joining term families. Form the old core's mode Gram after orthonormalizing U and rotating G, or include the appropriate `U.T @ U` contractions: after weight/Wick transport, `G_(1) @ G_(1).T` alone is not the physical tensor's mode Gram. Write down the term weights and how they transform with factor scaling. Use norm-selected deterministic initial columns and **one Gram power step**, followed by thin QR; deterministic rank-deficient handling is required. Derive/record the exact proxy before the first score read. Charge every construction and contraction. Do not substitute full covariance eigenvectors or silently search basis choices. A different basis after failure is a new recorded hypothesis.

**Layer order is a correctness gate.** Read closure inputs from the actual incoming carried tensor; compute the intended closure outputs; Wick-transport the inherited state; fold/rebase expired history; obtain the slices of the tensor **actually retained**; form the birth residual against those slices; append the birth. A lossy fold changes D3/D21, so subtraction of the pre-fold slices is wrong. Recompute the changed contribution only where algebra permits reuse, rather than doubling all young-source readout work. Verify exact closure-slice reconstruction with full-rank P2 on tiny examples; separately measure the existing production P2 rank-4 approximation error. Do not claim exact cancellation for a rank-4 approximation.

Initially keep the covariance path, closure tables and existing mean corrector fixed at v72 values. New shared gates may act on joining contributions using cheap current-state features (factor norms/inner products, birth coefficients, age and compressed summaries), with identity initialization and bounded outputs. G is a computed per-network state, never a table of fitted per-network coefficients. No teacher, labels, dense future suffix or uninstrumented math may appear in the shipped predictor.

**E1 exit evidence:** disabled-path v72 parity; enabled no-fold parity on short tiny chains; dense tiny tensor identities for projection/transport/Wick/join/rebase; symmetry; D3/diagonal identity; gauge/sign and neuron-permutation checks; zero/rank-deficient factors; widths including 2 and odd widths; deterministic repeats; finite checks before inherited sanitization. Distinguish algebraic tolerance from floating-point reassociation. Export through allowed flopscope/stdlib imports and immutable arrays. These tests protect novel tensor bookkeeping, not incidental implementation details.

#### E2 — price the shipping path, then screen analytic retention

- Price both complete implementations with the real flopscope 0.12.1 f64 read-only grader twin, including basis selection, rotations, birth compensation, auxiliary state and gate overhead. Reconcile a site census to total FLOPs; record ops, materializations, peak arrays and projected wall. Torch/shim counts and asymptotic formulas cannot admit a fit. Target **F/B ≤ 0.12**; the prior proposal's student cost is still unknown.
- An arm above 0.12 stops before training by default. A bounded exception is allowed only if its **untrained analytic** paired score already improves by at least 15% on the small screen and a full VAL check of that **same untrained analytic student confirms the gain before any training**; record why the cost target was relaxed. This research price screen does not change §5 canary rules. No expensive fit to rescue an unpriced architecture.
- On the same TRAIN rows full0–7, compare frozen v72, age3 with zero compressed history (a diagnostic deletion control), and each affordable analytic student. Record final teacher-mean distortion, labeled final MSE (floor ~7.7e-11), and D3/D21 errors by depth; accuracy-only GPU probes are not counted scores. Use the measured complete cost for each arm's score. Do not rerun the already closed old-tier deletion experiment unchanged.
- Admit a bounded pilot if the analytic student is finite, passes algebra, and retains useful teacher signal relative to the matching age3 zero-core control: lower mean final teacher distortion and wins on at least 6/8 rows. This small TRAIN screen is a triage rule, not generalization evidence. If it fails, first inspect projection/birth residual error; allow one bounded failure diagnostic under E5, not an automatic long fit.

**Cost-aware targets.** For each net compute `s_i = MSE_i * max(0.1, F_i/2**41)`. On a common panel report the paired differences `s_v72,i - s_student,i`, their mean/t95 interval, wins/N, median and worst decile; report percentage gain as the paired mean difference divided by mean baseline score. Never infer a gain by dividing unpaired panel means. Raw MSE does not have to improve when the cost saving pays for it.

Arithmetic against the current public anchor, assuming constant cost 0.12B:

| target relative to v72 | public score guide | raw MSE guide at 0.12B |
|---|---:|---:|
| break even | 4.62074e-9 | 3.85061e-8 |
| 15% better | 3.92763e-9 | 3.27302e-8 |
| 20% better | 3.69659e-9 | 3.08049e-8 |
| 30% better | 3.23452e-9 | 2.69543e-8 |
| match Sep28 observed leader | 1.748e-9 | 1.45667e-8 |

These are planning arithmetic, not predictions or promotion evidence; all quoted final MSEs have reference noise floor ~7.7e-11. Actual decisions use same-panel v72 comparisons and actual cost. In particular, raw≤1.5e-8 is a leader-level ambition at this cost, **not** the pilot admission bar.

#### E3 — short, controlled cross-network learning test

Use full TRAIN rows0–79 (`fit80`) for the pilot. Read VAL rows900–919 at fixed checkpoints, then the whole VAL900–999 for the selection decision. Neither subset is an untouched test. Do not fit any normalization, basis objective, gate or teacher target on DEV/VAL labels. No additional corpus in this first isolation experiment.

Initial comparison: frozen analytic q32/q64 controls; one shared-update learner for each E2 survivor; and one continued-v72 control. Freeze covariance/closure/corrector parameters in the student pilot; train only its new small fold/birth update. The continued-v72 control trains its existing transport-gate parameters from their actual v72 values. List the trainable tensors for each arm; these are different parameterizations, so report that limitation. Baseline is always the better of frozen v72 and its continued control on the comparison panel.

**Bounded pilot default:** batch one network, accumulate eight; six passes over the same80 rows = **60 optimizer steps**. Use the same fixed network orders, effective batch, optimizer family, 60-step cosine horizon and validation cadence (every10 steps) for trained arms. Start from the established relative-RMS Adam parameter scaling at lr3e-4, identity new gates, and a documented zero-tensor scale. Verify a one-step gradient/finite smoke before launching the pilot. Loss is final-layer organizer MSE plus `beta * MSE(student_mean, frozen_v72_mean)`, with beta=1 for steps1–20 and beta=0 for21–60. This makes teacher stabilization explicit while retaining the scored output as the objective; use the same loss convention for the continued control. Do not add unweighted tensor-energy loss. Intermediate/readout loss is a separately documented single corrective experiment only if instability, rather than representation loss, is measured.

Implement this schedule explicitly; the present twin CLI does **not** already implement compact-core training or this teacher loss. Preserve other stages. Do not silently rely on `--max-steps`: the present early smoke exit can precede checkpoint saving, and changing `--epochs` changes the cosine horizon. Checkpoint model, optimizer, schedule, data/RNG position, stage and source/panel hashes atomically; verify resumed behavior in the small smoke. Explain whether gradients traverse basis selection/QR; a detached basis is a training approximation, not the exact gradient of the full recurrence. Handle rank deficiency without masking nonfinite gradients.

Select the best scheduled checkpoint by VAL score, then evaluate its full VAL100 paired against frozen and continued v72. **Long-run admission:** at least15% paired mean score improvement versus frozen v72 on full VAL, positive paired mean improvement versus the continued control, no nonfinite/failing networks, and no unexplained concentrated tail regression. Report t95 and checkpoint-selection bias; the interval is descriptive, not an independent test or an extra significance gate. An analytic student that already qualifies may skip fitting and advance to E6. One small smoke-driven learning-rate correction is allowed if necessary; no learning-rate/basis/rank sweep.

#### E4 — expand only the winner, then finalize

Expand the single best affordable student to TRAIN900. Start with **two epochs**, accumulation8 (224 optimizer steps in the present floor-dividing loader), full VAL every28 steps and early stopping after three non-improving reads. Record the treatment of the four remainder rows and rotate order by epoch. Fix the horizon before the run; do not compare controls with different horizons. Continue an incumbent control on the same data/order and schedule if doing any new joint tuning.

Start with only the proven shared update trainable. If useful gain plateaus with stable state, allow one matched joint stage unfreezing the existing small mean corrector/transport parameters in both student and continued incumbent; keep covariance and closure machinery fixed. Reprice/export any change in inference arithmetic. Permit one additional bounded continuation of at most two epochs only when late full-VAL results show continued improvement; give it an explicit new schedule and matched control, not an endless extension. Choose by full-VAL **score**, not training loss or moment-energy fit.

Do not delay a student that already passes the target and all §5 gates for speculative training. Run full DEV100 only for selected exports. Target20–30% lower score for the next substantial result; at least15% is the structural research milestone. A smaller genuine improvement can still qualify as a canary under unchanged §5, but label it incremental and keep the structural objective open. No research threshold authorizes a submission without §5.

#### E5 — diagnose a failure and change one mechanism

Classify the failure before spending more compute:

| failure | next bounded action | prohibited inference |
|---|---|---|
| total cost/ops too high | Use the new exclusive census to fuse/reuse algebraically identical contractions, remove obsolete source state, or reassociate the same readout. Reprice once. If insufficient, stop this carrier. | Torch speed or smaller allocated arrays imply a better counted score. |
| analytic projection loses signal | On the same8 TRAIN nets locate the first harmful fold with depthwise slices and final-output ablations. Compare full-rank tiny identity and q32/q64 retention. Fix an algebra defect before judging capacity. | More tensor energy retained must improve score. |
| gates cannot recover despite finite gradients | One diagnostic at a fixed late boundary: bounded core optimization through the real suffix on at most4 TRAIN nets, fixed known basis, to separate representation limitation from update learning. This is an oracle diagnostic only. | Per-network core fitting proves learnability; q32 already has5,984 symmetric coefficients versus1,024 final outputs. |
| training gain disappears on full VAL/DEV | Check teacher/init/export/panel parity and the paired tail. Preserve the negative. Stop this trained configuration after one diagnosed correction. | More checkpoint selection or another canary substitutes for generalization. |

If both capacities fail, close **this** carrier and record a short decision in PLAN/RESULTS. The next concrete fallback to investigate is factor compression of the **recent source A/M transport and D21 contractions** while retaining the existing older-history representation. It must project full source terms with consistent compensation and optimize final means; merely lowering `_SHARED_AGE`/`_SHARED_RANK` is a closed knob sweep. Before a new fit, derive and meter one complete operator replacement and compare its cost-aware analytic retention on the same small panel. If the remaining cost cannot plausibly support15% score gain, stop it at the price screen and state the next missing evidence. Do not promise or manufacture a winning submission; surface the precise blocker and best valid candidate. Keep the Oct5 strategy decision and Oct8 write-up start.

#### E6 — complete the graded loop

Use the existing evaluator on the selected final export, e.g. in PowerShell:

```powershell
uv run python tools/ev.py <candidate.py> --panel dev --n 100 --baseline estimator_k3_v72_gt.py --jobs 1 --mem --json <run-dir>/dev100.json --append
```

Replace placeholders and use the current incumbent if E0 changed it. Begin with one evaluator process; increase only if measured memory/headroom supports it. Check every §5 item including manual new-code noisy-wall projection; `ev.py`'s accepted flag alone is insufficient. Package the exact tested hash, validate the package, inspect the UTC-day ledger, and submit one eligible non-designated canary under standing authorization. Watch it to a terminal result. If submission is blocked by an actual hook/approval request, report the precise action and reason; do not bypass guards or designate an entry. If the quota is exhausted, preserve a ready, hash-verified package and record the earliest allowed UTC date.

After grading, collect per-network results/failures/cost/wall, compare with the prediction, and update RESULTS, CURRENT_STATE, calibration and DESIGNATION **evidence only**. A better graded row becomes the baseline. Refresh its wall-safe B only after a useful A exists and under applicable gates; retain v68_ws meanwhile. Report achieved improvement separately from targets and remaining distance to the observed leader.

#### Resource construction for E1–E5

Before every heavy probe/training job, record the optimization review required by AGENTS: dataset/loader duplication, workers, caching/prefetch, batches, device placement, avoidable passes, recovery and reusable artifacts. Research/training goes through `workerq`, with conservative declared peaks, no priority changes. Ordinary `tools/ev.py` candidate evaluation stays on the direct path. Tiny CPU algebra checks may run directly.

For the first GPU forward/backward smoke, use a defensible24GiB host/16GiB VRAM/2CPU estimate based on the existing checkpointed one-net twin and new small core; revise upward before submission if the live-tensor analysis warrants it. Measure peaks and time, then request measured conservative resources (prefer16–24GiB host when valid) for later jobs. Batch one; no simultaneous teacher autograd graph. Cache only frozen teacher means initially, keyed by teacher/network hashes; all1000 networks'16×1024 float32 means are about62.5MiB. Cache only the needed pilot set first. Stream any optional D3/D21 supervision per network/layer; never build a TRAIN900 dense D21 cache. Reuse public-data storage instead of duplicate loaders. Preserve dtype and checkpoint semantics; do not trade evidence or recovery for queue admission. Record local throughput as local, never server wall.

B6. **One selected candidate at a time.** No experiment above establishes a winning architecture yet. Continue independent coding/research while jobs queue; report stage completion, graded submissions, twice-failed gates, or a real decision need. A failed hypothesis narrows the next experiment; it does not justify an unchanged sweep.

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
