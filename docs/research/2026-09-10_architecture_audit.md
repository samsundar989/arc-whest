# Repository architecture audit — 2026-09-10

Read-only architecture/evidence audit for Samuel's request to reassess the path to winning. This file is research synthesis, not a promotion, experiment contract, lane reopening, or write-up claim. No estimator, instrument, seed registry, truth corpus, or human-owned file was modified; no experiment was run. References below are repository-relative with one-based line numbers at audit time. External leaderboard and literature verification belong to the companion research audit; dated board figures here are repository records, not a fresh board observation.

## Assessment

The repository has a strong, deployable **exact factorized SIMPLE-K3 baseline**, but has not identified or demonstrated a winning architecture. Its newest candidate retains the third-cumulant lineage rather than regenerating it and expresses the arithmetic through float32, shared factor identities, wide contractions, and batched Strassen. That is a substantial improvement over the root `estimator.py`, which still routes width 1024 to Gaussian moment propagation. Keeping the archive stable is intentional; presenting the root file as the current best algorithm would nevertheless mislead a new driver.

The main obstacle is now **accuracy per counted FLOP with deployment safety**, not a missing generic ML framework. Small repairs to the current readout or matrix multiplication spelling do not bridge the documented gap. A winning result needs a new way to preserve/use the information that matters to final means, a materially better self-consistent closure, or both. Existing evidence does not prove such a representation impossible, and public telemetry does not identify the leaders' implementation.

The largest immediate repository defect at audit intake was contradictory knowledge management: failed experiments were generalized into broad impossibility claims, then narrowed by a Skeptic while the older headline survived. The pre-audit snapshot said the score lane was done; the latest LOG review explicitly struck that conclusion. The parent audit has now replaced the snapshot and roadmap, preserving their originals under `docs/archive/`. This section records the original finding, not a defect still present in the new snapshot.

## Actual implementation

| Object | What it actually does | Evidence |
|---|---|---|
| `estimator.py` v2.6.1 | Historical hybrid supporting Sobol sampling at widths up to 256; width 1024 takes `_analytic_only`, which uses Gaussian/K2 mean and covariance propagation | `estimator.py:187`, `:525`, `:530`, `:755`, `:867` |
| `estimator_k3_strassen_L2.py` | Candidate submitted as 330566; SIMPLE third-cumulant propagation with full factorized lineage at live shape, Strassen depth 2 | `estimator_k3_strassen_L2.py:406`, `:510`, `:767`, `:1005`; `experiments/LOG.md:40327` onward |
| `local_sim/p2_k3_fnp.py` | Research reference/port behind regenerating and full modes; its opening docstring remains historically incomplete although later candidate work exists | `local_sim/p2_k3_fnp.py:1`; candidate parity references at `estimator_k3_strassen_L2.py:397` |
| `local_sim/p2_weff.py` | Algebraic factor representation reference for the third-cumulant tower | candidate source `:397`–`:415` |
| `local_sim/p2_augment.py` | Earlier **K2-AUGMENT** reference: covariance plus degree-3 vector and degree-4 scalar; not the same object as later K3-AUGMENT | `local_sim/p2_augment.py:18` |
| `local_sim/p2_t1_augment_lite.py` | Later **K3-AUGMENT** diagram-deletion experiment, including a matrix-valued fourth-cumulant component and extra third-cumulant births | `local_sim/p2_t1_augment_lite.py:106`, `:169`, `:349`, `:582` |

“Exact K3” means exact transport of the selected SIMPLE factorized third-cumulant representation, up to floating-point arithmetic. It does **not** mean exact Gaussian integration, a complete exact third-order probability model, or a complete fourth-cumulant representation. The source carries `(mu, cov, tower, c4)`, with a scalar projected fourth-order component (`estimator_k3_strassen_L2.py:770`, `:806`, `:818`, `:950`). Remaining approximation error is therefore expected even when tower transport is exact.

For each layer, the candidate transports means and covariance through weights, advances the retained factor blocks, extracts the third-cumulant read slices, evaluates the ReLU/Wick polynomial program, creates new factors, and retains the history. The final layer computes only the mean-relevant terms. The youngest factor has a symbolic identity leg; two older factors share the same propagated leg; another factor can be reconstructed by right diagonal scaling. These are real algebraic savings, not empirical pruning (`estimator_k3_strassen_L2.py:406`, `:443`, `:455`, `:604`, `:626`, `:638`, `:787`).

Wide matrices concatenate generation blocks, reducing metered call count without pretending to reduce the underlying arithmetic. Batched Strassen reduces the arithmetic but adds temporary-array and memory traffic. A deeper recursion therefore cannot be chosen from FLOPs alone (`estimator_k3_strassen_L2.py:419`, `:477`, `:513`, `:581`).

The candidate derives a generation cap from shape and budget before spending computation, casts immutable float64 weights through the supported API, has no broad exception swallowing, and clamps nonfinite/negative outputs. These guards mitigate failures; zero replacement is not evidence of accurate predictions (`estimator_k3_strassen_L2.py:723`, `:747`, `:1005`, `:1072`, `:1084`). At live shape the shape model returns 797 matmul equivalents and approximately 0.781 B before Strassen, fitting its 0.95-B guard without generation deletion. I verified this using only scalar arithmetic transcribed from the source; no model or seed was evaluated.

The candidate's long opening comment and cost-model comments still quote **1,022** matmuls / approximately 0.9994 B, describing an older build. The LOG already identifies this exact defect and the function's correct 797 (`experiments/LOG.md:38072`). The comments must not be used to price another port. Editing these frozen/submitted bytes requires the normal candidate process; an external manifest can carry the correction immediately.

## The score and the two different meanings of “floor”

The objective is the arithmetic mean of per-network final-layer MSE times `max(0.1, F / 2**41)`. The **0.1 multiplier floor** is a scoring rule. The **approximately 1.9e-8 raw-error cluster** is a dated empirical observation across different teams, not a proven approximation lower bound.

For a fixed feasible candidate with network-independent FLOPs above the multiplier floor, reducing cost by 20% reduces score by 20%. Below the floor there is no further direct reward for cost savings. Increasing accuracy at a large multiplier can therefore lose to a less accurate cheap candidate; conversely forcing an algorithm onto the floor can discard too much information.

Using only the repository's dated 3.80e-9 winning score as an illustrative target:

| Feasible multiplier | Raw MSE needed to tie 3.80e-9 |
|---:|---:|
| 0.6042 | approximately 6.29e-9 |
| 0.20 | 1.90e-8 |
| 0.10 | 3.80e-8 |

Thus the latest candidate near 3.8e-8 raw at 0.604 B requires roughly a sixfold improvement in the combined accuracy/cost product, before an additional safety margin. Matching the AUGMENT reference's raw error at the candidate's current cost would still be about threefold behind that dated target. None of these arithmetic illustrations is a promotion statistic: a real decision requires paired per-network scores and the preregistered arithmetic-difference interval, including failures and counted cost.

The latest certification artifact records 0.604217 multiplier, 9,321 operations, and candidate hash `390bc226bbf517df0110facfc0dc91c78ef670e8a49368a536cabf630057d1db` (`experiments/2026-09-10_slice5/certify_k3_strassen_L2.json:5`, `:67`). Its local residual figures are not transferable server guarantees. The submission record acknowledges two failed certification clauses and a later clean CONFIRM run; submission approval did not amend those gates (`experiments/LOG.md:40327` onward). Latest server results require fresh retrieval.

## What recent experiments support

### Full lineage matters, within the tested families

The regenerating K3 approximation deliberately discards all-distinct history, leaving roughly 22 times the raw error of full lineage in the stored comparison. Three-generation memory and the tested selector/sketch variants did not recover enough of the lost information at their cost. Crucially, this is a statement about those represented states and mechanisms, not a proof that all compression fails.

WP20 explicitly contains a counterexample to ordering schemes merely by “more information”: its scored sketch initially discarded a diagonal already available to it; a corrected construction improved to 3.330x. Other sketches were non-monotone. The later 3.874x bound and cost rejection concern the specified short-memory family (`experiments/LOG.md:32023`, `:32179`, `:32341`). A proposer must identify exactly which earlier limitation a genuinely different representation escapes.

### Complete K3-AUGMENT improves accuracy; tested deletions fail

The raw artifact contains paired 24-network results using two independent `2**26`-sample labels per network, with recorded per-half truth noise 1.1552e-9:

| Arm | Mean cross-estimated MSE |
|---|---:|
| SIMPLE | 4.004685e-8 |
| AUGMENT | 1.912323e-8 |
| Births deleted / LITE | 4.365096e-6 |
| Raised-budget legs deleted / LEGSONLY | 5.523768e-7 |
| Both deleted / MINIMAL | 2.976232e-6 |

Source: `experiments/2026-09-10_slice5/t1_augment_lite_v2.json:194` onward; truth-noise metadata at `:69` onward. These are cross-estimated errors against independent noisy labels, not leaderboard scores. Full AUGMENT is a reference with an unaffordable presently priced factorization, not a certified deployable candidate.

The two deletion channels were not isolated: the fourth-cumulant update depends on the factor blocks being deleted. The intervention changes both channels. The correct finding is that **these partial term sets perform badly**, often non-monotonically; the experiment cannot attribute the gain to a cleanly isolated cheap or expensive channel (`experiments/LOG.md:40017`). The three partial arms do not establish that every possible subset, closure, or compressed representation fails. “A theorem that cumulant information pays only in an exact tower” is too broad for these observations.

### T2 closes a specific readout attempt

T2's 384-network training corpus and 128-network read corpus yielded gain 1.016345, paired MSE difference 6.2074e-10, CI [4.8472e-10, 7.5675e-10], with 100/128 networks improved. Recorded label noise per replica is 4.4988e-9 (`experiments/2026-09-10_slice5/t2_corrector.json:6`, `:23`, `:28`). This is real small signal and a large miss relative to the 1.15 build gate.

The meaningful positive control is the later gate-strength recovery curve, not the earlier near-perfect planted signal. A smooth weighted-R² target of 0.1304 was recovered at 0.12984; a deliberately oscillatory target at 0.3 was not recovered (`experiments/2026-09-10_slice5/t2_controls.json:19`, `:36`). That limits the null to the tested feature/function family and corpus. A ridge replication strengthens the conclusion but does not prove the features contain no possible information. The in-flight sensitivity test advertised in the module docstring was never implemented or run (`local_sim/p2_t2_corrector.py:20`, `:30`; `experiments/LOG.md:40285`).

P08 was read three times while repairing the instrument. The winning instrument has no pre-result committed state. This does not erase the observed small gain or justify rescuing it; it means this is not a pristine single-use confirmatory read. Fresh verification is needed for any future promotion. The LOG states this explicitly (`experiments/LOG.md:40282`, `:40297`). The original JSON still contains the superseded noise ceiling and the controls JSON still has `passes: false`, so consumers must follow the corrections instead of treating every serialized verdict as current authority.

### Packing rejects this transformation of this factorization

The packing probe prices 797.1778 matmul equivalents, zero aliased calls, 17 calls with a symmetric operand, and a best modeled gain of 1.006689 (`experiments/2026-09-10_slice5/packed_feasibility.json`, final `price` object). It convincingly rejects the proposed symmetry-tag/alias/congruence savings on the existing dominant products. It is **not** a lower-bound proof over all algebraic representations of the same estimator, and does not exclude a different estimator at lower cost.

## Pre-audit contradictions and subsequent consolidation

Line references in this section describe the pre-audit documents. CURRENT_STATE and GAME_PLAN references point to their preserved archives. Other cited historical files received a two-line supersession banner during consolidation, so their current line numbers are two larger than the intake references below.

1. `docs/archive/CURRENT_STATE_before_2026-09-10_audit.md:3` says the whole score lane is done; `experiments/LOG.md:40281` explicitly retracts that conclusion. `docs/STRATEGIST_PACKET_2026-09-11.md:14` repeats the broad closure even while asking whether the unrun in-flight arm matters. Correct status: two named levers failed; present cost transformations exhausted; winning architecture unidentified.
2. `docs/PHASE2_STRATEGY_2026-09-05.md:11`, `:66`, `:70` treats public telemetry as establishing a particular method/floor. Similar cost and error signatures do not uniquely identify algorithms, and comparisons across teams are not within-method budget sweeps. Keep hypotheses separate from observations.
3. `docs/archive/GAME_PLAN_before_2026-09-10_audit.md:3` delegates authority to Sep-5 strategy, while `:115` still calls itself sole active plan and its body directs obsolete carrier work. The Sep-5 strategy's own last amendments still contain the falsified “any deletion returns Gaussian,” old 0.40 label ceiling, and in-flight T2 status. The parent audit replaced this stack with a concise current plan and preserved the historical rationale.
4. The later packet says a “single held-out read” at `docs/STRATEGIST_PACKET_2026-09-11.md:82`, then acknowledges three reads at `:276`. The archived pre-audit snapshot similarly retains “in flight” statements about experiments reported complete above them. Dates and precedence do not reconcile contradictions within one document.
5. `local_sim/p2_adjoint_k3.py` still embodies WP4's confounded intervention. The LOG retracted that kill and returned the adjoint lane to **open, speculative, with a broken instrument** (`experiments/LOG.md:23738`–`:23758`). Do not treat running the existing probe as a valid new test or cite its original headline as a closure.

At intake the snapshot was 4,519 lines and the log 40,401. The log appropriately remains append-only evidence; the snapshot is now a compact replacement-state document with links. Otherwise a new agent could follow a stale instruction while sincerely obeying the startup contract.

## Architecture direction and work boundaries

Retain the working candidate and evaluator. A broad framework rewrite would add risk without improving the estimator. Separate responsibilities conceptually into: (1) immutable reference/candidate manifests, (2) mathematical state and update definitions, (3) counted-operation spelling, (4) shape/budget/failure policy, and (5) an evaluator using exact candidate bytes. The submission remains one legal file; reproducible export and parity checks connect it to the lab.

Three bounded research questions remain defensible to investigate **only after a new evidence-backed contract and Samuel's predictions**, and after resolving any relevant closed-lane status:

| Question | Why it differs from a failed attempt | First discriminating work; no claim of a likely win |
|---|---|---|
| Can final-mean-sensitive response coordinates preserve old lineage at materially lower cost? | A full-history, query-aware representation differs from three-generation carry, covariance-range selection, and generic tensor-norm truncation | Specify the consumed `(3,)` and `(2,1)` contractions and their error propagation; price the complete operator, not just storage. Any adjoint test must preserve the propagator and pass a linear recurrence positive control before interpreting nonlinearity |
| Is a **self-consistent** cheaper fourth-order closure possible? | Deriving coupled updates is different from deleting coupled AUGMENT terms while leaving incompatible updates live | Trace all dependencies algebraically; show valid Gaussian/low-dimensional limits and SIMPLE/AUGMENT controls; establish a feasible counted-FLOP path before requesting fresh label comparison |
| Does in-flight correction on the current exact-K3 state have a usable error tolerance? | T2 evaluated a final readout; the quoted 0.99849 in-flight R² threshold came from a different K2 chain | Complete the previously promised sensitivity specification with coherent and incoherent perturbation controls; estimate an attainable training target and online price before building a learned system |

None of these is established by leaderboard coincidence. None should silently reopen a closed family under a new name. Parent research should attach genuinely relevant external mechanism/evidence, prioritize one question, and carry the exact price, effect threshold, scope, falsifier, truth precision, and single-use data plan into its contract. The strong current baseline should remain the deployment fallback while this happens.

The most valuable immediate restructuring is **epistemic**: one current architecture description, a candidate manifest, and a closed-lane table whose entries include the actual represented object, intervention, budget regime, data tier, and exact scope. That would prevent repeating already-failed variants and prevent a narrow failure from prematurely ending the search for a winning method.

## Late external evidence: 504aldo changes the research priority

During this audit the competition researcher identified a September 9 public code release. I independently read its README and derivation map. The release reports V29 at raw 2.13e-8, multiplier 0.2526, score 5.40e-9, with regenerated fourth-order structure, compressed old sources, and Strassen. These are author-reported results, not a local reproduction. The README also discloses that its scalar tables were fitted on eight public networks. That fitting protocol is incompatible with this repository's future fitting policy; published code is useful architectural evidence but fitted constants should not be adopted as independently validated choices. [504aldo repository](https://github.com/504aldo/whest-p2-cumulant-k3)

The derivation map describes three concrete mechanisms: a source leg written `M = P diag(s) + 3 A diag(e) + Z L^T`; fourth-order regeneration `G = diag(g) + lambda C_off`, with consistent use-side and birth-feed updates; and old source legs represented in a shared transported rank-384 basis, then a nested rank-224 basis. Its rank-16 residual factor and pruned nonlinear term program are approximations. Therefore “7 to 4 exact” applies conditionally to that chosen representation, not automatically to our full SIMPLE chain. [Derivation/code map](https://raw.githubusercontent.com/504aldo/whest-p2-cumulant-k3/main/docs/derivation_code_map.md)

The following scope comparison is my inference from those disclosed constructions and this repository's own instruments:

| External mechanism | Why our recorded negative does not settle it | Required local proof |
|---|---|---|
| Coupled fourth-order regeneration | T1 deleted source blocks while the fourth-order update still depended on them. A jointly derived replacement of both sides is a different intervention | Implement one explicit coupled state definition with toggles separating use-side and feed-side. Reproduce the unmodified control, then compare complete variants on newly minted data. Do not copy the public-fit lambda table |
| Shared basis for all older sources | WP20's ceiling applies to a three-generation horizon. Retaining all generations in common coordinates removes that horizon constraint | Price basis maintenance, joins, reconstruction and consumed contractions. Measure final-score error rather than captured matrix energy alone |
| Gram built from the actual transported legs | A source-leg Gram is not covariance's dominant eigenspace, which the repo found uninformative as a selector | Compare exact and compressed legs while preserving the diagonals actually consumed. Include a matched random-basis control and separate young/old approximation effects |
| Two-leg plus thin residual algebra | Our current `A_D` term is transported densely; a low-rank residual changes the approximated state before its algebra is fused | First prove exact algebra with the residual retained densely; then isolate rank truncation as its own approximation. The external published “exact” label cannot stand in for this proof |
| Buffer reuse and deeper Strassen | Existing measured wall limits describe the current allocation/operation spelling; a different memory schedule can change them | Confirm official supported semantics and local immutable-array requirements before adopting any `out=`/buffer pattern. Re-price and certify exact bytes; never infer a safe depth from the external timing |

This public implementation is the named new external evidence needed to **review reopening these specific mechanisms**. It does not justify reopening generic pruning, recent-only carry, arbitrary high-frequency correctors, or all old sampling work. It also does not prove this architecture will beat the current leader. The appropriate first priority is now a pinned, mechanism-by-mechanism reference comparison with this release, ahead of speculative fresh adjoint work. A coupled closure plus full-history source sharing is a concrete target architecture to test; the current exact SIMPLE candidate remains the control and fallback.

The parent researcher's fresh Chrome inspection separately resolves two stale facts: submission 330566 has now returned a graded result, and rules v14 §7 states the best eligible submission determines the participant's ranking. The earlier packet's unresolved §6/maximin argument should be retired in the consolidated update, with the captured browser evidence cited there. The repository-only 3.80e-9 target used above is historical; current competition arithmetic should use the new board snapshot rather than that illustrative target.
