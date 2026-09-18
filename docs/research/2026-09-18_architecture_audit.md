# Architecture audit — 18 September 2026

Read-only source and existing-artifact audit. No candidate generation, measurement, label access, seed claim, submission or designation. Queue snapshot during review: stage-1 job 1663 succeeded; batch-B job 1668 running. The latter's partial predictions/rows were not analysed. This note proposes hypotheses, not a new execution contract, and does not supersede the running slice.

## Decision

Keep the full mean/covariance, factored third-cumulant history, regenerated fourth-order core family. It has both strong mechanistic grounding and measured improvements. It is not yet a validated winning architecture. The present code is an approximate finite closure with compressed source residuals and, in recent candidates, shared old-source coordinates. Calling it complete or exact without qualifying the represented state obscures the remaining scientific problem.

Finish and synthesize batch B before changing the experiment plan. Two concrete source-level limitations in its query-basis implementation deserve narrowly scoped label-free checks before concluding that query-aware compression is exhausted. Neither establishes an accuracy gain.

## What is actually shipped

| Role | File / evidence | Status |
|---|---|---|
| Root archive-compatible legacy ship file | `estimator.py`, v2.6.1 | Not the latest submitted method; the header in AGENTS is historically stale |
| Latest graded file | `estimator_k3_v31.py`, #331317 | Adjusted 8.7520303e-9; cost about 0.301 B; raw 2.907e-8; 0/100 failures in the stored report |
| Design base | `estimator_k3_v32.py` | Adds compensated distinct-index fourth-cumulant feed; not a new graded result |
| Best prior observed compression bundle | `estimator_k3_v37_a4r256_p4d5.py` | About 1.186x over v31 on P11; failed the registered 1.15x base gate versus v32 at 1.14927x; no promotion |
| Running production-oriented composition | `estimator_k3_v41_pairq_a4r256_{gram,qE}_p4d5.py` | Own PAIR+Q closure, shared age4/rank256 coordinates, P4; batch B pending |
| Diagnostic reference / swaps | Public v17/v29 and v38 reference-program chains | Attribution only under the current provenance policy |

Graded noise floors are not exposed by the server; do not interpret the raw official MSE as noiseless error. P12 uses a two-replica truth average with mean noise about 2.24805e-9 (4.4961e-9 per replica). Official-precision adjusted estimates must accompany cross-cost comparisons.

## Newest completed evidence, not the stale top snapshot

`experiments/2026-09-17b_stage1/verdict.json` establishes:

| Chain | Raw MSE on P12 | C/B | Observed adjusted |
|---|---:|---:|---:|
| v32 | 3.061004e-8 | 0.302321 | 9.254055e-9 |
| v17 | 2.513546e-8 | 0.492154 | 1.237051e-8 |
| Our chain + complete reference nonlinear program | 2.842203e-8 | 0.304371 | 8.650855e-9 |
| v17 with our nonlinear program | 3.082965e-8 | 0.489769 | 1.509942e-8 |
| v29 | 2.381050e-8 | 0.253237 | 6.031786e-9 |

All 24 networks complete in each arm. The program gives our chain 1.07698x raw gain, below the registered 1.10 attribution threshold. Deleting it from v17 gives 0.81530x raw gain, with every network worse. Thus the completed batch selects branch (b), the registered interaction test. The terminal-readout-only change has raw gain 0.99812x: the useful effect comes from recurrent state, not simply enriching the final mean formula. These are DESIGN results, not promotion or proofs of optimality.

LOG amendment 4 has already resolved the composed candidate to our `closure_pair_q.json`: the reference program wins the selection rule but is diagnostic-only; PAIR+Q is the better eligible own closure. Do not repeat the reciprocal experiment or ask for decisions it has already resolved.

## What the state representation does and omits

1. v31 retains full covariance and a complete history of its *represented* third-order births. Its dense (2,1) source leg is expressed as separable pieces plus a rank-32 residual (`_P2_RANK`). The saving is algebraically exact before residual rank reduction; the actual candidate is not an exact full third-cumulant calculation.
2. v32 adds two compensated fourth-to-third birth routes under the assumed fourth-order core model. This closes those routes under that model, not every fourth-cumulant effect of the actual activation law.
3. The regenerated fourth-order state is a per-neuron diagonal plus a lambda-scaled covariance-off-diagonal model. Harmonic projection and transport are coherent for that model. No evidence establishes that this restricted family captures the error directions that matter at the leader's precision.
4. The closure program and the regenerated state are coupled. Stage 1 demonstrates that a program which pays strongly in v17 pays less in ours. It does not alone isolate whether lambda, source construction, residual approximation or another native-state difference explains the interaction.
5. P3 includes beta1+beta2 but omits gamma, the all-distinct incoming third-cumulant times covariance link. `local_sim/s2_build_p3.py:12–16` explicitly reports that beta1+beta2 alone leaves 20% of the tiny oracle response. This is a local-model response, not 20% of final prediction error. A cheap complete representation remains an open research problem.
6. `local_sim/s11_build_candidate.py:84` deliberately refuses P3 plus shared storage. If P3 pays in batch B, compositional engineering and new integration checks are required before claiming a combined candidate. The refusal is safe; it also documents the current generator stack's integration bottleneck.

## Two concrete query-compression limitations

### The qEd negative does not test the documented D3 summand

The source header says the reserved directions protect `G_j = A_j * M_j * C_j`, with `C_j = A_j * w2b`. The actual generated consumer defines exactly this relation in `estimator_k3_v41_pairq_a4r256_qE_p4d5.py:1158` and uses the corresponding cubic product.

But `local_sim/s10_query_basis.py:182` builds:

`G = (tw.A_L[:, :n] * tw.M[:, :n]) * tw.w2b[:n]`

This lacks one `A_L` factor. The negative qEd measurement remains valid for its implemented proxy. It does not establish that reserving the actual D3-summand space is ineffective. First add a tiny direct identity check against the consumer's summand, then decide whether the already declared secondary warrants a corrected experiment. Do not silently relabel the existing negative as a test of the corrected method.

### qE keeps cross terms but retains a Wick-boundary weighting mismatch

`local_sim/s10_query_basis.py:131` stashes the left contraction factors before the Wick transform; line 116 stashes the old-source inner contraction. `_carry_stashes` at line 256 copies both unchanged. In the emitted candidate, `_wick_tower` at line 1013 scales every transported leg and Q by `w1`, then the join occurs after that scaling (lines 1630–1632).

For the retained third-order contraction, a Wick transform multiplies the (2,1) query by `diag(w1^2)` on the left and `diag(w1)` on the right. The joined legs carry the right factor. The stashed left factors still omit the `w1^2` row factor. Consequently the frozen objective is proportional to `E_pre diag(w1)`, whereas the carried post-Wick query is `diag(w1^2) E_pre diag(w1)`. Nonuniform activation gates change the right singular space through this missing row weighting.

This is a source-level objective mismatch, not a failure of the observed qE gain. A bounded follow-up should compare the stashed objective with directly recomputed post-Wick contraction on tiny arbitrary states, then assess whether scaling `_qU` and `_qold` by `w1^2` improves the already declared query diagnostics. It adds only elementwise work in principle. It still will not make the objective a whole-chain guarantee: projection also changes left factors, D3 and future transport. No new DESIGN reads are justified until the identity, price and scope are explicit.

## What a winning implementation would need

The code architecture should separate scientific state from export format: an explicit per-layer state with mean, covariance, third-source history, fourth-core state and a storage backend; one nonlinear closure interface; one source birth/compensation interface; exact dense and compressed consumers; one generated single-file export. Existing generators can remain while stable interfaces are introduced incrementally. A full rewrite would delay the actual experiment and erase hard-won parity evidence.

The score target requires both improved accuracy and better storage economics. On the dated public scale, 1.7e-8 raw at 0.13 B gives approximately 2.21e-9, close to the September-17 leading score. This is planning arithmetic only. Achieving the current 2x release target is an intermediate milestone, not evidence of winning. The present age4/rank256 route is nearer 0.22 B; modest closure gains cannot bridge the entire gap alone.

After batch B, spend at most one next registered accuracy question and one next registered compression question. Accuracy: the interaction's implicated fourth-core/source mechanism, chosen from completed results rather than another large closure menu. Compression: fix and verify the precise consumed-query boundary first, then test the single previously selected cheaper storage setting if diagnostics justify it. A stronger alternative, requiring separate evidence, is downstream-weighted query preservation: protect errors according to their propagation to final means rather than present-layer matrix norm. That is a hypothesis, not a theorem or measured result here.

Maintain exact P4 contraction savings, source/hash provenance, noise-aware economics and one frozen release challenger. Certification belongs after that challenger exists. The biggest process gain is to avoid repeated expensive evidence on obsolete candidates, not to weaken independent evaluation or interpret more submissions as scientific progress.

## Documentation cautions

The active `GAME_PLAN.md` begins with “not yet executed or preregistered” and later records execution. `CURRENT_STATE.md` still labels stage 1 running despite a complete verdict and batch-B amendment. v31's inherited header calls the lambda table both not fitted and fitted, and calls an already graded file “not a submission candidate.” These are continuity defects, not reasons to modify frozen archives. Maintain a small authoritative release/candidate manifest and reconcile current-state summaries from completed receipts; leave historical sources immutable.
