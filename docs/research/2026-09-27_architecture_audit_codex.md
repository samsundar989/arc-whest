# Architecture audit — 2026-09-27

Read-only code and evidence review, requested by Samuel as part of the competition/frontier review. No estimator, harness, data panel, or experiment was changed and no expensive job was launched. This is an audit and diagnostic proposal, not a replacement for `docs/PLAN.md`.

## Conclusion

We have a sophisticated, working **deterministic statistical solver with small learned corrections**. We do not solve the challenge by drawing lots of random inputs, and the submitted estimator is not a large trained neural network. It propagates a compressed description of the activation distribution through the supplied network.

The implementation has substantial measured engineering value: it preserves important third-order history, reduces the cost of transporting that history, and learns small corrections through the same computation it submits. The graded results support its correctness and usefulness at the competition shape. They do **not** establish that this representation can reach the leaders' accuracy/cost tradeoff. Two major proposed improvements have already failed, although PLAN and CURRENT_STATE still describe them as future work.

The most useful untested diagnostic is to measure the **all-distinct third-cumulant contribution in the directions the next layer actually consumes**. The existing closure bench only directly checks marginal/pair slices. A cheap, state-dependent change to the factored third-order births or inherited transport would address a different object from the recently rejected closure networks. Its value remains unknown.

## Current shipped artifacts and evidence

The scored-layer ground-truth noise floor used in this report is approximately **7.7e-11 MSE**, measured from two independent billion-sample label sets; it is approximately 0.35% of v68's scored-layer raw MSE. The first-layer floor is different, approximately 6.8e-10. Do not apply the first-layer number to the final layer.

| role | source / submission | graded score | final raw MSE | FLOPs / 2^41 | measured server wall |
|---|---|---:|---:|---:|---|
| best score A candidate | `estimator_k3_v68_wino.py`, #332478 | 4.7039e-9 | 2.190577e-8 | 0.21473 | median 84.00 s, worst 93.86 s |
| latest safer B candidate | `estimator_k3_v68_ws.py`, #332544 | 5.0664e-9 | 2.190628e-8 | 0.23127 | median 51.89 s, worst 55.67 s |
| previous B | `estimator_k3_v64_ws.py`, #332364 | 5.2699e-9 | 2.271497e-8 | 0.23200 | median 55.27 s, worst 63.55 s |

Evidence: `RESULTS.md:220–222`. Each graded run completed 100/100 networks with zero failures. A's observed worst is not a guarantee about the private rerun. Its median times the existing noisy-day factor gives approximately 104.2 s; B's gives approximately 64.3 s. These are projections, distinct from measured walls.

The two v68 files differ by exactly one source line: `_STRASSEN_MIN_BLOCK = 32` versus `64`. Their small raw-MSE difference is consistent with floating-point reassociation. They implement the same statistical method.

The current v68 A source SHA256 is `98e40c83a728522507a6f98cd4a8270bf2df8a7236b73e4ec7af5de0167a46bb`; it matches `estimator.py` inside `experiments/2026-09-26_v68_canary/v68-canary.tar.gz`. B's packaged source SHA256 is `0db9afba06d25ebf17e4eaafb9d6356ba98fa3fd91449d8757efdb472cd7b9bc`.

The earlier `v68_ev.json` records hash `56b92735`, before a memory-lifetime repair. The source difference is deletion of eight temporary array references before recursive multiplication, not changed arithmetic. `v68_mem.json` tests the final `98e40c83` source. Do not mistake the earlier combined evaluation artifact for the final memory evidence.

## What the estimator actually does

For input `x ~ N(0,I)`, each given layer computes `h_l = ReLU(W_l^T h_(l-1))`. The desired result is the vector of average activations at each layer. The solver instead carries the following summary of the current random activation vector:

| carried object | plain-English purpose | implementation |
|---|---|---|
| mean vector | expected activation of each neuron | `mu` in `_layer`, `estimator_k3_v68_wino.py:3831` |
| full covariance matrix | how pairs of neurons vary together | `cov`; transport at lines 3846–3856 |
| factored third-cumulant history | asymmetric, non-Gaussian dependencies that affect later means | `_Tower`, line 932; `_advance`, line 1156; consumers at lines 1354 and 1379 |
| diagonal fourth-order core plus a model for its off-diagonal entries | an inexpensive approximation to additional non-Gaussian structure | `gk4`, `dG`, lambda profiles, lines 3870–3916 and 4151–4186 |
| small fitted tables and neural correction pages | compensate systematic errors in the approximations | `_closure_apply`, line 557; `_corr_delta`, line 3770 |

One layer has five stages:

1. **Transport through the weights.** Compute the next mean and covariance, and multiply the factored third-order state by the layer matrix. This linear transport is algebraically exact for the state represented, subject to numerical rounding.
2. **Approximate the ReLU transformation.** Gaussian/Wick formulas plus finite cumulant correction terms estimate post-ReLU means, variances, pair covariances, and selected third/fourth cumulants. This is a truncated closure, not an exact integration of the actual activation distribution.
3. **Correct the mean.** A tiny learned model adds a correction to `mu_out` before that mean enters the next layer. It is part of the recurrent computation, not a correction added only to the final answer.
4. **Create a new third-order source.** The new source combines the marginal/pair residuals, a covariance-driven two-leg contribution, and selected fourth-to-third-order hub terms. Previously born sources are transported forward rather than recreated from scratch.
5. **Compress older history.** Sources older than five layers share a rank-256 basis. Newer sources keep the full-width factored representation; the residual component of each birth is itself approximated at rank four.

The final layer skips state components used only by later layers. The first output row is replaced by the known closed-form expectation `||W[:,j]|| / sqrt(2*pi)` (`predict`, lines 4291–4303).

### What “exact tower” does and does not mean

Many inherited comments call the model an “exact tower.” This describes identities for carrying its chosen factors, and sometimes distinguishes it from dropping generations. It does **not** mean exact third-cumulant propagation or exact expected activations. The rank-four residual, rank-256 old tier, truncated nonlinear closure, approximate fourth-order core, and fitted gains are genuine approximations. `_SHARED_AGE`, `_SHARED_RANK`, and `_P2_RANK` are set at lines 93, 94, and 110.

### The learned component is small

The mean corrector has one parameter page for each of 16 layers. Nine measured features become 33 features after squares and selected pair products, then pass through a **33 → 32 → 32 → 1** tanh network. This is 34,832 trainable weights/biases in total, plus standardization constants. Features include normalized mean, skewness/kurtosis summaries, weight-column norm, third-order pair energy, and the uncorrected mean (`_corr_raw`, line 3735; `_corr_delta`, line 3770).

The differentiable training twin executes the estimator's source against a PyTorch implementation of the array interface, with narrowly targeted hooks for trainable coefficients (`tools/twin/twin.py:1–32`, `319–390`). It is not an independently rewritten solver. Training uses one network at a time, per-layer checkpointing, and gradient accumulation. The submitted file contains only constants and allowed runtime operations; PyTorch is an offline training dependency.

Current fitting is end to end against final-layer MSE, with optional intermediate-layer loss (`tools/twin/twin.py:27–29`, `612–615`). The old architecture report's statements that fitting is only greedy and that end-to-end learning has never run are obsolete.

## Where cost and accuracy are concentrated

Most arithmetic is spent transporting and consuming the third-order history. The older site census put the exact tier near 54% and old tier near 25% of total FLOPs. The newer v68 drop-old-tier measurement directly prices the old tier at approximately **28%** of v68's bill (`RESULTS.md:321–361`). These are different measurements on successive implementations, not identical decompositions.

Matrix products use batched Strassen-Winograd: seven smaller products replace eight, and the seven are issued together to avoid excessive metered operation calls (`_sb`, line 1051). At the same time, small products are less efficient on the grading server. Thus leaf 32 reduces billed arithmetic, while leaf 64 is substantially faster in real seconds. A uses 472,199,315,449 billed FLOPs and 22,701 calls; B uses 19,361 calls. Final memory gates report approximately 6,736 MiB for A and 5,085 MiB for B.

The runtime implementation is float32 after casting the immutable grader weights. Large arrays are handled only through the allowed instrumented interface. The source imports stdlib, flopscope, and whestbench; no numpy/torch import appears in the submitted estimator. `_p2_compress` chooses sketch columns by their norms, deterministically, rather than sampling (`line 1204`). There is no broad exception-to-constant predictor fallback; the setup exception handler retries table construction inside predict.

The strongest measured accuracy fact is that old third-order content cannot simply be discarded. The harmonic experiment reduced cost but lost far more accuracy. A more compact representation must retain the directions that matter, not merely preserve a scalar size statistic or radial trace.

## Completed experiments that the plan has not caught up with

All absolute MSEs in this section refer to the final layer, whose organizer-label floor is approximately 7.7e-11; diagnostic labels have their own sampling uncertainty described in the underlying artifacts.

* **Harmonic old tier is closed.** The best tested radial-only arm is approximately 5.93 times v68 raw MSE across DEV-100; deleting the tier is approximately 17.8 times. Its cost saving is not competitive. `RESULTS.md:321–361`.
* **Refit ceilings are measured.** Replacing selected channels with exact labels and refitting gives approximately 40.5% improvement for covariance, 53.4% for third cumulants, 27.9% for fourth cumulants, and 91.4% for all but mean. These are artificial information ceilings, not attainable candidate gains, and are not additive. `RESULTS.md:225–245`.
* **Generic marginal/pair closure calibration has been tested.** Offline residual prediction is good, but the embedded and fine-tuned S5 model improves raw MSE only 0.75% while costing 1.68% more FLOPs. The per-neuron-only variant matches simply continuing the existing model's training. `estimator_k3_v69_clos.py` was not submitted. `RESULTS.md:247–273`.
* **Extra training is already in progress in the recorded workspace.** `experiments/2026-09-27_S2t3/train.log` contains a continuing run using the existing parameter family and additional public corpus data. Its final result should be checked before authorizing duplicate work; this audit did not inspect or alter the running process.

`docs/CURRENT_STATE.md` still names v64_ws as B and describes completed work as in flight. PLAN rev 14 still anticipates the harmonic/refit/closure experiments. RESULTS is newer evidence and should inform the next revision. The older September 25 architecture report also has stale frontier, fitting, and label-noise statements.

The actual twin manifest currently uses full rows 0–899 for training and 900–999 for validation; mini is separate DEV (`data/panels/twin_train_split.json`, `tools/twin/twin.py:123–140`). The manifest records the prior authorization for expanding training. Accordingly, do not describe all 900 non-FIT full rows as pristine holdout today, or call repeatedly selected validation data an untouched final test.

## Fourth-order matrix core: avoiding the forum terminology trap

The current code does **not** construct a full fourth-order matrix core from the `(2,2)` slice. It constructs its **diagonal only**.

For a symmetric fourth-cumulant tensor `T`, let `H_ij = sum_k T_ijkk`. The Euclidean harmonic matrix-core projection has the form

    G = 6 H/(n+4) - 3 trace(H) I/((n+2)(n+4)).

On the diagonal, `H_ii = T_iiii + sum_(k != i) T_iikk`, so K4 plus the row sum of K22 is sufficient. That is exactly the `g_out` computation at v68 lines 4180–4186.

Off the diagonal, however,

    H_ij = T_iiij + T_ijjj + sum_(k != i,j) T_ijkk.

This needs `(3,1)` terms and the `(2,1,1)` hub contraction. K22 alone does not supply it. Our implementation replaces off-diagonal core information by a lambda-times-covariance model. Its transported diagonal is computed using the previous post-activation covariance and the current pre-activation variance; its Q31 consumer uses current pre-activation covariance directly, plus the five learned correction modes (`_closure_apply`, lines 596–604; `_layer`, lines 3900–3916).

The September 22 forum clarification reported by the lead researcher therefore identifies an important missing object, but does not by itself demonstrate an indexing or projection bug in our diagonal formula. The full fourth-order core and its source history are not what our model carries. Do not equate K22, the full G matrix, its diagonal, and the directly modeled pre-activation Q31 slice.

## Recommended next diagnostic: the third-order content not seen by the closure bench

The existing bench checks K1, K2, K11, K3, K4, K21, and K22. It does not directly measure the outgoing third-order tensor on three distinct neurons (`tools/closure_bench.py:1–27`). Its marginal K3 and pair K21 residuals are too small/noisy to explain all the observed third-order drift (`RESULTS.md:295–301`). This motivates a targeted measurement; it is not a proof that all drift is created in new births.

Let `T_abc` be the centered post-ReLU third cumulant, `t_a = T_aaa`, and `S_ab = T_aab` with `S_aa = 0`. For next-layer weights `W` in the repository convention, the exact next-layer marginal third cumulant is

    D3_next[j] = sum_(a,b,c) W[a,j] W[b,j] W[c,j] T[a,b,c].

Its repeated-index contribution is computable without storing an n-by-n-by-n tensor:

    repeated[j] = sum_a W[a,j]^3 t[a]
                  + 3 sum_a W[a,j]^2 (S @ W)[a,j].

Therefore

    all_distinct_projection = labeled_D3_next - repeated.

This uses the existing post-K3/post-K21 and next pre-D3 labels, one dense matrix product, and n-squared storage. Compare it with the same projection from the model's outgoing factored tensor. This tests the directions the very next layer actually consumes, rather than a generic tensor norm. The analogous repeated-index contribution to the next D21 matrix is

    (W*W).T @ (t[:,None]*W)
    + (W*W).T @ (S @ W)
    + 2 * (W * (S @ W)).T @ W.

Here D21 uses the first output index twice, matching `tools/anatomy2.py:16–23`. These identities follow by partitioning the three source indices into all-equal, exactly-two-equal, and all-distinct cases; verify orientation on a tiny symmetric dense tensor before treating a measurement as evidence.

Important limits:

* The post-K21 label has N=1e8 samples while many other labels have N=1e9. Measure the projected noise using available independent replicas or an explicit uncertainty estimate. Cancellation can amplify relative error.
* Exact marginal and pair inputs are **not an exact incoming third-order tensor**. The existing carried all-distinct content remains approximate. A projected discrepancy isolates a missing tensor component, but does not alone separate newly created birth error from inherited transport error.
* Compare birth and inherited-transport interventions separately on common states. An annealed finite-width result averaged over random weights is not a proof of the required fixed-weight, per-network closure.

## A distinct, cheap candidate family to consider after the diagnostic

Current all-distinct factors use fixed Wick vectors: `fac1 = w1[:,None] * C11`, `w2b = w2`, and inherited transport through `_wick_tower(w1[:,None], adv)` (approximately lines 3996–4057). Existing per-layer term gains and S5's K-dictionary corrections do not provide state-dependent gates on these full-tensor factors.

A small per-neuron function of already available normalized state could gate the two-leg birth, hub coefficients, or inherited Wick transport while leaving the factor count unchanged. That could be cheap enough to matter. This is a proposed family, not an implemented or measured gain.

Every changed full-tensor factor must be reflected in the repeated-slice subtraction (`p111_3`, `p111_21`, then `S3 = K3 - p111_3`, `S21 = K21 - p111_21`). Otherwise the candidate double counts or changes the already modeled slices inadvertently. First verify exact repeated-slice preservation in a tiny full-rank P2 control; then measure the additional error introduced by the actual rank-four compression. Fit in the existing differentiable twin and compare against an equal-training control. This addresses a different representation degree of freedom from merely increasing the failed S5 pair network.

## Provenance and review scope

This is not an entirely independent-from-public-work architecture. `docs/PROVENANCE_504aldo.md:146–158` records the public fourth-core regeneration, separable regrouping/range finder, and shared-basis mechanisms, alongside the project's own representation and implementation contributions. The submitted source contains no import of `external/504aldo`. This audit checked source structure and package identity, not a complete byte-level originality comparison.

No decisive implementation defect was established in this audit. The main limitations are representational and evidentiary: compressed third-order information, incomplete higher-order state, state-dependent compensation already partly saturated, and documentation that lags completed experiments. Preserve the successful solver as the control while testing a genuinely different, well-localized mechanism.
