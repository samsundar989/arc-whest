# Literature and research-frontier audit — 2026-09-27

Prepared for Samuel's request to understand the competition and choose the next architecture. This is a research audit, **not a replacement for PLAN.md and not authorization to start new experiments**. Read PLAN first, then current RESULTS and the Sep 27 strategist packet. No estimator changes or workloads were run. Public primary sources were searched and checked on Sep 27; the root agent separately inspected the live competition in Chrome.

Status labels: **established** = supported by a primary paper/code or a measured repository result; **inference** = an interpretation of those results; **proposal** = an untested experiment. All current-shape final-layer raw MSE numbers below have organizer-reference noise floor approximately **7.7e-11**. Public scores are different quantities from raw MSE, and paper experiments are not competition scores.

## 1. What the literature establishes

### The direct foundation: fixed weights, random inputs

The most directly applicable paper found remains Wu, Lecomte, Winer, Robinson, Hilton and Christiano, [Estimating the expected output of wide random MLPs more efficiently than sampling](https://arxiv.org/html/2605.05179). It studies a **particular supplied network**, averaging over Gaussian inputs. It develops cumulant propagation, power cumulants to handle repeated neuron indices, harmonic augmentation and factorized high-order tensors. Its theory assumes polynomial activations and fixed depth; ReLU results are empirical. Reported improvements over Monte Carlo depend strongly on width/depth. The paper uses variance-normalized MSE and a FLOP counter with hypothetical symmetric-tensor discounts; its curves are therefore not directly comparable with Phase 2 flopscope scores. Equality of depth/width ratios does not itself establish equal accuracy at different shapes. The paper explicitly distinguishes this fixed-network problem from averaging over random network weights.

**Consequence for our architecture:** mean + covariance + compressed third-order information + partial fourth-order information is a scientifically well-motivated family. The paper does not establish that our particular truncations, compression or fitted coefficients are optimal, nor that the current architecture can attain first place.

The [ARC research announcement](https://www.alignment.org/blog/mechanistic-estimation-for-wide-random-mlps/) presents the same sample-free research direction and emphasizes that sufficiently wide networks can favor analytical estimation. This supports the family choice, not any specific leaderboard forecast.

### Important correction in the upstream implementation

ARC's current [TODO.md](https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/main/TODO.md) says factorized AUGMENT drops unaffordable non-hypertree top-slice diagrams, including covariance cycles and products involving the all-distinct highest-order cumulant. It explicitly says factorized AUGMENT is **not equivalent** to unfactorized AUGMENT, and the omitted terms have the same asymptotic squared size as some retained augmented terms. The paper needs an update on this point.

**Consequence:** a missing-diagram diagnosis has a real mathematical basis. It is not enough to say that the factorized representation is exact for every augmented term. However, the repository already priced broader restoration of these terms at smaller widths and found it score-negative. This source is a reason to isolate promising omitted terms, not to repeat a full expensive AUGMENT implementation. The current [kprop_harmonic.py](https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/main/src/mlp_kprop/kprop_harmonic.py) contains the associated filtering logic.

### Finite-width neural-process papers are adjacent, not plug-in closures

[Nica and Ortmann, Improving the Gaussian Approximation in Neural Networks: Para-Gaussians and Edgeworth Expansions](https://openreview.net/pdf?id=92q7WV4od7) introduces structured finite-width corrections to Gaussian neural-network limits, including one-input and two-input ReLU cases. The direct PDF fetch encountered an OpenReview browser challenge; the indexed primary PDF text was accessible. Treat detailed theorem-level transfers with that limitation.

[Celli, Optimal Non-Asymptotic Edgeworth Expansions for Multivariate Neural Network Outputs](https://arxiv.org/html/2605.24072v1) gives finite-width distributional approximation bounds for networks evaluated on finitely many inputs, under Gaussian weight initialization and a nondegenerate Gaussian-limit covariance. This concerns the random-weight network law and Bayesian-prior approximation. It is not a theorem about closing the input-distribution cumulants of one fixed supplied network using only our stored slices.

**Correction to the earlier G review:** these papers do not establish that omitted cumulants in our fixed network are deterministic functions of the retained low-order state. Distinct distributions can share the same retained moments and disagree in omitted moments. A learned conditional approximation can still be useful, but that must be demonstrated empirically. Likewise, a relative error norm growing approximately linearly with depth does not by itself prove a signed, local, deterministic closure bias; propagation/amplification and correlated increments can produce similar patterns. The new exact-input closure bench provides stronger evidence for specific channels, and its failed in-loop follow-up is more decision-relevant than the analogy.

### Learned closures and training through the estimator

[Maddu, Weady and Shelley, Learning fast, accurate, and stable closures of a kinetic theory of an active fluid](https://arxiv.org/abs/2308.06675) learns omitted moment information in a symmetry-constrained representation and uses a differentiable solver/time-stepping constraints to obtain useful rollouts. Its physical system and symmetries differ from ReLU networks. It supports the methodology of structured corrections and online validation, not a quantitative gain here.

[Um et al., Solver-in-the-Loop](https://arxiv.org/abs/2007.00016) demonstrates why learning corrections with the solver inside training can outperform snapshot supervision: later inputs reflect earlier corrections. Our differentiable twin and S5 already test this central lesson. Citing this paper is not a reason to rerun the same failed per-neuron/per-pair closure net.

[Gupta and Lermusiaux, Neural Closure Models for Dynamical Systems](https://arxiv.org/abs/2012.13869) motivates memory-dependent corrections using neural delay models. Its positive results are in other dynamical systems. The analogy suggests retaining information lost by projection, but our own measured failure of particular recursive corrector features remains binding for those tested features.

[Zhou et al., Permutation Equivariant Neural Functionals](https://proceedings.neurips.cc/paper_files/paper/2023/hash/4e9d8aeeab6120c3c83ccf95d4c211d3-Abstract-Conference.html) is a more appropriate symmetry reference for learned functions of neural-network weights. Hidden-neuron order is arbitrary, so permutation-equivariant parameter sharing is useful. Its experiments are not this competition and provide no comparable score.

**Symmetry correction:** coordinatewise ReLU does not commute with arbitrary orthogonal rotations. It does commute with neuron permutations and positive coordinate rescalings when paired with the corresponding weight transformations. An arbitrary orthogonally invariant closure copied from turbulence could discard real coordinate information. Covariance symmetry/PSD, symmetric third cumulants, permutation equivariance, and suitable scale homogeneity are valid design constraints. The earlier G review's general claim that an orthogonally equivariant tensor basis is naturally the correct nonlinear ReLU state representation is too broad.

### Tensor compression with an actual mechanism

[Sherman and Kolda, Estimating Higher-Order Moments Using Symmetric Tensor Decomposition](https://arxiv.org/abs/1911.03813) develops implicit symmetric CP fitting of moment tensors without constructing the dense tensor. The [authors' Tensor Toolbox implementation documentation](https://www.tensortoolbox.org/cp_isym_doc.html) exposes iterative fitting methods and recommends multiple starts. This gives a practical decomposition mechanism, but does not show that our third-cumulant remainder has small CP rank or that per-network iterative fitting fits the grader budget.

[Wang et al., Fast and Guaranteed Tensor Decomposition via Sketching](https://arxiv.org/abs/1506.04448) computes approximate tensor contractions and decompositions using sketches. Its guarantees and experiments do not establish low approximation error for our non-orthogonal, signed cumulant tensor. Sketch-building and reconstruction costs must be included, as must accumulated error through subsequent layers.

[Han, Avron and Shin, Polynomial Tensor Sketch for Element-wise Function of Low-Rank Matrix](https://proceedings.mlr.press/v119/han20a.html) approximates elementwise functions of low-rank products with polynomial/sketched features. This is relevant to the rank growth caused by products of features in our tensor readouts. It is not a ready-made recurrent state representation for arbitrary weight matrices.

[Improving TensorSketch Using Complex Random Variables](https://arxiv.org/html/2608.10523v1), a recent August 2026 preprint, improves sketching analyses/constructions for polynomial features. It is a possible implementation reference for a later sketch experiment, not evidence of a competitive estimator. Complex arithmetic, supported flopscope operations and sketch variance would need pricing before use.

**Key distinction:** a sketch that cheaply estimates a tensor contraction once need not be cheaply updatable under an arbitrary dense weight matrix. One must explicitly derive the operations for birth, linear transport, ReLU scaling, merge/recompression, and both required readouts. Otherwise a claimed saving merely moves the expensive operation elsewhere.

## 2. The public implementation frontier

The strongest detailed Phase 2 implementation disclosure located remains [504aldo's published account](https://discourse.aicrowd.com/t/everything-we-tried-a-factorized-k-3-cumulant-propagation-estimator-at-0-25-x-b-where-its-flops-go-and-25-measured-dead-ends-team-504aldo-rank-10/18218), with [public code](https://github.com/504aldo/whest-p2-cumulant-k3). It reports adjusted **5.40e-9**, raw **2.13e-8**, and cost **0.2526 of budget** for its disclosed state. That is a dated public result, not a fresh leaderboard position. Our current best adjusted **4.7039e-9** beats that disclosed adjusted score, primarily with a lower bill; our raw **2.1906e-8** does not establish an accuracy advance over its quoted raw. Matching data and current grader conditions is required for a direct accuracy comparison.

A second public Phase 2 write-up, [exact bivariate ReLU covariance plus calibration, submission 329951](https://discourse.aicrowd.com/t/phase-2-write-up-exact-bivariate-relu-covariance-plus-per-layer-calibration-3x-over-the-kit-baseline-with-negative-results-submission-329951/18214), reports adjusted **1.32e-7**, versus **3.93e-7** for its starter-kit control. It is useful evidence that a mathematically exact pairwise Gaussian formula does not solve the whole non-Gaussian propagation problem, but its score is well behind the current K3 family.

[James Henry's Phase 1 trajectory-calibrated chain](https://github.com/jamesrahenry/arc-whitebox-replication/blob/master/WRITEUP.md) is relevant prior work on fitting corrections along the actual propagation trajectory and on the value of all-distinct third-order information. Its Phase 1 shape, grader and errors differ from Phase 2; do not place its results in a same-task score comparison without qualification. This prior work also means broad claims that learned corrections to moment propagation are novel would be unsafe.

The [public d8b corpus](https://huggingface.co/datasets/keenanpepper/arc-whestbench-p2-d8b-corpus-14k) contains Phase 2-shaped networks, Monte Carlo mean labels and teacher information. Our running S2t3 already uses part of this resource. More labels alone have not yet demonstrated a large jump; teacher-label accuracy and independence of final validation must be tracked.

No source found in this audit discloses the current leaders' complete algorithms or an academic estimator with a verified Phase 2 score below them. The live board demonstrates superior submitted performance, not the mechanism behind it. Their lower cost is evidence that better tradeoffs exist, but does not prove they use our family with a free old-history tier.

## 3. Repository results that change the research priority

Current measured evidence is in RESULTS.md sections beginning at lines 225, 247, 275 and 321, and docs/STRATEGIST_PACKET_2026-09-27.md. It overtakes PLAN rev 14's forecasts.

- **Radial-only harmonic compression failed.** Best tested radial arm on DEV100 is about **5.93 times** baseline raw error. Dropping the old tier is about **17.8 times** worse. This rejects the tested radial summary, not every possible compression of the nonradial remainder.
- **The tested learned closure failed on score.** S5 improves raw by approximately **0.75%** but costs **1.68%** more FLOPs. The nearly free per-neuron-only arm matches continued training of the old model. High offline residual R-squared did not translate into a useful new estimator.
- **Oracle refit diagnostics still expose headroom.** Replacing third-order, covariance, or fourth-order consumed channels with high-accuracy labels, then refitting, reduces raw by **53.4%, 40.5%, or 27.9%** respectively. These are diagnostic ceilings on previously used FIT data, not deployable improvements or independent estimates of unseen-data performance. They are interacting channels, so percentages must not be added.
- **All-distinct third-order content is a plausible next target.** The marginal/pair closure bench does not explain the third-order drift. However, it does not directly observe the complete third-order tensor. Attribution specifically to new all-distinct births rather than transport/amplification is still an inference to test.
- **Conventional capacity knobs are locally exhausted.** Age/rank changes trade accuracy almost directly against the scoring multiplier. A different way to create or represent useful third-order information is more promising than another nearby age sweep.

There is an important arithmetic limit on the old-tier narrative. A hypothetical lossless, zero-cost old tier reduces **0.21473 B to 0.15450 B**, based on the measured drop-tier cost. Against the Sep26 leader cost **0.119 B**, that remains about **30% more compute**. Holding our current raw constant, it would score approximately **3.38e-9**, not the leader's **1.90e-9**. This is a hypothetical product calculation, not a measured candidate or paired improvement. Even perfect old-tier compression alone would not close the full gap.

## 4. Decision-relevant probes, not a new experiment schedule

### A. Audit the all-distinct error in the form the next layer actually consumes

**Proposal; first diagnostic priority.** Avoid constructing a dense 1024-cubed tensor. The immediate consumer only needs contracted marginal and two-index slices after the next known weight matrix. Existing post-ReLU marginal/pair third moments describe the repeated-index part; next-layer preactivation D3/D21 labels describe the complete transported tensor. By multilinearity, contract the repeated-index part through the known weights and subtract it from the complete target. This yields a target for the **transported all-distinct contribution**, subject to consistent centering, conventions and Monte Carlo noise.

This can test whether the error resides in the omitted content that the scorer actually uses. It does not alone separate inherited all-distinct error from incorrect new births. For that separation, one needs selected additional projected-moment labels or a trustworthy lower-dimensional complete-tensor diagnostic. Any newly generated high-budget labels must be streamed and brokered through workerq; no such job is authorized by this document.

For one concrete check, with `W[a,j]` the current-to-next weight and `postK21[a,b] = cumulant(X_a,X_a,X_b)` off the diagonal, the repeated-index contribution to next-layer D3 is `sum_a W[a,j]^3 postK3[a] + 3 sum_a W[a,j]^2 (postK21 @ W)[a,j]`. Subtract this from the next-layer labeled D3. This needs an n-by-n contraction rather than an n-cubed stored tensor. The architecture agent independently derived the same diagnostic. K21's N=1e8 label noise requires explicit attention.

Measure error after contraction as well as tensor-norm error, and compare the target's noise floor. Strong cancellation can make a large tensor component irrelevant or a small component decisive. Do not equate fraction of removed final MSE with fraction of tensor information retained.

### B. Modulate the existing third-order factor vectors, rather than another pair net

**Proposal; potentially cheap accuracy probe.** In v68's all-distinct path, source births use `fac1 = w1 * C11` and a birth Wick vector `w2b`; inherited sources are multiplied by `w1` in `_wick_tower`. Fourth-order feed hubs use `cA/cB`. The S5 experiment modifies marginal/pair K outputs, and existing fitted feed gains are layer scalars. A small shared function of normalized neuron-state features could instead adjust one of these **vector factors**.

For example, start with a bounded, identity-initialized gate on the new two-leg birth coefficient only. The gate can depend on alpha and normalized skew/kurtosis, with optional already-available row summaries. Keep the tensor's symmetric factorization, and recalculate its repeated-index compensation using exactly the same modified factors so the marginal/pair cumulants are not counted twice. Separate birth-only, inherited-transport-only and feed-only arms; do not change them together initially. Train against the rolled-out final score through the existing twin and compare with a no-new-parameter continued-training control.

Why it differs from S5: the corrected object is the all-distinct third-order factor block, not another estimate of K1/K2/K3/K11/K21. The potential extra state/compute is O(n) or O(n squared) elementwise scaling, rather than an additional O(n squared) MLP. Why it can fail: the missing tensor might not lie in the span of the existing factors, and more flexible gains may simply reproduce current compensation. This is a hypothesis, not a claimed gain; architecture review should verify that an equivalent arm was not already run.

The parallel architecture audit confirmed this distinction in actual v68 code: `fac1/w2b` near line 3996, `p111_3/p111_21` near 4004, repeated-slice subtraction near 4049, and inherited `_wick_tower` near 4055. It found no state-dependent full-K111 gate in S5. One additional validation requirement follows: first check repeated-slice preservation using a full-rank P2 residual control, since the shipped rank-4 residual compression can introduce a secondary error after algebraic compensation.

### C. Test a compact traceless remainder before designing its online compressor

**Proposal; high risk, potentially important cost probe.** A symmetric CP tensor has form `T = sum_r lambda_r v_r tensor v_r tensor v_r`. Multiplication by a weight matrix transports each vector exactly without increasing CP rank. Its D21 matrix is `(V squared elementwise) diag(lambda) V-transpose`, hence has rank at most r before killing the diagonal. New ReLU births increase rank; recompression is the difficult operation.

A useful optimistic screen is the singular-value tail of the remainder's **full D21 readout including its diagonal** (D3 supplies the diagonal). If even an unconstrained matrix of rank r cannot approximate the required readout in a useful metric, an r-term symmetric CP tensor cannot do better on that readout. This is a necessary condition, not a sufficient one: CP is more constrained, and preserving one current slice does not preserve future weighted slices. Diagonal masking can increase matrix rank, so the test must restore the diagonal before interpreting the rank bound.

Test the remainder after the radial part, across source ages and several subsequent weight transports, with r in a modest ladder such as 8/16/32/64. Use the tensor implicit contractions available from the current factor representation; do not materialize a dense cube. Price vector transport, slice extraction, source count and recompression together before investing in an online CP fitter. The old-tier rank256 cliff makes a rank16 miracle doubtful; literature on generic CP decomposition does not negate that empirical warning.

### D. Isolate omitted connected diagrams with cheap structure

**Proposal; diagnosis before implementation.** The upstream factored-AUGMENT caveat points to explicit covariance cycles and interactions missing from the all-distinct tensor. Inspect which are absent from our own symbolic generator and determine their contracted effect using exact current-state labels on a few existing networks. Select terms by score-relevant effect per counted cost, not solely asymptotic order.

Low-rank covariance factors can sometimes represent a cycle more cheaply than a dense triple tensor. But elementwise products create feature-rank growth, and the low effective rank of covariance does not prove that a signed third-order correction can be compressed with comparable rank. This direction should be closed quickly if a cheap oracle contraction has little effect or if its full carried cost exceeds the attainable score gain. It differs from simply restoring every expensive AUGMENT term.

### E. Sketching is a later branch, conditional on a usable update law

**Proposal; lower priority.** TensorSketch and polynomial feature sketches could approximate the quadratic readout of a factored tensor. Before a performance run, write down a complete update law and check determinism under `mlp.seed`, the permitted operation surface, and precision/cost. Unbiasedness of a single contraction does not guarantee unbiasedness after nonlinear feedback; sketch noise must be measured through the full chain. Nothing in the reviewed papers supplies a ready-to-use winning solution here.

## 5. What should not be claimed

1. We have the winning architecture. We have a strong disclosed-family implementation with significant unexplained competitive gaps.
2. Academic Edgeworth error rates or equality of L/n prove a Phase 2 score. They do not.
3. The top teams have a free old tier. Their code is not public evidence, and the cost arithmetic is not explained by that tier alone.
4. All closure learning is dead, or conversely that another generic closure MLP is likely to win. The specific S5 mechanism failed; a genuinely different tensor target remains open.
5. Exact-label substitutions are achievable predictors. They are finite-Monte-Carlo oracle diagnostics, with parameter refitting and validation-selection caveats.
6. A good offline tensor approximation guarantees a lower adjusted score. Arithmetic, residual dispatch, wall time, memory and propagated bias all matter.
7. This audit proves novelty. There is relevant published competition work on trajectory correction and all-distinct cumulants. A contribution claim needs a specific independently derived mechanism and a measured benefit, or a precise useful negative result.

The research conclusion is that the architecture family remains credible, while **the current representation and approximation are not validated as sufficient to win**. The most defensible next research questions concern the creation, transport and compression of all-distinct third-order information, with low-cost factor modulation and explicit contraction diagnostics ahead of a large replacement architecture.
