# Predicting the missing third-order correction — frontier options, 2026-09-28

Research note for Samuel's request to seek a substantial next gain. This is a ranked set of falsifiable options, **not a replacement for PLAN.md**. PLAN rev 15 was read first; the later B1/B3a and v71/v72 rows in RESULTS supply the current evidence. No experiments, training, submissions, or estimator changes were performed for this note. Sources were checked on September 28. Triangle representation/cost concerns were independently discussed with the architecture agent.

Labels: **M** = measured in this repository; **S** = supported by a cited primary source; **D** = mathematical derivation here; **P** = untested proposal. Every current-shape final-layer raw MSE quoted here has organizer-reference noise floor approximately **7.7e-11**. None of the adjacent papers below reports a comparable Phase 2 competition score. Competitor internals remain unknown.

## 1. What the new evidence actually allows us to claim

- **M:** v71, submission #332753, graded **4.6368e-9**, raw **2.159346e-8**, billed F = **472,199,314,830 = 0.214731 B**, where B = 2^41. It uses 22,701 operations. Graded median/worst wall = 83.25/94.45 s; the current noisy-day projection is 103.2 s. The root agent's later live competition check reports **v72 graded 4.62073776e-9** and the leader **1.748e-9**; RESULTS still showed v72 pending when this note was read. v72's DEV raw improves 0.42% for 291 extra operations. This is useful incremental progress, not a new accuracy regime.
- **M:** B1 finds approximately 25% relative error in the model's projected all-distinct third cumulant at deep layers, far above label noise. Giving the estimator the true all-distinct contribution at every layer reduces final raw MSE by median 33.1% on the 20 diagnostic TRAIN nets, without refitting. This is a diagnostic ceiling, not a deployable predictor.
- **M:** B3a finds deep-layer rank-64 approximations capture 81–92% of the missing D21 matrix's squared Frobenius energy. The rank-32 correction improves final raw MSE by 18.9% on eight TRAIN nets. **That arm also gets exact D3 at every layer.** Its D3-only control improves 9.5%. Therefore 18.9% must not be advertised as the measured value of a rank-32 D21 predictor alone.
- **M:** Factor gates recover mostly a scalar shrink of existing transport. Applying the diagnostic least-squares shrink without refit worsens final MSE, while the component orthogonal to existing model AD is helpful. S5's learned local marginal/pair correction was score-negative. A new learned candidate must predict a new spatial pattern, not repeat that architecture or relearn a scalar gain.
- **D:** Low matrix rank of E21 does **not** establish low symmetric CP/Tucker rank of the underlying third-order tensor, nor predictability of its singular vectors from available state. These are separate tests. D3 occupies only about 0.3% of the diagnostic matrix's squared energy yet has substantial score value; a Frobenius-only fit can ignore the valuable part.

Two additional results arrived while this note was prepared and change the next-action ranking:

- **M:** [The fresh FIT-8 feature screen](../../experiments/2026-09-28_ad_features/report.md) removes only **6–8% of deep E-orthogonal-to-model energy** with leave-one-network-out local/global scalar features, versus up to 16% for full E. This is diagnostic on previously trained nets, not independent validation, and is not downstream MSE. The label-free covariance **right** rank-32 basis captures 58–68% of deep residual energy, but the same rank-32 basis on the **left** captures only about 3%. Quadratic-left features from the leading covariance modes improve bilateral oracle capture to 25–33% at the screen's largest 128-column setting. Thus useful right modes are observable, but a generic shared linear left/right basis is inadequate. A stronger quadratic/symmetric-core test is justified; a long generic feature-MLP fit is not yet justified.
- **M:** The corrected-consumer FIT-2 audit (`experiments/2026-09-28_review_arch/consistent_fit2/net_000.json` and `net_001.json`) retains **34.86%** median AD-oracle raw improvement versus 37.63% with the legacy consumer semantics. Rank-32 plus exact D3 retains **22.59%** versus 23.06%; full D3/D21 remains 47.75%. The useful oracle signal survives the bookkeeping repair on these two nets. This does not replace the larger-panel estimates or make the oracle deployable.
- **M:** The subsequent [symmetric-core FIT-2 screen](../../experiments/2026-09-28_review_arch/REVIEW.md) fits a coherent q32 tensor to the model-orthogonal residual. At consumer layers 9/12/15 its unweighted core captures **21% / 28% / 32–34%** of residual energy. Heavily weighting D3 (1024) fits about **99% of D3**, but leaves only **1–3% / 9–11% / 16–20%** off-diagonal capture. q16 is weaker and its D3-heavy fit harms off-diagonal error. These are oracle coefficient fits on two previously used nets; none is a final-score result. The q32 high-D3-weight solves stop at 100 iterations with small but nonzero normal residual, so do not call their result a proven global optimum.

**Current next-action ranking:** (1) the symmetric residual tensor's **consistent downstream oracle test**, since its quadratic-core screen explains some but not most error; (2) one explicit omitted covariance-cycle diagram with exact repeated-slice bookkeeping; (3) a nonlocal learned matrix correction only after a better left-feature basis has useful downstream oracle value. The sections below retain the full algorithms and falsifiers, including the already-run screens. No recommendation assumes that a 20% accuracy improvement by itself wins: at current cost it does not close the gap to the live leader.

## 2. Relevant primary literature and the limits of transfer

| Primary source | Established result or implementation fact | Useful consequence here; limit |
|---|---|---|
| [ARC, Estimating the expected output of wide random MLPs more efficiently than sampling](https://arxiv.org/html/2605.05179), 2026 | Cumulant propagation and factorized/harmonic approximations address a supplied random network averaged over inputs. Polynomial-activation theory and empirical ReLU results support this family. | Direct foundation. The paper's curves/cost conventions are not Phase 2 scores; its theory does not certify our current truncation at 1024×16. |
| [ARC implementation TODO](https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/main/TODO.md), checked September 28 | Factorized AUGMENT omits non-hypertree top-slice diagrams, including covariance cycles and products of the all-distinct top cumulant with other blocks. Some omitted terms have the same asymptotic squared order as retained augmented terms. Cached results are flagged for refresh after truncation changes. | A specific missing-diagram search has a sound motivation. Broad AUGMENT restoration already priced poorly locally; select terms and representations rather than repeat it. |
| [Pan and Duraisamy, Data-driven Discovery of Closure Models](https://arxiv.org/abs/1803.09318), 2018 | Sparse polynomial and neural models infer compact closures with memory; observability/compactness statements require special system assumptions. | Use a small carried state when snapshot features cannot identify the correction. It does not prove our retained cumulants determine the missing tensor. |
| [Peherstorfer, Sampling Low-Dimensional Markovian Dynamics…](https://epubs.siam.org/doi/10.1137/19M1292448), 2020 | Under stated conditions, operator inference on reprojected trajectories recovers projection-based reduced models; arbitrary projected trajectories need not give a stable Markovian reduced model. | Do not train a recurrence only on oracle snapshots and assume it remains correct after its own corrections alter the trajectory. Our task is not covered by the theorem. |
| [Dunlavy et al., Goal-Oriented Low-Rank Tensor Decompositions for Numerical Simulation Data](https://arxiv.org/abs/2508.11139), 2025 | CP/Tucker fitting can include both reconstruction and functions/quantities of interest in the objective; numerical studies concern combustion and plasma data. | Train the **new residual** against D3, D21 and downstream score sensitivity, not energy alone. This is not a reason to rerun the closed old-tier projection unchanged. |
| [Lubich, Walach and Vandereycken, Time integration of rank-constrained Tucker tensors](https://unige.ch/math/vandereycken/bibtexbrowser.php?bib=my_pubs.bib&key=Lubich_WV%3A2018), 2018 | Develops stable low-rank tensor evolution methods, including projector-splitting ideas. | Supports a numerically controlled tensor-state representation. It gives no evidence that this particular error has low Tucker rank or a closed birth rule. |
| [Koike et al., Streaming Operator Inference](https://arxiv.org/abs/2601.12161), revised February 2026 | Incremental SVD and recursive least squares learn reduced models from streamed data; reported savings are on their dynamical-system benchmarks. | If regression is promising, accumulate small sufficient statistics by layer/net instead of loading all labels/states into host memory. This is an offline resource method, not a new estimator accuracy claim. |

The earlier [September 27 audit](2026-09-27_literature_audit_codex.md) covers neural-process Edgeworth papers, neural functionals and other closures. Its distinctions still apply: a random-weight law is not a closure theorem for a fixed network's random-input distribution; arbitrary orthogonal rotations are not ReLU symmetries. Permutation equivariance and appropriate scaling are useful constraints.

## 3. Shared notation, scoring arithmetic and test requirements

Use the repository orientation: z_next = W^T x. At a layer let C be preactivation covariance, s_i = sqrt(C_ii), alpha_i = mu_i/s_i, and D21_ij = cumulant(z_i,z_i,z_j). D3 is its diagonal. Define normalized Cbar_ij = C_ij/(s_i s_j), Dbar_ij = D21_ij/(s_i^2 s_j), with the same defensible small-variance guard as the estimator; never invert C.

E21 denotes the B1/B3a **projected missing all-distinct contribution**, label minus current model, in the next layer's D21 coordinates. It is not an observed full third-order tensor. Its row and column roles differ, so a generic low-rank E21 is not symmetric as a matrix. Its tensor of origin is symmetric.

For cost estimates, ordinary dense matrix multiplication m×k by k×n costs about 2mkn scalar operations. **The incumbent casts weights/state to float32. The f64-input grader twin does not imply that its arithmetic is float64.** Source inspection confirms dtype-aware billing and warns that accidental float64 promotion doubles the float32 bill. The doubled arithmetic numbers below are explicitly a **conservative 2× conventional-count envelope**, not a verified billed cost for a correctly float32 implementation. Actual new-path flopscope, operation, wall and memory measurements are required. Thin products may not have the dense Winograd savings. No local wall time is presented as a server time.

Above the 0.1B floor, an added fractional cost c needs fractional raw improvement g > c/(1+c) merely to break even. Thus +2% cost needs >1.96% raw improvement; +5% needs >4.76%; +10% needs >9.09%. A useful research target should leave a margin over break-even. Current wall headroom is much tighter than FLOP headroom: the new-path noisy projection rule is 105 s, and v71 is already at 103.2 s. Batch thin operations and price operation calls before a long fit; a promising accuracy mechanism might require the wall-safe code path or an independent wall reduction.

Every learned screen should distinguish three quantities: an oracle core/basis ceiling, a cross-network predicted correction, and the free-running final score after equal-step refitting against a no-new-mechanism control. Use TRAIN/VAL/DEV as presently defined in PLAN. The diagnostic eight/twenty nets were used in base training. DEV is repeatedly selected on; it is not a pristine test. The organizer run is unseen evaluation.

## 4. Rank 3 — predict the consumed correction from nonlocal state features

**P, partly screened:** This directly tests whether compact missing patterns can be predicted without constructing a full tensor. The fresh simple-feature screen is weak, so do not launch a long fit without a stronger left-basis result. It is a pragmatic recurrent-estimator correction, not a claim to an exact cumulant closure.

### Concrete construction

At the recipient preactivation layer, form a small shared per-neuron feature matrix H0 from bounded functions of alpha, standardized third/fourth marginal cumulants, Phi(alpha), and a constant. Start with 8–16 columns. Add nonlocal messages:

    HC = Cbar H0
    HD = Dbar H0
    HT = Dbar^T H0
    H = [H0, HC, HD, HT]

Apply small shared row maps to obtain F,G in R^(n×r), r = 16 or 32. These maps are permutation equivariant because the matrix state and row features permute together. A first model can use explicit polynomial/bounded features and a constant learned r×r core per layer, avoiding a neural network until its feature span is useful:

    E21_hat = diag(s^2) F A_l G^T diag(s), with its diagonal zeroed
    E3_hat_i = s_i^3 h_l(H_i)

A separate D3 head is deliberate: the useful D3 signal must not be sacrificed to a matrix-energy objective. Add these to the consumed D3/D21 at the linear-step boundary with correctly matched B1 orientation. Train the residual relative to the current incumbent, not an obsolete baseline. Zero initialization must reproduce the incumbent exactly.

If static features work but leave identifiable history error, one explicit extension is a thin carried feature bank Hmem_next = W^T diag(w1) Hmem followed by a shared bounded update. This is learned memory, **not** exact transport of a cubic cumulant. It should be tested separately rather than bundled with the first result.

### Cheap falsification sequence

1. On existing diagnostic states, QR the **label-free physical-unit factors** diag(s^2)F and diag(s)G, and compute the best oracle projected correction QF(QF^T E21 QG)QG^T. Equivalently, work consistently with normalized targets and their matching weighted norm. No SVD of E determines the inference basis. Separately fit/test the D3 head. Include the component orthogonal to model AD, because recovering scalar overshoot alone has already failed.
2. Inject the oracle projection in the diagnostic chain. If even its score value is small relative to measured overhead, stop before training. A proposed initial stop criterion is <5% median raw improvement on the pilot; this is a design threshold, not a measured effect.
3. Fit small A_l / h_l across TRAIN nets and validate on other nets. Compare fixed/global core to state-predicted core. Stream statistics instead of caching all dense states. Hold out complete nets, not entries from the same matrix.
4. Only after cross-net prediction works, fit through the rolled estimator with equal steps for a zero-head control. Score-value loss should include D3 and D21 separately; final mean loss is authoritative. An oracle basis or core is never used at inference.

### Cost and novelty

A deliberately conservative five width-n, rank-r products per layer (feature messages, one optional transport, output) cost approximately 10 L n^2 r conventional FLOPs. The 2× envelope is 20 L n^2 r. For n=1024, L=15, r=32 that envelope is **10.07e9 = 0.00458B**, or **2.13%** of current F; r=16 halves it. Narrow H0 can reduce message cost. Small row maps, normalization, QR if used online, and operation calls must be added. No dense n×n SVD is proposed in predict.

This is distinct from S5: its input includes nonlocal row/column messages, and its output targets the missing consumed all-distinct pattern, not solely local post-ReLU marginal/pair outputs. It differs from gates because F A G^T can create new matrix directions. It may still fail: the state might not identify the missing modes, or a locally accurate correction may disrupt end-to-end compensation. That failure would be useful evidence to move to explicit memory.

## 5. Rank 2 — one omitted Gaussian covariance triangle

**D/P: A concrete analytic term with a compact carrier, but bookkeeping may dominate its cost.** The ARC implementation's top-slice hypertree filter excludes covariance cycles. For Gaussian preactivation, the leading three-cycle contribution to the distinct-index post-ReLU third cumulant is

    Ttri_ijk = f2_i f2_j f2_k C_ij C_jk C_ki,   i,j,k all distinct,
    f2_i = E[ReLU''(z_i)] = phi(alpha_i)/s_i.

The coefficient is **positive one**: three second-order Hermite factors have connected Gaussian expectation 8 C_ij C_jk C_ki, canceling their 2! denominators (with the appropriate variance factors absorbed in f2). Although f2_i is positive, the product of three off-diagonal covariances can have either sign. This is not a universally positive skewness correction. The familiar tree terms f2_i f1_j f1_k C_ij C_ik are a different diagram and already represented. This derivation identifies a term; it does not show that this term dominates the measured error. At non-Gaussian later layers it is only one contribution among many.

### Compact representation and exact transport

Approximate the **full PSD covariance** C by U U^T of rank q, first q=4, optionally 8. Do not PSD-truncate the zero-diagonal C_off, which is indefinite. On all-distinct indices C and C_off agree. Set

    H_i = f2_i u_i u_i^T  (symmetric q×q),
    Ttri_ijk = tr(H_i H_j H_k).

For symmetric H this trace is symmetric in i,j,k. It transports exactly within this representation under a linear map:

    H_next,j = sum_i W_ij H_i;
    Wick scaling at a nonlinear step: H_i <- w1_i H_i;
    D3_tri,i = tr(H_i^3),  D21_tri,ij = tr(H_i^2 H_j).

Let pack preserve the symmetric-matrix inner product (sqrt(2) on off-diagonals), R_i = pack(H_i), Q_i = pack(H_i^2). Then D21_tri = Q R^T; carrier width d=q(q+1)/2 is 10 for q=4, 36 for q=8. Different birth layers need separate source blocks: tr((sum_sources H)^3) creates spurious cross-source terms. The source axis cannot be discarded for free.

### Mandatory repeated-slice subtraction

The represented trace tensor includes repeated indices, but the intended new birth is all-distinct only. Existing marginal/pair closures already account for their own repeated slices. At birth let B_ij=tr(H_i^2 H_j), t_i=tr(H_i^3). The current tower's D-block convention can cancel these exactly using

    A_D,cancel = -3 B^T + 2 diag(t).

The off-diagonal repeated slice becomes -B, and the diagonal becomes -t. Factor -3 B^T as Z L^T with Z=-3R, L=Q; add 2t to the diagonal sb contribution. This was independently checked against the architecture's `_p2_recon` / `_dslice21` conventions. A tiny dense tensor identity fixture must still precede implementation on full networks.

There is a useful sharing opportunity: the transported extra Z columns are already the packed H carrier up to -3. Reuse them; do not separately transport H. Keep the extra cancellation columns exact in the first screen rather than compressing them back into the incumbent rank-4 residual and silently breaking cancellation. If a learned scalar coefficient is later introduced, scale the positive tensor and negative canceler consistently; avoid accidentally cubing a linear coefficient.

**Unresolved integration cost:** the cancellation also depends on the birth lineage's transported identity legs M. When the main tower folds/evicts that source, those legs may no longer exist separately. Preserving a compact H alone does not preserve the subtraction. The architecture agent independently confirms there is no generic exact cheap fallback with arbitrary dense M. Keeping a new dense M for every source could ruin the price; folding the canceler together with the positive tensor is approximate and must be evaluated. Restrict the first pilot to recent/deep births still covered by existing M, or explicitly price/test the fold. A candidate starting at zero-based source layer 10 would have only 15 remaining source-layer pairs across five births, but verify the exact retained window in the code before choosing that boundary. The final layer omits some consumers, so price its actual path separately. No all-source low-cost claim is justified until this is solved.

### Cost and falsification

While a source's existing M is available, five skinny products dominate: Z transport, Z L^T, (M^2)L, its final thin contraction, and Q R^T for the positive triangle. Conventional cost is about 10 n^2 d per source-layer; the conservative 2× envelope is **20 n^2 d**. H_i^2 adds O(n q^3); covariance range finding and elementwise work add more. For illustration only, the q=4 envelope across 15 source-layer pairs is **0.00143B ≈ 0.67%** of current F before those additions; across 45 pairs, **0.00429B ≈ 2.0%**; across 120 pairs, **0.01144B ≈ 5.3%**. q=8 multiplies those figures by 3.6. These assume no new dense M transport; they are not final candidate prices or a claim that float32 arithmetic is billed twice.

1. Verify the triangle coefficient and cancellation on tiny full-rank Gaussian fixtures, then linear/Wick transport against a dense tensor. Check exact diagonal and (2,1) cancellation before any compression.
2. On existing B1 states, compare the **compensated** q=4/q=8 projected correction to E and E orthogonal to model AD. Test theoretical coefficient 1 and a single cross-net fitted coefficient separately. An unconstrained oracle fit is not evidence of an analytic prediction.
3. Pilot only the last few source layers. If it has negligible alignment with useful residual or no downstream benefit even with modest scalar refit, stop. If it helps, measure complete billed FLOPs/ops/memory including M/folding before extending birth coverage.

This is new relative to wholesale AUGMENT restoration: one explicit covariance cycle, a trace-matrix carrier, shared cancellation transport, and a selected birth window. It is also new relative to adding rank to the existing hypertree representation: the tensor structure itself changes.

### If the pure covariance cycle is insufficient: one explicit cross-cumulant term

The next connected topology to diagnose is an all-distinct preactivation third cumulant K_ijk with one covariance edge. At the lowest such Edgeworth order its contribution is

    K_ijk [f2_i f2_j f1_k C_ij
           + f2_i f1_j f2_k C_ik
           + f1_i f2_j f2_k C_jk],  f1=Phi(alpha).

Each displayed edge has coefficient one from its cumulant/exponential multiplicities; the three terms enumerate the possible pair edges. The whole graph is connected, so it survives the output-cumulant subtraction. This is exactly the type of top all-distinct cumulant times another block excluded by the factored filter. It is **not** equivalent to multiplying K_ijk by independent per-neuron gates: it includes pair-dependent C_ij. Other same-order terms can coexist, and signs/cancellations should be tested, not assumed.

Unlike the covariance triangle, this term needs the incoming all-distinct K itself. If K had a CP approximation sum_h a_h u_h^3 and C≈V V^T, one pair-edge term would be a sum over h,a of symmetrized factors (u_h f2 V_a), (u_h f2 V_a), (u_h f1). This gives an implementable mechanism but expands the carrier roughly by q per CP component. Our existing tower is not a cheap small-CP input, so claiming the same q^2 price as the triangle would be wrong. First test the omitted term's projected alignment at small width or using existing low-rank factors with a separately measured approximation; only pursue the representation if it explains meaningful error beyond the covariance triangle. This term also illustrates why a learned nonlocal correction may be more economical than restoring every connected diagram literally.

## 6. Rank 1 — a small symmetric Tucker state for the residual

**P: Highest structural coherence, highest model-design risk.** Keep the incumbent tower and add a small residual tensor, r=16 before 32:

    DeltaT_ijk = sum_abc G_abc U_ia U_jb U_kc,

where U is n×r and G is fully symmetric. This is a model of the **missing error**, not a rank-16 replacement for the incumbent rank-256 old tier. The previous large-core cliff does not establish that a small residual core fails, but neither does B3a establish that it works.

### Update and readout

For its represented component, exact linear/Wick transport is

    Ustar = diag(w1_next) W^T U;
    Ustar = Q R; U <- Q; G <- G ×1 R ×2 R ×3 R.

Respect the actual ordering of the nonlinear and linear boundaries in the implementation; the expression combines two successive operations for notation only. Thin QR needs consistent sign conventions if a learned core consumes its coordinates. If the basis is recompressed or mixed with newborn feature columns, the core must be transformed/projected consistently.

Define B_ic = sum_ab G_abc U_ia U_ib. Then

    E21_hat = B U^T;   E3_hat = diag(E21_hat).

This links the two readouts in a tensor-consistent way. It is stronger than a generic rank-r matrix model: its left factors must be quadratic in the same U that supplies its right factors.

Predict small new core coefficients from projected C, D21, alpha-weighted feature summaries and the carried core. Use a deterministic, label-free basis formed from incumbent state features and transported old U; an oracle SVD basis from E is allowed only as a diagnostic. At r=16 there are 816 independent symmetric core entries; at r=32 there are 5,984. A modest shared predictor can emit these, with a zero-birth initialization and an explicit depth input.

There are two distinct birth semantics. A full preactivation residual-state correction intentionally changes consumed D3 and D21, so it does not need zero repeated slices at that boundary; its effect must be reflected consistently in subsequent closures/tower compensation. A claimed **post-ReLU all-distinct-only birth** must subtract its repeated slices exactly as in the triangle option. Do not switch between these semantics silently. Existing slice labels do not determine the unobserved full tensor; recurrent prediction must be tested on future readouts, not only the current fit.

### Cheap falsification sequence

1. Give U an optimistic oracle basis from the right singular vectors of E, but fit a **symmetric** G to E21 and E3 jointly. If even this cannot realize a useful fraction of B3a's value, stop. B3a's generic SVD truncation is not this test.
2. Replace oracle U with a feature-predicted basis. Keep G oracle initially to separate basis predictability from core predictability. Test the next layer's contractions too; otherwise arbitrary tensor completions may fit present slices and fail after transport.
3. Only then infer birth/core updates from TRAIN states, validate on complete other networks, and fit short rolled trajectories before end-to-end fitting. Compare static projected correction, transported core without learned birth, and learned birth separately. Equal-step incumbent refitting remains the control.

For each layer, one transport and one matrix readout cost about 4n^2r conventional FLOPs; core-to-B costs about 2nr^3; three mode transforms cost about 6r^4. The conservative 2× arithmetic envelope is

    8 L n^2 r + 4 L n r^3 + 12 L r^4.

At n=1024, L=15, r=32 this envelope is about **6.23e9 = 0.00283B**, or **1.32%** of incumbent F, **before** feature/basis construction, birth prediction, QR, compensation, operation calls and any enlarged basis projection. This attractive core arithmetic is not a complete candidate budget. At r=16 the core is much cheaper. No dense n^3 tensor or full rank-256 cubic core is proposed.

## 7. Loss design that can improve any of the three

Use the final score to select a candidate, with an auxiliary diagnostic loss such as

    a ||E3_hat-E3||^2 / scale3
      + b ||E21_hat-E21||_F^2 / scale21
      + c ||J_l(delta_state - teacher_delta_state)||^2,

where J_l is the incumbent suffix's sensitivity to these state channels, used **offline**. A few deterministic adjoint probes or short suffix rollouts can estimate task sensitivity; they need not run inside predict. Start with separately normalized D3 and D21 terms before introducing a complicated adjoint scheme. Final-output-only training can also be tried after supervised initialization.

This is motivated by the measured energy/value mismatch and by goal-oriented tensor literature. It is not proof that sensitivity-weighted regression will win. It must beat an equal-budget unweighted fit and cannot reuse a task-weighted old-tier experiment unchanged under a new name. Corrected rollouts move the states, so update training data or train through the chain; teacher-state snapshot accuracy alone is insufficient.

## 8. Why a fresh integration/control-variate lane is not the first recommendation

[Li et al., Neural Control Variates with Automatic Integration](https://arxiv.org/abs/2409.15394) learns an antiderivative and differentiates it to construct an analytically integrable control variate; its applications use Walk-on-sphere PDE integration. [Oates, Girolami and Chopin, Control functionals for Monte Carlo integration](https://arxiv.org/abs/1410.2392) uses Stein-derived function spaces and density gradients. Both establish useful methods in their settings. Neither supplies a cheap control function that nearly matches a fresh 1024-input, 16-layer random ReLU MLP.

The relevant identity is E[f] = E[g] + average(f-g). The expensive network still has to be evaluated on residual samples, and learning/evaluating g also costs something. Conventional dense forward cost for N samples is about 2 L n^2 N. At our entire 0.21473B budget and L=16, n=1024, that conventional count buys about 14,072 samples before any control overhead. To achieve raw MSE 2e-8, residual variance must be at most about **2.8e-4 per output**, under independent residual sampling and zero extra cost. At 0.1B, about 6,554 samples and raw target 1.76e-8 imply residual variance at most **1.15e-4**. A float64 implementation billed at twice the float32 rate halves these sample/variance allowances. These are conditional arithmetic screens; real counted forward/control costs replace the approximation before any decision. They are not a measured variance reduction or a QMC error bound.

A genuinely new control method would first need to demonstrate that residual variance cheaply on held-out networks, with an analytic E[g] and counted evaluation cost. Relabeling closed MC/QMC/polynomial-control experiments as neural control variates does not meet that bar. Deterministic quadrature of the residual still needs a dimension/regularity advantage that has not been established. No primary source found in this audit provides that missing guarantee or a comparable leaderboard result.

## 9. Decision recommendation

The first simple-feature screen is now complete and insufficient for a long learned-correction run. The symmetric-core screen also completed: it represents some missing content, with a substantial D3/D21 tradeoff. The concrete remaining decision tree is:

1. Finish the **consistent downstream joint-core oracle** at q32, comparing unweighted and D3-weighted fits on the same diagnostic nets with the unchanged control. If even the joint oracle cannot pay the fully metered online envelope with a useful margin, close this basis/core combination. Do not respond by immediately sweeping ever-larger ranks.
2. If that oracle has useful final-score value, test whether a shared small predictor can recover its coefficients from known projected state on complete other networks. A predictor that needs per-network label fits has failed the deployment test. If only D3 value survives, compare a much cheaper separate D3 predictor before carrying a tensor.
3. In parallel, complete the **compensated triangle identity and alignment** screen, then a late-birth-only downstream pilot if alignment is material. Poor alignment closes the pure-cycle arm; it does not establish that all omitted diagrams are irrelevant. The explicit K3-times-covariance term is a narrowly defined next diagnostic only if evidence points there.
4. Price the current incumbent by actual operation family before selecting a cost screen. A separate diagonal-plus-low-rank covariance screen below is bounded and optional. A cheaper representation of the old tier alone has a roughly 28% gross ceiling, so a winning cost architecture would also need recent-history savings or a larger independent accuracy gain.

None of these screens needs a new large label corpus. Only a useful, correctly consumed oracle result justifies learning its online update. The generic matrix correction remains a fallback if richer quadratic left features prove useful without a coherent tensor core.

Do not allocate a long training run to any option before its whole-operation budget and useful oracle ceiling clear the current score/wall constraints. Stream existing artifacts, record the pre-submission optimization review, and use workerq for any substantial workload under the user machine policy. Exact formulas, strong label-oracle effects and attractive low-rank arithmetic are reasons for small tests—not forecasts of a winning score.

## 10. Separate cost hypothesis: diagonal-plus-low-rank covariance transport

**P, bounded screen only after the current exact cost profile:** a cost reduction should also consider dense covariance transport, not assume the old third-order tier supplies the whole needed saving. Both this audit and the root's active RESULTS/PLAN search did not identify a current test of the following exact-diagonal covariance approximation. This is not evidence that it has never been tried anywhere in repository history. The bigger historical cost item is recent third-order history (about 54% in an older profile; remeasure), while eliminating the current old tier entirely saves only about 28%. One covariance Gram saving may therefore amount to only a few percent of the actual total; it cannot be presented as a leader-closing result by itself.

After the nonlinear step approximate C_out by

    Ctilde = diag(d) + U U^T,
    d_i = (C_out)_ii - sum_a U_ia^2.

Then compute the next full covariance as

    W^T Ctilde W = W^T diag(d) W + V V^T,   V=W^T U.

This preserves every source variance and replaces one dense covariance sandwich with a weighted Gram matrix plus skinny products. The full n×n covariance can still be materialized for the existing nonlinear formulas; the change is in its transport. The final layer only needs sum_i d_i W_ij^2 + sum_a V_ja^2, so it avoids a full dense product there if the approximation passes. It does **not** preserve the next variance exactly: discarded off-diagonal covariance contributes to W^T C W. That error must be measured through the chain.

An offline first screen can choose U from an optimistic PSD spectral truncation at r=32/64, restore d exactly, intervene once at early/middle/late layers, and then every layer. Compare paired final MSE to the incumbent, plus next variance error. If even this accurate-basis oracle exceeds the possible cost saving, stop before implementing an online factorizer. A successful screen would still need a cheap label-free decomposition, actual metering, and equal-step refitting. Near-singular/dead-neuron directions and any negative eigenvalues of the approximate model C require explicit diagnosis, not silent PSD clipping.

For online construction, [Harbrecht, Peters and Schneider, On the low-rank approximation by the pivoted Cholesky decomposition](https://edoc.unibas.ch/entities/publication/d4eba30a-9b48-45ca-a4cb-915bfe3a83ec) gives controlled PSD low-rank approximation under its assumptions. A PSD Nyström/Cholesky factor can leave a nonnegative diagonal remainder, whereas an arbitrary orthogonal projection Q(Q^T C Q)Q^T need not be bounded above by C. Scalar pivot loops may violate the operation budget; block construction should be priced rather than presumed cheap. [Epperly, Tropp and Webber, Embrace rejection](https://arxiv.org/abs/2410.03969) studies accelerated block pivoted Cholesky; its speedups are for kernel matrices, not our grader.

A more recent primary source, [Yeon and Anitescu, Beyond Low Rank: Fast Low-Rank + Diagonal Decomposition with a Spectral Approach](https://arxiv.org/abs/2512.17120), December 2025, explicitly optimizes a low-rank-plus-diagonal representation and gives Nyström-based variants. Its approximation improvements on structured matrices/financial covariance do not establish safe neural-network mean error. Our diagonal is already available, so stochastic diagonal estimation adds no value here. Use the paper as a decomposition reference, not as a claimed competition advance.

**Cost screen:** in conventional arithmetic, the current symmetric sandwich is roughly 3.5 n^3; the proposed weighted Gram can reuse the same block-symmetry idea at roughly 1.5 n^3 plus O(n^2 r) factor transport/reconstruction and decomposition. Thus it removes roughly one full n×n product per ordinary layer before skinny overhead. This is only a structural count: our Winograd path makes that dense product cheaper than classical GEMM, so multiplying 2 n^3 by the float64 factor and calling the result an actual saving would overstate it. Meter the removed `_mm` and added factorization/readouts with the real counter. A plausible full candidate must save more than the induced raw-MSE increase; no percentage gain is forecast here.

Low effective covariance rank and the strong covariance-right-basis result for E21 do not validate this approximation. If it fails, record the failure and keep full covariance. If it passes, it could supply cost headroom for an independently helpful residual correction; it does not, by itself, establish a winning architecture.

## Later same-day experimental decision

The downstream and selected-diagram tests proposed above are now complete, and the
canonical decision is PLAN rev16 B5. The coherent covariance-core oracle worsened
paired final MSE by4.71% (weight1) or5.81% (D3 weight1024) on FIT4, while unrestricted
orthogonal AD input correction helped23.24%. Thus covariance-core moment representability
was not sufficient for final-score value. The q4/8 selected one-step Gaussian triangle
captured under0.01% of deep orthogonal D21 error even with oracle scalar on FIT2. Close
both tested correction routes before long training. Sources are the September28 RESULTS
addendum and `experiments/2026-09-28_review_arch/core_oracle_fit4/` / `../2026-09-28_triangle/`.

Next: cost-first replacement of carried history with analytically folded compact state,
then shared cross-network training against final means. This differs from adding the
failed moment-fitted correction. Its learnability and score are unproven; per-network
oracle cores can overfit and are not acceptable inference machinery. The concrete
bounded design is `experiments/2026-09-28_review_arch/DISTILL_NEXT.md`.
