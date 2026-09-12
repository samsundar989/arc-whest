# Public frontier update — September 12, 2026

> This independent source update preceded the local discovery of exact D21 product fusion. Its public-source findings stand; the integrated action order is [STRATEGIST_RESPONSE](../STRATEGIST_RESPONSE_2026-09-12.md) and [GAME_PLAN](../GAME_PLAN.md): the small own-code contraction candidate first, faithful compact storage next.

**Research evidence and proposed experiments; not a promotion or write-up claim.**
Checked public sources at approximately **16:47 UTC / 11:47 CDT**. This review
read repository context, public code, API metadata and primary papers. It did not
run external estimators, mint challenge networks, read labels, change an
implementation, or access the vault. Public author measurements below are not
our reproduction. The standing candidate remains untouched.

## Decision supported by the evidence

The shortest credible improvement route is still a precisely defined compact
K3 baseline, followed by an explicitly approximate and internally consistent
fourth-order closure. A finite exact signed-law representation is a separate
research problem. Its correctness as distributional accounting does not make it
a prerequisite for a useful estimator, and its failure to compress does not
reject approximate cumulant estimation.

The main new findings from this reread are limitations that change the experiment:

1. There is **no new 504aldo code or forum reply** after our September 10 pin.
2. The published small-shape smoke path does **not exercise its regenerated K4
   closure**. It switches the closure off outside width1024/depth16.
3. The public code's variance floor is not a covariance PSD or moment-realizability
   guarantee. Its observed public performance supports testing the approximation,
   not calling it a stable physical source on arbitrary shapes.
4. “V16b,” “NO_REGEN” and “K3-only” are not interchangeable controls.
5. The latest ARC paper is **v2, May 14, 2026**; our September 10 review inspected
   v1. There is no newly published September theorem settling ReLU closure.

## Freshness and exact provenance

| Public source checked September 12 | Observed state | Dated primary URL |
|---|---|---|
| 504aldo main | `18c17e2d7a9aeacd399cfc2c6b571e4e16dbfb45`, September 10 03:24:28 UTC | [Commit](https://github.com/504aldo/whest-p2-cumulant-k3/commit/18c17e2d7a9aeacd399cfc2c6b571e4e16dbfb45) |
| Compare pinned commit with main | `identical`, ahead 0, behind 0, total commits 0 | [Comparison API](https://api.github.com/repos/504aldo/whest-p2-cumulant-k3/compare/18c17e2d7a9aeacd399cfc2c6b571e4e16dbfb45...main) |
| Forum topic 18218 | One post; created and last updated September 10 **03:23:10.653 UTC**; no replies | [Topic API](https://discourse.aicrowd.com/t/18218.json), [readable topic](https://discourse.aicrowd.com/t/18218) |
| ARC main | `93d091a4c26c042bfffa28f2e76a81bc0aba94bb`, August 4 20:54:27 UTC | [Commit](https://github.com/alignment-research-center/mlp_cumulant_propagation/commit/93d091a4c26c042bfffa28f2e76a81bc0aba94bb) |
| Wu et al. | Latest arXiv v2, May 14 22:12:20 UTC | [Version history](https://arxiv.org/abs/2605.05179), [v2 full text](https://arxiv.org/html/2605.05179v2) |

The forum's September 9 description in the public README is compatible with
September 10 UTC. The API timestamps above avoid that timezone ambiguity. The
freshness claim covers these named public surfaces, not private work or all
competition discussion.

ARC's August 1 correction explicitly separates factored AUGMENT from unfactored
AUGMENT: some extra diagrams cannot be represented by its factorization. Tests
compare against an unfactored implementation restricted to the same terms.
Preserve that distinction in every control. [Correction commit](https://github.com/alignment-research-center/mlp_cumulant_propagation/commit/56f736a7e74c60d9714fa06bd5cb6bf48217a6d8).

## What the public measurements actually support

These are rounded values reported by the author. **A numerical ground-truth
noise floor is unavailable for every row.** The eight public dumps were reused
for choices and fitting; they are development evidence, not fresh DECIDE data.

| Author's control or intervention | Reported raw final MSE | Reported FLOPs / `2^41` | Reported adjusted score |
|---|---:|---:|---:|
| V16b, eight dumps | `3.70e-8` | `0.4986` | `1.85e-8` |
| V17, regenerated K4, eight dumps | `2.41e-8` | `0.4921` | `1.19e-8` |
| V19, final-layer trim, eight dumps | `2.19e-8` | `0.4777` | `1.05e-8` |
| V25, compact method, eight dumps | `2.270e-8` | `0.3667` | `8.32e-9` |
| V29, author-reported graded submission #330093 | `2.13e-8` | `0.2526` | `5.40e-9` |
| Lean closure channel removed | `4.20e-8` | Not supplied for this ablation | Not supplied |
| Lean closure present | `2.23e-8` | Not supplied for this ablation | Not supplied |
| Lean K4 use retained, K3 feed removed | `9.2e-8` | Not supplied | Not supplied |
| Lean diagonal regenerated too | `3.27e-7` | Not supplied | Not supplied |

The closure ablations motivate a matched use-and-feed experiment. They do not
establish a physical K4 law or the result on our networks. The author also records
99/100 local runs for V28/V29 with a residual-time failure; public success is not
our deployment certification. [Ablation table at the pin](https://github.com/504aldo/whest-p2-cumulant-k3/blob/18c17e2d7a9aeacd399cfc2c6b571e4e16dbfb45/docs/ablation_table.md).

Do not calculate a promotion from these different panels or infer a private
entrant's method from its score. The public repository's claim that cost alone
explains a historical leaderboard gap is not adopted here.

## Source audit: cheaper K3 is real, but the control needs a name

The shared-leg identity `M = P diag(s) + 3 A diag(e) + Z L^T` separates exact
reassociation from residual approximation. V16b additionally uses a rank32
residual, a 62-of-90 pruned term table, and suite-specific fourth-order and
fitted mean riders. It therefore is not an unmodified exact SIMPLE reference.
[V16b source](https://github.com/504aldo/whest-p2-cumulant-k3/blob/18c17e2d7a9aeacd399cfc2c6b571e4e16dbfb45/estimators/estimator_v16b.py).

V29 enables regeneration only when `n == 1024` and depth equals the 16-row
coefficient table. `NO_REGEN` leaves an older fourth-order rider on that shape.
It floors preactivation variances at `1e-10` and clamps adaptive coefficient
ratios to `[0.5,2.0]`. The source inspection did not find a covariance eigenspectrum
check or PSD projection. The fitted mean correction is off by default. These
facts mean a successful ordinary tiny smoke test cannot validate the production
closure. [V29 source](https://github.com/504aldo/whest-p2-cumulant-k3/blob/18c17e2d7a9aeacd399cfc2c6b571e4e16dbfb45/estimators/estimator_v29.py).

Use **“compact K3 with the pinned SIMPLE power/radial terms, no additional K4
matrix-core source, no fitted mean rider”** for the proposed baseline, followed
by its exact term manifest. “K3” does not imply that every fourth-order power
quantity in the harmonic construction is absent. An alternative ablation that
deletes all such quantities is a different estimator and needs its own name.

The reconstructed control must preserve all old K3 generations. Start from the
accepted mathematical term set, verify full-rank shared-leg parity, then add the
one declared old-source compression schedule. Do not inherit public-data pruning
or public-fit tables silently. Shared transport-space compression is an
approximation to persistent history, not deletion of old sources.

## Why exact signed laws and useful closures are different requirements

The ARC v2 efficiency theorem concerns fixed depth and polynomial activations.
Its experiments include ReLU; supplement S.6 discusses unresolved analytic
conditions for non-polynomial activations. It does not prove that a chosen
K3/K4 truncation is a probability distribution, preserves PSD at every finite
width, or has small error on this challenge's exact shape. Conversely it does
support evaluating controlled approximations to a fixed network's input
expectation. [Paper v2, introduction and S.6](https://arxiv.org/html/2605.05179v2).

There are three separate claims to test:

1. **Algebraic conformance:** the implementation evaluates its declared closure.
2. **Admissibility and stability:** the resulting moment state has necessary
   positivity properties and the trajectory stays finite.
3. **Accuracy:** the approximate expectation improves paired competition score.

Passing the first does not establish the other two. Our current packet already
contains an algebraically conforming tiny path with negative variance, so this
distinction is operational rather than hypothetical.

A distribution cannot generally be made exact by retaining nonzero K3/K4 and
setting every higher cumulant identically to zero. Under the polynomial-log
characteristic-function premise, Marcinkiewicz's theorem permits only Gaussian
or degenerate distributions. This does not prohibit approximations, nor does it
say a finite set of low moments cannot have a positive representing measure.
[Dinh et al., current v3, Theorem 2.1](https://arxiv.org/html/2107.08469v3#S2).

For a direct information argument, consider symmetric scalar distributions with
probabilities on absolute magnitudes `(0,1,2,3)` equal to
`q+ = (.15,.40,.19,.26)` and `q- = (.35,.10,.31,.24)`; split each nonzero mass
equally between its two signs. Both have first four raw moments
`(0,3.5,0,24.5)`, yet their expected ReLUs are respectively **0.78 and 0.72**.
This is an elementary constructed example, not challenge data or an error
measurement. The difference vector `(-.20,.30,-.12,.02)` annihilates powers
0, 2 and 4, while its absolute-first-moment contrast is `.12`. Thus four moments
do not determine a universal next-ReLU expectation. It does not show that these
particular laws occur in a random He-init MLP or preclude accurate wide-network
approximations.

For any real distribution, `E[p(X)^2] >= 0`; this gives moment-matrix PSD tests.
Covariance PSD is only one such necessary test. Positive diagonal entries do not
even establish covariance PSD: `[[1,2],[2,1]]` has a negative eigenvalue. A
fourth-order cumulant harmonic **core** is not itself a covariance and need not
be PSD. Projecting that core onto PSD has no automatic probabilistic meaning.
The tests belong to assembled raw moments/covariances of the approximate full
law, not to each mass-zero signed source in isolation.

Gram–Charlier positivity has its own constrained parameter region; semidefinite
methods can enforce nonnegative univariate polynomial densities. That literature
does not give a cheap width1024 joint closure or prove our proposed moment state
realizable. [Lin and Zhang, primary paper](https://www.sciencedirect.com/science/article/pii/S0377042721005586).

## Shortest credible improvement experiment

**Proposed, not preregistered or authorized by this document.** One hypothesis:
the shared all-history K3 representation can improve score against the standing
L2 candidate without relying on the presently invalid new physical K4 source.

1. **Truth-free control first.** Freeze the pinned SIMPLE term manifest and the
   exact baseline definition above. Reuse already-valid shared-state identities
   where their assumptions match. Independently verify tiny full-rank contractions,
   repeated-index handling, last-layer diagonal readout and all births. Freeze
   one compression schedule before any accuracy data. Exact parity and
   compression error are separate checks.
2. **Stable integrated trajectory.** Run the actual same branch on fixed tiny
   full-depth fixtures; record raw variances before any floor, covariance minimum
   eigenvalues, and standardized third/fourth corrections. Keep the existing
   known-failing fixture as a negative control. Reject a material negative
   variance, material negative covariance eigenvalue, overflow, or an unreported
   fallback. A tiny full-depth approximation failure is a failure of the chosen
   architecture gate, not a theorem against every wide-only approximation.
3. **Price before labels.** After independent acceptance, measure a complete
   official-package width1024/depth16 context, including setup, all transport,
   compression/QR, conversions and readout. The public `0.4986` V16b count is not
   this candidate's bill. Freeze a conservative counted-cost model; report
   worst-run residual headroom `0.4/observed`, and obtain actual Linux 8-GiB
   evidence before deployment. A source inventory is not process-cap evidence.
4. **One paired comparison.** If the stable counted candidate is plausible,
   preregister the exact score difference, block size, label precision and
   noise-floor reporting before new data. Compare against the standing L2
   control with common random numbers. A reasonable proposed continuation gate
   is an upper permitted-confidence-bound on paired arithmetic score difference
   below a frozen absolute margin corresponding to 10% of the preregistered
   control planning score. The actual number and `decide(gate=...)` convention
   must be written and committed with Samuel's prediction before the block.

Do not spend a fresh DECIDE block to debug a loader or source formula. Exposed
P01/P07 remain DESIGN. No coefficient may be fitted on the public panel. Heavy
jobs require the recorded optimization review and workerq resource declaration;
none were submitted for this review.

## One bounded frontier extension after that baseline

**Coherent approximate K4 with an explicit admissibility audit**, rather than
an unbounded exact signed-law construction. Retain the baseline's compact K3
history. Specify a closure core

`G_l = diag(g_l) + lambda_l * offdiag(C_l)`

and derive every use-side slice and K3 birth from that same state and the same
normalization. For the declared core,

`diag(W G_l W^T) = (W*W) g_l + lambda_l * [diag(W C_l W^T) - (W*W) diag(C_l)]`.

The diagonal transport identity is exact **for this approximate core**; neither
`g_l` nor the core is thereby an exact physical cumulant. The public generator
provides inspectable use-and-feed equations. [V17 generator](https://github.com/504aldo/whest-p2-cumulant-k3/blob/18c17e2d7a9aeacd399cfc2c6b571e4e16dbfb45/generators/make_v17.py).

Bound the first extension to one independently implemented tiny dense closure,
one compact implementation, and three declared arms: baseline; coherent use plus
feed; deliberately mismatched use-only negative control. Force the closure
branch on the tiny fixture explicitly. Compare to continuous low-dimensional
integration as an **approximation-error** reference, not as an equality target.
Require compact/dense algebra parity, finite depth16 trajectories and admissible
assembled covariance before any full-width cost run. If it fails, stop this
extension and keep the K3 result; do not respond by expanding exact lineage.

If a repair is necessary, a damping/PSD mechanism is a new closure intervention,
not housekeeping. It must act consistently on the updated mean, raw moments,
covariance and births. For example, blending **valid full-law raw moments** with
Gaussian raw moments preserves the PSD covariance of the mixture. Blending an
invalid Taylor state with a Gaussian does not have that guarantee; eigenvalue
or moment-matrix checks are still required. A variance floor alone cannot be the
repair. Full dense PSD checks and recomputation can add cubic work, which must
be included before claiming affordability.

Only after these truth-free gates should a small newly registered DESIGN study
choose coefficients, followed by a single fresh paired DECIDE test of the frozen
closure. The public improvement supplies a reason to try this approximation;
it supplies no guarantee that it will pass our stronger stability requirements
or beat our independent compact K3 control.

## Scope of the conclusion

The present evidence rejects treating an exact finite signed source as the only
legitimate route to improvement. It does not establish a new deployable
estimator, a guaranteed score gain, a universal PSD-preserving cumulant closure,
or knowledge of the leaders' methods. The bounded sequence is: make the cheap
K3 control concrete, test its score, then attempt one coherent approximate K4
extension with failure conditions fixed in advance.
