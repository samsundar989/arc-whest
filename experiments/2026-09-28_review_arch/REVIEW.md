# Review of the K111 worker results — 2026-09-28

Scope: actual-source review of `tools/k111_diag.py`, `tools/build_gate.py`, twin gate stages,
and v72. Two tiny CPU algebra checks were run. No estimator or existing harness was edited.
The new standalone diagnostic `tools/k111_consistent_probe.py` awaits/uses separately queued
heavy evaluation by the integrating agent. This is an architecture review, not a finals
Skeptic review or a replacement for PLAN.

## Findings

### 1. A real oracle-compensation defect needs a controlled measurement

`tools/twin/twin.py:232,241` inserts hooks replacing D3/D21 after `_advance` computed the
carried tensor. In `estimator_k3_v62_s1.py:3939–3941`, the replaced values enter `p111_3`
and `p111_21`. These are subtracted from the new repeated-slice birth at lines 3991–3992.
But line 3998 scales the **original** `adv` tensor. Thus, even with full P2 rank and no
shared compression, the next post-ReLU tower's repeated slices are

    actual_out_K3  = closure_K3  + w1^3 (original_D3 - injected_D3)
    actual_out_K21 = closure_K21 + w1_i^2 w1_j (original_D21 - injected_D21).

The injected inputs correctly influence the current closure and mean. The defect is in
compensating a carried tensor as though it had also been replaced. It affects K111 and
older exact-channel substitution experiments; it is **not a defect in the shipped gate
candidate**, which scales the actual carried factors and recomputes compensation from
those same factors.

Tiny actual-source check (`consistent_small/small.json`, n=24, float64, full-rank P2):

| quantity | legacy injection | compensation corrected |
|---|---:|---:|
| outgoing K3 mismatch norm | 0.00372584 | 5.8e-16 |
| outgoing K21 mismatch norm | 0.00153086 | 1.1e-15 |
| residual against the predicted legacy defect | <=1.1e-15 | n/a |

The current-layer mean, covariance, and every closure K output are bit-identical between
the two semantics. With no injection, the baseline is bit-identical. The new probe keeps
the injected values as closure/mean inputs and uses saved actual advanced slices only
for p111 subtraction. At real rank four, ordinary P2/shared-tier mismatch remains; this
repair does not purport to remove it.

Consequences: existing full/rank-limited oracle gains are not necessarily wrong in sign,
but are gains for a value-injection intervention with an extra state inconsistency. They
should not yet be interpreted as a slice-consistent candidate ceiling. Measure old and
new semantics side by side before choosing a representation from those percentages.

### 2. The projection identities and model tensor decomposition are sound

`rep3` and `rep21` at `tools/k111_diag.py:73–84` implement the index-partition identities
correctly. The model's own repeated slices are obtained by advancing with identity and
its next slices by advancing with the actual next weights. `adv_full` correctly avoids
adding another identity lineage when handling an already advanced tensor. Newborn
decomposition zeros the unwanted P2/hub fields, and the recorded additivity checks
support the implementation. The diagram label “all-distinct” refers to the source
neuron indices, not to the destination D21 indices.

Free-running projected discrepancy remains useful evidence independently of the oracle
defect. Exact-input mode, however, shares the compensation issue. It additionally never
supplies an exact incoming all-distinct tensor. Its similar discrepancy therefore cannot
prove that the error is newly born rather than inherited.

### 3. “Outside the gates' span” is a hypothesis stronger than the measurement

The tested source feature basis is `(1, alpha, alpha^2, gamma3, Phi)`
(`tools/k111_diag.py:63`). It has no gamma4, no arbitrary nonlinear mixtures, and the
`gate_basis` implementation at line 434 does not include the birth end-vector gate.
The fitted deployed gate uses a nonlinear MLP on alpha, gamma3, gamma4. Its function
class is not the same finite span.

Moreover, the `AD_orth` intervention at line 621 subtracts the error projection onto
**one model-AD direction**. It is not the projection orthogonal to the full tested gate
basis. A beneficial AD_orth intervention establishes value beyond uniform model scaling;
it does not prove that value is inaccessible to all state-dependent gates.

The actual trained gate's small effect is strong practical evidence against expecting a
large leap from this tested gate family. Do not enlarge it merely to rescue a negative
hypothesis. But distinguish that evidence from a mathematical exclusion of the class.

### 4. Low rank of the readout error is not a compact transported tensor

The rank screen restores the D3 error on the diagonal before SVD, as requested. Its
implementation at lines 884–935 is otherwise a conventional matrix spectrum screen.
The rep1 noise matrix measures the post-K21 component only; full1000 label uncertainty
is not in that particular matrix and should not be called the total noise floor.

The rank-r oracle at lines 954–963 always applies a **full exact AD correction to D3**,
then rank-truncates the D21 correction. Therefore the reported rank-32 benefit is not a
self-contained rank-32 tensor predictor. The report discloses this, but strategy must
not lose the qualification.

An arbitrary low-rank D21 matrix need not arise from a symmetric low-rank third tensor
with the matching D3 diagonal, a causal birth law, and cheap transport under the next
weight matrix. The SVD is evidence that a destination readout error is compressible;
it does not establish that an online low-rank state can generate that error. Also,
diagnostic medians of relative changes are not the competition's pooled mean score.

### 5. The gate implementation and controls are substantially credible

`build_gate.py` replaces the transport Wick vector and uses it consistently in
`p111_3/p111_21`. Birth-end changes also update the separable leg and hub compensation.
The actual-source full-rank checks demonstrate preservation of the closure slices;
rank-four leakage is explicitly measured in the design record. The twin trains the
actual rank-four function, so leakage is an attribution limitation, not an unnoticed
train/serve mismatch.

G1tg trains only gate leaves and freezes the existing parameters. Its initial validation
MSE matches the v71 parameter set. S3c starts from the same set but trains the existing
leaves, initially overshoots markedly, and never beats initialization at the stated
learning rate/cadence. Thus “specific to gate updates versus that control” is supported;
“no further ordinary fitting can help under any suitable schedule” is not established.

The latest ledger now records v72 #332766 graded at **4.620737758949154e-9**, with
final raw MSE **2.1518525734620654e-8** (final-layer label floor approximately 7.7e-11).
This is a real small improvement, not a score leap. RESULTS/CURRENT_STATE/PLAN snapshots
lag the ledger at the time of this review.

## Distinct candidate representation: a selected covariance-triangle birth

The literature agent proposed the omitted connected Gaussian triangle

    delta T_ijk = f2_i f2_j f2_k C_ij C_jk C_ki,

on distinct source indices. This adds a different tensor pattern from rescaling existing
two-leg paths. A full-cost oracle and a projection onto the observed residual should
precede any candidate build; the sign and score value are not known.

If a useful covariance component has `C ~= U U^T` of small rank q, define symmetric
matrices `H_i = f2_i u_i u_i^T`. Then

    delta T_ijk = trace(H_i H_j H_k)
    D3_i = trace(H_i^3)
    D21_ij = trace(H_i^2 H_j).

Pack the symmetric matrices with sqrt(2) weights on off-diagonals, so their Frobenius
inner products become ordinary dot products. With `R_i = pack(H_i)`,
`Q_i = pack(H_i^2)`, and `t_i = trace(H_i^3)`, the full repeated slice is `B = Q R^T`.
The exact repeated-slice canceler at birth is

    A_D = -3 B^T + 2 diag(t),

which has the existing P2 form with `Z = -3 R`, `L = Q`, and `sb += 2t`. This cancels
all repeated source indices while leaving the all-distinct triangle intact. It must be
added with this bookkeeping, not independently subtracted a second time in p111.

**Transport reuse:** the extra Z columns already carry packed H, since both evolve by
`W^T` and the same per-neuron Wick row scale. There is no algebraic need to transport a
second copy of H. A q=4 source adds 10 columns; q=8 adds 36. Repeated compensation
and triangle readout still cost arithmetic and memory; real flopscope measurement is
required. A rough P2-rank analogy is only a screen, not a metered price.

Each source's H must remain separate: `trace((sum_sources H)^3)` would create unwanted
cross-source terms. Full PSD covariance is the natural low-rank object; zero-diagonal
covariance is generally indefinite and must not be treated as PSD. On all-distinct
indices, the diagonal entries of C do not affect the triangle; their repeated effects
are removed by the canceler.

The independent tiny CPU test `triangle_carrier_algebra.py` verifies covariance-product
versus trace spelling, packed readouts, repeated-slice cancellation, reuse of Z under
transport, and full-tensor transport/Wick to approximately 1.6e-13 absolute at n=7,q=3.
This establishes the representation identities only. It provides no evidence yet that
the selected covariance rank preserves the important triangle or that the triangle
improves final MSE enough to pay for its cost.

## Immediate measured sequence

1. Compare legacy and slice-consistent AD/FULL/rank32 oracles on two common FIT nets,
   then expand only if the difference is material. Use the same old v68-equivalent
   parameter set first, followed by v72-equivalent parameters if needed.
2. Complete the integrator's existing-state covariance-basis predictability diagnostic.
   Matrix compressibility without a predictable basis is insufficient.
3. If the out-of-scaling residual remains score-relevant, project the selected triangle
   onto it and run a source-consistent oracle for its effect. Only then price q=4/8
   or another justified basis at the real counted meter.

The standalone consistent probe has an explicit pre-submission resource review in its
header: one net at a time, no gradients or saved layer histories, already-valid labels,
per-net atomic JSON checkpoint, CPU cap. Proposed first job: 16 GiB RAM, 10 GiB VRAM,
2 CPUs. Prior related diagnostics peaked at approximately 5.0–6.3 GiB VRAM. Repository
portability verification passed **102 checks, zero failures** after the new diagnostic.

## Addendum: feature screen, measured cost limit, and next bounded probes

`tools/k111_feature_probe.py` is read-only and uses the actual carried tensor for the
repeated-slice subtraction. Its last-layer reconstruction correctly restores pairwise
objects that production intentionally omits. Its leave-one-network-out Gram regression
excludes the target network from coefficient fitting; the baseline parameters had already
seen these FIT networks, which the report discloses. Its basis projections use oracle
coefficients, and Eorth removes only one uniform model-scaling direction. The initial
quadratic-left rank128 screen uses an ordered subset of top16-mode products, not the best
128-dimensional quadratic subspace.

The next screen, `tools/k111_core_probe.py`, removes that arbitrary product truncation:
q16 uses all136 pair products and q32 all528. It reports the unconstrained product-space
projection upper bound, then fits an actually symmetric third-order core with matrix-free
conjugate gradients. D3 gets independently specified weights1 and1024; off-diagonal and
D3 residual reductions are reported separately. Finite-iteration CG gives a feasible
captured-energy lower bound until converged, not an impossibility certificate. Its CPU
test against explicit symmetric-orbit least squares passes within3.7e-13; the dense tensor
slice agrees within9e-16, and an exactly representable signal is recovered within4.4e-13.
Both basis and coefficients must be distinguished: covariance modes are computable from
known state, while these fitted core coefficients are still oracle diagnostic values.

### What cost improvement is actually sufficient?

The current near-identical cost architecture has472,202,879,721 FLOPs and22,992 calls.
The real-meter v68 drop-old-tier control has339,757,269,313 FLOPs and17,673 calls versus
472,199,315,449 and22,701 for its base. Thus the removed old-tier machinery costs
132,442,046,136 FLOPs, **28.048%** of v72's cost. This ablation costs considerable accuracy;
it is useful here only to bound possible savings. It leaves71.952% of the bill even if a
replacement could be free and lossless. A precise current split of that remaining bill
requires a new real-meter source-site census; the older54% recent/25% old decomposition
should not be presented as a current exact breakdown.

Using the integrator's common50-network leader ratios0.50424 cost and0.75376 raw MSE,
the required score ratio is0.38007594. Consequences:

* At unchanged cost, raw MSE must fall about62.0%.
* At a free, lossless old tier, raw MSE must still fall47.2%.
* At the0.1 multiplier floor, raw MSE must still fall18.4%.
* At leader-like raw MSE, cost must fall49.576%, to approximately238.1G FLOPs.

These are conditional arithmetic using the stated paired-panel ratios, not a forecast
for an untested candidate. Old-tier compression alone is not a winning architecture.

### Triangle price and the folded-source caveat

Five ordinary skinny products per active source-layer give a leading arithmetic estimate
of10*n^2*d, d=q(q+1)/2; a conservative twice-arithmetic billing envelope is20*n^2*d before
packing, eigenspace construction, H-squares, and other elementwise overhead. This is not
a measured flopscope total. For late births10..14 (zero-based), there are15 transport stages:
q4 has10 columns and leading envelope3.15G FLOPs,0.67% of v72; q8 has36 columns and11.32G,
2.40%. A45-stage set gives2.00%/7.19%; all120 stages give5.33%/19.19%. Final-layer pruning
can reduce these figures. A separate batched carrier is a handful of matrix operations
per layer, not five calls per neuron/source, but packing/Strassen glue/shape handling must
be measured before predicting operation-count headroom.

The reused Z/H transport is exact, but the canceler's source M becomes approximate once
that source enters the shared tier. There is no generic exact cheap replacement for the
arbitrary dense M in Sym(Z,L,M,M). Keep an independent dense M (and pay for it), project
the added carrier consistently and measure its error, or first restrict to late births
that never fold before their last scored consumer. The15-stage late-birth candidate is
the clean initial implementation scope if the triangle probe is positive.

### A different compression diagnostic, without reviving the closed rank sweep

The earlier N1 experiment ranks individual eigen-directions of the leg basis by deletion
sensitivity. High correlation of those individual sensitivities with leg energy argues
strongly against merely reordering that basis. It does not measure a jointly fitted
symmetric core or all off-diagonal correlations in the final-output Jacobian. Neither
fact is evidence that such a core will work; a cheap representability rejection test
must precede implementation.

The executable next test design is:

1. At selected late boundaries, isolate the **model old tier**, including its P2 and hub
   contributions. The target here is the old tier to be replaced, not the label-model AD
   error tested by the current core probe. Freeze the reference Wick gates for the first
   readout screen, preserving source-consistent compensation on actual rollout.
2. Obtain q16/q32 candidate coordinates from model covariance (known-state basis), and
   compare three cores: direct multilinear projection of the factored old tensor;
   least-squares fit to that tensor's next D3/D21 readouts; and an offline upper-bound
   fit through several remaining readouts. Fit all D3/D21 together with a symmetric
   core. A D21-only SVD or independent perfect D3 correction is not an admissible control.
3. For the adjoint-weighted variant, differentiate the actual remaining twin recurrence
   with respect to the symmetric core, with checkpointing. Optimize final-mean mismatch
   to the uncompressed teacher plus readout penalties, not organizer labels. Bound the
   number of gradients and compare the same optimization budget with ordinary Frobenius
   readout fitting. Test the resulting core in a full free-running continuation.
4. Require the direct projection/causal update to work, or learn a shared correction on
   TRAIN and validate on other networks. A separate per-network fitted core, a label
   SVD basis, or the expensive teacher tail must never appear in a shipping predictor.
   The causal shipping update projects each joining lineage from its existing factors
   into the current small basis and adds it to the transported old core. Its leading
   work is O(n*q^3+n^2*q), which is far smaller atq32 than the previously pricedq128
   Tucker join. It may nevertheless lose crucial information: that is what the test
   decides.

Controls are unchanged-v72, zero old tier, direct projected core, readout-fitted core,
adjoint-fitted core, and enough-rank tiny full-tensor identity. Quantify old-tier slice
errors and final-mean drift against the teacher before asking whether the candidate
improves labeled MSE. If even the oracle core cannot preserve useful accuracy atq32,
close the lane. If only the oracle core works, the next task is predicting its update,
not claiming a deployable architecture. The maximum28% old-tier saving also means this
lane needs a successful earlier-fold/recent-tier application or an independent accuracy
gain to close the leader gap.

## Completed bounded runs (September28, jobs2293 and2302)

The slice-consistency control has completed on FIT0/1. Baseline parity is bit-identical.
Using paired panel-mean MSE changes (different from median per-network ratios):

| intervention | legacy compensation | consistent compensation |
|---|---:|---:|
| all-distinct oracle | -37.593% | -34.823% |
| rank32 D21 plus exact D3 oracle | -23.004% | -22.524% |
| full D3/D21 oracle | -47.707% | -47.707% |

The defect is real and changes the AD result, but the useful AD signal survives this
two-network control. FULL being identical is expected: exact D3/D21 are independently
reset at every consumer, so the changed hidden carried tensor does not reach these
consumers' closure inputs. This is not proof that full-tensor correction is affordable.
These are two previously used FIT networks, not independent competition performance;
the final-layer label-noise floor remains approximately7.7e-11.

The symmetric-core screen uses v71-equivalent parameters and the component orthogonal
to one model-AD direction. Both FIT networks agree closely:

| consumer layer (zero-based) | q32 product-space upper capture | symmetric q32 capture, weight1 | D3 capture, weight1 | D3 capture, weight1024 | off-diagonal capture, weight1024 |
|---|---:|---:|---:|---:|---:|
|9|36.9–37.2%|20.5–21.0%|13.4–18.5%|99.0–99.1%|1.3–2.7%|
|12|43.5–44.2%|27.4–28.0%|25.7–30.2%|99.1–99.2%|9.1–11.0%|
|15|47.9–48.6%|32.3–33.8%|30.4–38.4%|99.1–99.3%|15.7–19.5%|

q16 captures only7.7–15.7% at weight1. Making q16 prioritize D3 instead harms
off-diagonal error18–31%. Thus the quadratic-left/linear-right asymmetry does identify
some compatible symmetric-tensor content, but it does not explain most of the residual.
The reported rank32 independent-D3 oracle is more permissive than this physical core.

Weight1 CG meets the1e-7 normal-residual criterion; q32 weight1024 reaches100 iterations
with residual2–4e-7, close but still a finite feasible fit. There is no claim of exact
global optimality for that fit. Runtime was3.6/2.9s per net, peak4.44GiB torch VRAM.
The next bounded intervention, if pursued, is to inject these jointly fitted symmetric
slices with consistent compensation and measure final MSE. Even that would remain an
oracle representability result until a cheap causal core update predicts its coefficients.

The downstream core oracle's first run, job2303, stopped on its exact prepared-consumer
guard at the first intervention: D3 differed by7.4505805969e-09, one float32 unit at this
scale. Preparation had copied `W.T` to contiguous storage; production uses the transpose
view. The copied layout permits a different CUDA GEMM reduction. The probe now preserves
the production view and keeps the **bit-exact** guard; a retry is required to verify this
fix. No completed network result was produced by2303.

Pre-retry optimization/resource review: reuse the same public labels, no loader workers,
no prefetch or gradients, one arm/net and per-net checkpoints. Job2303 measured roughly
2.4GiB host and6.02GiB GPU process residency before the guard; completed symmetric-core
job2302 measured4.44GiB peak torch allocations. Those GPU quantities differ because
process residency includes allocator/library reserves. q32's largest new factor is
n*q^2 float64 (~8MiB), residual/covariance matrices are8MiB apiece, and a core is0.25MiB.
The existing tower and linear algebra workspaces dominate. RAM16/VRAM10/CPU2 is a
conservative comparable-workload estimate. The queue reported22.5GiBhost/20.2GiBGPU from
two supposedly matching historical runs despite this being a newly created script;
that signature-history warning does not identify matching source/work. It must not be
called a measured peak of this probe. Preserve the current calculation and monitor the
actual retry rather than discarding evidence or silently lowering resources.

### Downstream result: close the tested covariance-core correction

Job2305 completed FIT0–3 after the layout fix. All prepared D3/D21/var values match the
actual consumers bit-for-bit, and legacy/consistent baselines match at every layer.
Peak torch allocation is4.405GiB. The fitted corrections were recomputed on each arm's
current state at consumers1..15; all use v71-equivalent parameters and consistent
compensation. Paired panel-mean results:

| arm | final MSE | change versus baseline | networks harmed |
|---|---:|---:|---:|
| unchanged baseline |2.0201338834e-8|—|—|
| unprojected ADorth oracle |1.5507308244e-8|-23.236%|0/4|
| symmetric q32 core, D3 weight1 |2.1152609875e-8|+4.709%|4/4|
| symmetric q32 core, D3 weight1024 |2.1374238117e-8|+5.806%|3/4|

Final-layer reference noise floor approximately7.7e-11. These FIT networks and oracle
moment labels are not independent validation. This experiment decisively rejects the
direct covariance-core **Frobenius-fit AD correction as tested**. Capturing moment-error
energy did not translate to useful final-mean error. Do not treat the energy screen as
a successful architecture result or proceed to a larger predictor to rescue it.

A narrowly bounded way to distinguish wrong amplitude/sign from a useless direction
would freeze each baseline-derived joint-core correction, roll out fixed gains+h/-h
(for exampleh=.1), and estimate its final-output response Jg by central difference.
Fit one shared scalar on other networks, g=-sum(e dot Jg)/sum(||Jg||^2), where e is the
baseline final-mean error. Test that held-out-network scalar in the nonlinear rollout.
This uses a single shared coefficient, not per-network free cores or16 layer gains.
It would remain an oracle-direction diagnostic, not a shipping predictor; a negative
result should close this metric question without expanding the function class.

The separate cost-first history-distillation hypothesis must change the representation,
not revisit existing shared-leg rank or age settings. It would retain only2–3 recent
lineages and replace all old content by one transported U(n*q),G(q^3) symmetric core.
At each fold, project the available joining lineage's A_L/M/Z/hub factors into U and
form its core analytically, then add the transported old core. Require an actual whole
chain cost near0.12B or lower before substantial training; age5 alone cannot save enough.
q32 leading work O(n^2q+nq^3) differs from the previously rejected expensiveq128 joins,
but its accuracy is unproved.

A possible **shipping** update learner may apply small shared gates to those analytic
joining-factor contractions using available source-column features: norms ofA/M, A dotM,
w2, sb/eb, age, and compressed-state summaries. It may not use per-network optimized
cores, label SVD bases, or an expensive teacher suffix. Train through the actual compact
student to a final-output loss

    L_teacher = mean ||mu_student,16 - mu_v72,16||^2

with a small intermediate-mean stability term if needed, then fine-tune to TRAIN means
and select only on VAL. The final-output gradient through the real recurrence is the
goal-oriented weighting; it avoids assuming that Frobenius moment error or a mean-path
proxy predicts the score. First test one late fold with direct projection and a bounded
offline adjoint-core upper control in the same known basis. If even that upper control
fails, stop. If only freely fitted cores succeed, the missing result is a transferable
causal update, not a deployable solution. This remains a research proposal, with strong
negative prior evidence from earlier compression experiments.

## Pre-submission review: coherent-core downstream oracle

New probe `tools/k111_core_oracle.py`, FIT0–3, v71-equivalent parameters. Hypothesis:
q32 symmetric-core corrections in observable covariance modes retain useful downstream
value after dropping the independent perfect-D3 shortcut. Prediction: a smaller benefit
than the unrestricted ADorth oracle; perhaps5–15% raw, with uncertain diagonal weighting.
Decision: proceed to a learned-core design only if one coherent arm gives at least10%
paired raw benefit without severe tails; these are four diagnostic TRAIN nets, so expand
before relying on a percentage. An oracle coefficient is not a deployable predictor.

Optimization: reuse cached labels/weights and the tested matrix-free core solver; one
network/arm at a time, no gradients or saved tower history, no n-cubed tensor, CPU2.
Recompute on each arm's actual state to preserve intervention integrity; do not save time
by borrowing baseline-state corrections. Joint slices are prepared before the consumer;
assert bit equality against actual D3/D21/variance and baseline every-layer means. Save
atomic per-network results. Comparable core job2302 peak8.53GiBhost/6.05GiBVRAM; declare
16GiBhost/10GiBVRAM/2CPU with margin. Expected1–3minutes for4nets. No changes to source
or parameters during a run; queue snapshots preserve executable versions.
