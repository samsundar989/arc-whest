# E14 independent mathematical review: a concrete mixed residual query

September 14, 2026. Static review only: no candidate or instrument changes,
new fixtures, trajectories, rank measurements, labels, or protected reads.
The numerical and cost results below have different evidentiary status.

## Decision

The worker correctly stopped the literal decoder wrapper and the specified
ordinary-f64 residual query. Neither refusal closes full-residual compression.
There is a concrete next implementation: retain the complete residual tower in
F21's packed float32 form, transport it with F21's existing batched Strassen,
and use a float64 common basis only for the two other legs. Query its residual
return with explicitly float32 operands. Preserve the existing nonlinear
evaluation order. The complete conservative static model below gives about
615.26 billion FLOPs at a *hypothetical* common rank 64, below the exploratory
742.77 billion ceiling. This is a feasible operation budget, not evidence that
rank 64 is accurate, that the numerical kernel passes, or that it meets runtime
limits. Rank is not being selected or measured by this review.

The immediate intervention is the full-rank numerical kernel discriminator in
section 7. Its full-rank mode need not save FLOPs: that mode establishes what
the rank approximation means. Requiring an uncompressed reference to meet the
compressed implementation's saving screen is an unnecessary extra condition.
Do not relax the old failing numerical gate or proceed to rank tests before
the new kernel's declared numerical conformance passes.

## 1. What the existing evidence establishes

Read sources: frozen `estimator_k3_strassen_L4_f21.py`,
`local_sim/p2_compact_reference.py`, the E14 compression cost note and strategist
packet, plus existing exposed raw JSON under `experiments/2026-09-14_e14`.

- `b4_f64_v2.json` supports the already reviewed real-arithmetic operator
  correspondence, to its recorded tolerance. It does not certify a deployment
  precision policy.
- `b4_real_backend/b4_f32_summary.json` records B4 L2 squared output drift
  `7.2608825576e-5`, B4 forced-F21 `2.3064086735`, B3 L2
  `6.2740923568e-12`, B3 forced-F21 `4.2012704426e-10`, against `2e-10`.
  These are actual failed/passed outputs, not accuracy against truth. The
  summary also explicitly says the tiny forced branches do not test live L4.
- `b4_mixed_qr/b4_mixed_qr_summary.json` records zero final-output drift in
  four dense-wrapper cases, despite source roundtrip mismatches. This supports
  retaining the frozen nonlinear program and treating basis rounding separately.
  It does not establish a cheap direct query.
- The original `196*M64(n,n,n)` refusal is correct for its named ordinary-f64
  operations. It is not a matrix-multiplication lower bound.

No MSE, noise-floor estimate, score improvement, or numerical result is newly
measured here. The historical F21 raw MSE `4.1821088467e-8` has its P10 midpoint
truth floor `2.2363657945e-9`; neither number evaluates the proposal.

## 2. The legal residual arithmetic was not exhausted

Let `S_t(a,b,c)` be the counted float32 cost of the **existing** `_sb` spelling
for `(a,b)@(b,c)`, at depth t. When the existing shape/minimum-block rules stop
recursion, `S_0=a*c*(2*b-1)`. Otherwise, counting the actual additions and copies,

    S_t(a,b,c) = 7 S_(t-1)(a/2,b/2,c/2)
                 + 3(ab+bc) + 4ac.

The ten input additions contribute `5(ab+bc)/4`; the two packed input
concatenations contribute `7(ab+bc)/4`; eight output additions contribute
`2ac`; inner plus outer output concatenations contribute `2ac`. This is not
the informal `(7/8)^t` discount. The float64 bill for the same spelling is
`2*S_t`. Use the source's shape fallback, rather than forcing a level at odd
dimensions or tiny sizes.

At n=1024, `S_4(n,n,n)=1,366,126,592`. Thus even 196 separate float64 square
products cost `535,521,624,064`, instead of `841,402,548,224`. Packing old ages
as F21 does lowers the corresponding old residual transport/query subtotal
to `523,349,770,240`. This does not price the whole method, but it disproves
using the ordinary-product subtotal as a family obstruction. These operations
remain instrumented `fnp` operations and use an already accepted own-code
primitive; no raw-array/unmetered work or new algorithm attribution is implied.

Float64 arithmetic everywhere is still unattractive: the complete dominant
product/QR subtotal of a packed f64-shared/f32-youngest construction is
`733,615,402,344` at r=32, leaving little room under the 20% screen. This
motivates an explicit mixed residual boundary, not a claim that f64 solves
the numerical problem for free.

## 3. Retained state and a complete mixed-precision operator

Write the source tensor as the symmetric sum of columns of `(D,P,P)` and
`(A,3P,A diag(w))`, with exactly F21's symmetrization/repeated-index convention.
Here `A=A_L`, `P=M`, `D=A_D`, `w=w2b`. The generalized bridge has `s=e=0`,
`Z=D`, `L=I`; no residual factorization is performed.

At the end of a nonfinal layer retain:

1. The full packed residual `D=[D_1,...,D_k]` in f32, all ages and all n columns
   per age. This is full residual rank, regardless of r.
2. For the older m=k-1 sources, f64 `Q` of shape n by r and packed `F_A,F_P`
   of shape r by mn, with `A=Q F_A`, `P=Q F_P`. Retain their birth w vectors,
   ages, and weighted Gram core G. These matrices are not sourcewise residuals.
3. The newest exact f32 A, w, and symbolic P=I. Its D is the final block of the
   packed residual. Retain this youngest source outside the basis for one step.
4. The unchanged f32 mean, covariance, scalar K4, table and frozen controls.

Use `W=WT` in the source's naming. Advance all residual ages in a **single**
original f32 call `Dadv=_mm(W,D)`. Advance old Q by ordinary f64 `W64@Q`.
Advance the youngest A by f32 `_mm(W,Ayoung)` and take `Pyoung=W` symbolically.
The old stored Q and D already contain the preceding Wick scale. Do not
multiply that Wick a second time; do not apply it to the preceding newborn.

For older sources reconstruct only the packed A and P needed for the current
queries, in f64. Set `R=cast64(Dadv_old)` and compute

    C = A*w
    PP = P*P
    D3old = rowsum(R*P*P) + 3*rowsum(A*P*C)
    LA = 2*(A*P)*w
    LP = A*C + (2/3)*(R*P)
    X = LA@F_A.T + LP@F_P.T
    Y32 = _mm(cast32(PP), Dadv_old.T)
    D21old = X@Qadv.T + cast64(Y32)/3.

All products in X and its return use ordinary f64 matmuls. Both operands of
Y32 are f32. Storing D in f32 without casting PP would *not* earn this tariff.
There is no dense residual decoder multiplication by I. Streaming or packed
temporaries are implementation choices only if their changes in accumulation
order are declared; the cost model here uses the packed version.

Evaluate the separate youngest D3/D21 with the frozen f32 `_dslice3` and
`_dslice21` on its advanced three legs. Mask the older D21 with the same
off-mask; the youngest D21 is already masked by its frozen consumer. Combine
old and youngest slices in f64, then cast consumed slices to f32. The new program
must specify this combine/cast order literally. It is an explicit arithmetic
policy, **not** asserted to be bit-equivalent to the original wide contraction.
The displayed formulas prove equality only over real arithmetic.

Run the unchanged F21 nonlinear program, including its normal/Wick evaluation,
term table order, fast one-dimensional path, centering, variance floors, alpha
saturation, scalar K4 evaluation, and `p111_3/p111_21` subtraction. Read the
same consumed D3/D21 in both nonlinear consumption and source subtraction.
Birth the exact new f32 `A=w1*C11`, `D=diag(S3)+3*S21.T`, `P=I`.

After consumption, scale all advanced residuals by w1 in f32, and Qadv by w1
in f64. Scale the advanced youngest A and P by w1 in f32. Admit this *aged*
youngest pair to the old basis as below. Append the newly born D only after
the old Wick scale, exactly as F21. The final layer performs transport and D3
readout, with no D21, join, new source, or Wick state publication.

For a real-arithmetic proof, substitute `A=QF_A`, `P=QF_P` into the source's
three `_dslice21` terms; the two returns ending in `A.T` and `P.T` share a
rightmost `Q.T`, while `(P*P)@D.T/3` does not. D3 is its repeated diagonal.
This reproduces every source age and does not drop the full residual term.

## 4. One specified join, with rotations and Gram maintenance

Use a deterministic, declared n-by-r sketch Omega, e.g. the first r columns
of I for this *proposed* program. Its suitability for approximation is unknown.
The full-rank test uses Omega=I. No random dataset or seed is needed to define
it. A future random-sketch alternative would be a separately declared choice.

Let Q denote the just-transported, just-Wick-scaled old basis. It need not be
orthonormal. Let B and E be the advanced youngest A and P cast to f64, and
`a=9+w*w`, `p=1`. Maintain

    G = sum_j [F_Aj diag(a_j) F_Aj.T + F_Pj F_Pj.T].

The weighted n-space Gram of the represented legs is `QGQ.T`; full residuals
do not enter or get projected. Form

    H = Q[G(Q.T Omega)] + B[a*(B.T Omega)] + E(E.T Omega),
    Qnew = reduced_qr(H).Q,
    T = Qnew.T Q,
    F_Aold_new = T F_Aold,      F_Pold_new = T F_Pold,
    F_Anew = Qnew.T B,          F_Pnew = Qnew.T E,
    Gnew = T G T.T + F_Anew diag(a) F_Anew.T + F_Pnew F_Pnew.T.

Omit terms involving empty old state on the first admission. Form the Gram
using the declared evaluation order. Scale columns with broadcasting; do not
materialize diagonal matrices. Append both factor blocks and the birth weights.
All prior ages are rotated, none deleted. Residual D is unchanged by this join.

At r=n, an orthogonal Qnew has `Qnew Qnew.T=I`, so the join is a change of
coordinates, including when the Gram is singular and QR completes an
orthonormal basis. At r<n, the intervention is exactly projection of the two
legs onto `Qnew Qnew.T`, retaining the full residual. This establishes what
is being approximated; it makes no spectral/rank adequacy claim.

One common Q is a useful way to share WQ and the rightmost query return. It
is not a mathematical requirement to put D, the covariance, the newborn I,
or future K4 components into that basis. Separate age-cohort bases would be
valid alternatives, with extra WQ/return/join bills; they are not needed for
this bounded intervention. Requiring one basis for *every* leg would incorrectly
turn full residual preservation into an impossible small-r demand.

## 5. Complete parameterized static cost

Use `M(a,b,c)=2*a*c*(2*b-1)` for ordinary f64 matmul and the legal f32 S above.
Let `QR(n,r)=4*(2*n*r*r-floor(2*r^3/3))`. All S calls apply F21's depth/minimum
block/odd-size rule. Here k=l incoming ages, m=max(l-1,0), l=0,...,d-1.

The following charges include *all* dominant operations of the proposal:

| Block | Counted product/QR cost |
|---|---|
| All residual transport, k>0 | `S(n,n,kn)` |
| Exact youngest A transport, k>0 | `S(n,n,n)` |
| Old basis transport, m>0 | `M(n,n,r)` |
| Packed old leg reconstruction, m>0 | `2 M(n,r,mn)` |
| Old nonfinal D21, m>0 | `2 M(n,mn,r) + M(n,r,n) + S(n,mn,n)` |
| Youngest nonfinal D21, k>0 | `3 S(n,n,n)` |
| Admission at l=1,...,d-2 | `J_m` below |

With an empty-old-state indicator `I_m`, the join's complete product/QR bill is

    J_m = 4M(n,n,r) + QR(n,r) + 2M(r,n,n) + 2M(r,n,r)
          + I_m [2M(r,n,r) + M(r,r,r) + M(n,r,r)
                 + 2M(r,r,mn) + 2M(r,r,r)].

This counts the two new Gram-sketch terms, old carried Gram sketch, QR,
new-source projections, old-basis overlap, both old factor rotations, Gram
rotation, and both newly admitted factor Grams. No basis construction,
projection, residual transport or dense return is free.

For the unchanged backbone, a conservative exact-source bound is

    B(n,d) = F_F21(n,d) - sum_(l=1)^(d-1)
      [2S(n,n,ln) + I_(l>1)S(n,n,(l-1)n)
       + I_(l<d-1)3S(n,ln,n)].

`F_F21(n,d)` must be the correctly matched existing ledger (or static full
source census). B retains the original covariance matmuls, mean matvecs, whole
nonlinear/scalar program, readout, and even original tower elementwise/copy
work. It therefore overcounts retained work rather than forgetting the
backbone. At the live shape, the subtracted tower product tariff is
`880,957,820,928`, giving `B=47,499,816,550` from the measured F21 total.
As a cross-check the source's full product tariff, including covariance, is
`924,673,871,872`, leaving `3,783,765,606` of the measured count for everything
else. This arithmetic does not turn the proposed program into a measured one.

Casts, copies and pointwise work must be additional. A deliberately loose
complete envelope for the specified straight-line blocks is

    E(n,d,r) = 512 [K n^2 + d n r + K n r + d r^2] + 128 d n,
    K = d(d-1)/2.

This allows up to 512 charged scalar operations per stored/working matrix
element across these blocks; it includes dtype weights. To make the envelope
reviewable, implementation must respect these individual upper allocations:

- Old query arithmetic, reductions, cast D64, cast PP32, and result casts:
  `64mn^2+16n^2+8nr+8n` per old-query layer.
- Youngest query arithmetic and packed state publication (including immutable
  concatenations): `32kn^2+16kn+16n^2` per nonempty layer.
- W64 cast, residual/Q Wick carry, extra result combinations:
  `8kn^2+8nr+8n^2` per nonempty layer.
- Join's column scaling, Gram additions, source casts, factor/weight copies,
  and Q/sketch handling: `32n^2+32(m+1)nr+32nr+32r^2+32n` per admission.
- Sketch setup and output housekeeping fit the unused envelope; no explicit
  n-by-n diagonal/identity matmul is allowed. Identity/sketch buffers still
  count toward storage. QR arithmetic is separately charged in J.

These are conservative **static allocations**, not observed counts or license
to run extra unpriced passes. A different implementation that exceeds them
requires re-costing. In particular, repeated old-factor Gram rebuilding or
decoding a packed residual through a dense identity is not this program.

The total is B plus all table rows plus E. Pure integer arithmetic at 1024x16
gives the following sensitivity illustration, **not a rank experiment**:

| Hypothetical r | B + products + QR | Extra envelope E | Total upper model |
|---:|---:|---:|---:|
| 32 | 475,232,271,310 | 66,716,696,576 | 541,948,967,886 |
| 64 | 546,234,997,046 | 69,023,563,776 | 615,258,560,822 |
| 96 | 619,607,647,846 | 71,347,208,192 | 690,954,856,038 |
| 128 | 695,373,554,638 | 73,687,629,824 | 769,061,184,462 |

Thus complete static cost does not block the proposal at r64 or r96. Nothing
here demonstrates that those ranks retain the signal or that r128 actually
fails: the last column is an upper envelope, not a lower bound. Include the
known server offset near any eventual score boundary; these margins dwarf it.

## 6. Memory, operations and runtime remain separate

At k incoming ages and m=k-1, retained array payload is approximately

    4kn^2 + 4n^2 + 8nr + 16mnr + 8r^2 + O(kn)

bytes, excluding backbone and weights. It includes full residual D, newest A,
Q, two factor towers and G. No L=I buffer is necessary. Queries can co-hold
two reconstructed f64 packed legs (16mn^2), a f64 cast residual (8mn^2),
pointwise temporaries, a packed f32 PP (4mn^2), query coordinates (O(nr)),
and the n-by-n return. Join holds old/new factors, Q/Qnew, overlap T, G/Gnew,
H, QR R and workspace; immutable residual publication can co-hold old/new D.

The Strassen workspace is material: for an `(a,b)@(b,c)` call its depth-t
input stacks contain `(7/4)^t(ab+bc)` elements and output products contain
`(7/4)^t ac`, in addition to retained parent buffers. Account for their
simultaneous recursive lifetime, not just the leaf matmul. At maximum residual
width these may dominate state payload. Include f64 incoming weights, f32
working copies, nonlinear scratch, BLAS/QR workspace and flopscope metadata.
No 8GiB pass or process-memory peak is asserted by these array formulas.

Pack source ages into the same calls: sourcewise recursive `_sb` loops can
destroy operation-count/runtime headroom even at the same arithmetic cost.
The frozen primitive's batched recursion is essential to this proposal. Every
new query/join/QR call and allocation adds residual overhead. Historical
F21 server residual headroom 2.5603x cannot certify this new call graph.
Queue any eventual heavy verification with a recorded resource review.

## 7. Cheapest falsifier and what numerical equivalence should mean

One proposed IMPORTANT kernel implementation, separate from all frozen files:
copy the frozen nonlinear schedule verbatim and replace only source state,
transport, admission and D3/D21 consumers as specified above. No new K4,
normal evaluator, variance rule, generation chooser or precision relaxation.

Run only the already exposed fixed B3/B4 fixtures after the required written
prediction and independent review. Three controls distinguish failure causes:

1. No-basis frozen source mode: reproduce original F21/L2 outputs and fields.
   This tests copying of the nonlinear boundary; it is not the saving arm.
2. Nonidentity full-rank f64 basis plus the new mixed residual query: compare
   every consumer and complete output with the frozen arithmetic control,
   under the unchanged declared numerical allowances. This is the actual
   discriminator; setting Q=I everywhere would evade it.
3. The same source/birth equations in the independent f64 oracle: establish
   algebraic correspondence separately from proximity to f32 F21. Do not
   deploy the generic oracle as the production nonlinear operator.

Stop at the first material mismatch, retaining its operands and outputs.
If f64 algebra agrees but the mixed query fails the existing end-output
allowance, this **specific arithmetic policy** is numerically unsuitable as
an equivalent replacement. The cheap failure is valuable; no labels or rank
scan are needed. A succeeding tiny test still needs fixed, unprotected
production-shape conformance through the actual L4 path before a useful-rank
experiment. Preserve all prior failures and do not relabel them as passes.

There are two distinct goals: reproducing F21 within a declared tolerance,
and improving statistical prediction under a different stable arithmetic
regime. Reproducing its exact f32 bits is not a mathematical requirement of
the competition, and tiny ill-conditioned forced-Strassen fixtures do not
prove every approximation useless. However, this proposed intervention is
explicitly an equivalent-kernel prerequisite; it does not waive its agreed
tolerances. If achieving them requires different nonlinear arithmetic, treat
that as a separately preregistered scientific intervention, with its own
numerical rationale and fresh accuracy evidence. Better agreement with a
real-arithmetic oracle alone is not an accuracy win.

Also distinguish squared prediction drift from change in truth MSE. If
`delta=mean((candidate-reference)^2)` and the reference truth error is E,
then `abs(MSE_candidate-E) <= 2*sqrt(E*delta)+delta` by Cauchy-Schwarz.
A drift allowance `delta=0.01E` therefore permits a bound of `0.21E`, not
`0.01E`. Existing `2e-10` drift receipts must not be described as proving a
one-percent truth-MSE impact at assumed E=`2e-8`. A new numerical contract
may allocate an explicit score/MSE perturbation budget using the cross term;
that does not retroactively modify the prior conformance outcomes.

Only after the kernel passes should a separate contract test a declared common
rank or selection rule while leaving full residual rank and all ages fixed.
The rank-n reference may cost more than F21. Its job is to make that future
comparison interpretable; the target-r program's complete cost and later
paired score gate decide whether compression is useful.

## Scope of the recommendation

Proceed with this single mixed residual kernel discriminator, subject to the
repository's normal IMPORTANT contract and Skeptic. Do not resume the dense
decoder wrapper, treat ordinary-f64 costs as universal, select ranks from
the arithmetic table, or expand K4 in the same change. The unresolved issues
are the new query's numerical behavior, useful common rank, executed ledger,
and lifecycle feasibility. They are concrete falsifiable questions, not a
reason to ask another strategist for an unspecified algorithm.
