# Accuracy after E14: simplify the declared closure before compressing its feed

September 14, 2026. Independent strategy review; static source reads, algebra and
integer cost arithmetic only. No candidate, probe, truth read, seed claim, vault
touch or workload was run. These are proposed interventions, not accepted
mathematical tests, accuracy evidence or release claims.

**Recommendation:** do not make the exact two-metric connected-tree implementation
the prerequisite for an accuracy attempt. Its transport preserves one chosen
projection exactly while that projection discards an uncontrolled fourth-order
residual every layer. There is no evidence that this particular exactness is the
valuable part. A scalar-metric closure has a much simpler coherent persistent
feed; a covariance-aligned version can share F21's existing full-rank source
matrices. This is a newly specified approximation, not a cheaper implementation
of the accepted E14 two-metric recipe.

The best bounded first step is a no-label dense oracle for the scalar-metric
feed identity below and its F21 implementation points. It can proceed on the
working full-rank F21 tower without waiting for compact QR conformance. Whether
any such closure improves accuracy remains unknown.

## Evidence limits

The E14 connected-tree note defines its approximation precisely, including
sixteen Gaussian trees, repeated-index anchors, projection, complete consumers
and feed. Its independent ACCEPT establishes those equations and the cost of
one spelling. It does not establish that the trees approximate the relevant
fourth-order error, that the discarded residual is small after another random
linear map, or that a positive law exists. At repeated coordinates correlation
is one; a small-correlation expansion gives no uniform remainder control there.

T1 is evidence against deleting pieces of its particular coupled factored
AUGMENT program. Its published local comparison has SIMPLE/MINIMAL arithmetic
ratio 0.01346 on 24 exposed DESIGN networks. Deleting the feed also changed the
regenerated K4, through `K211_contrib`; it did not isolate the value of persistent
feed at fixed K4. The later legsonly arm falsified the broad claim that every
deletion returns to Gaussian behavior. It does not prove every affordable
matrix-K4 closure fails, and no new T1 read or remeasurement is proposed. See
the T1 result, corrections and second ablation in
[LOG](../../experiments/LOG.md) and
[the preserved summary](../../experiments/2026-09-10_slice5/t1_augment_lite_v2.json).
Its manifest records per-half truth noise `1.1552182657290076e-9`; no raw MSE is
re-estimated here.

The external implementation supplies a narrower, useful existence observation:
its scalar-metric covariance-aligned K4 channel uses existing source factors
and thin vectors for persistent feed. Its generator also exposes public-fitted
layer coefficients and changes beyond the K4 channel; these are not portable
evidence for F21 or coefficients to copy. The author reports use-and-feed
improvements and harmful use-only ablations, on reused public development
networks with unavailable truth-noise floors. Neither the author report nor
source inspection is our DECIDE evidence. Primary sources inspected today:
[pinned V17 generator](https://github.com/504aldo/whest-p2-cumulant-k3/blob/18c17e2d7a9aeacd399cfc2c6b571e4e16dbfb45/generators/make_v17.py),
[pinned ablation table](https://github.com/504aldo/whest-p2-cumulant-k3/blob/18c17e2d7a9aeacd399cfc2c6b571e4e16dbfb45/docs/ablation_table.md).
The equations below are derived directly in the repository's tensor convention;
no public-fitted table is adopted.

## 1. An exact simplification within a different, scalar-metric closure

Let `b=w(1,1)`, `a=w(2,1)` use F21's actual clipped/floored coefficients.
At a preactivation define the extra cumulant

```
Q = m L_I(G)
L_I(G)_ijkl = (G_ij delta_kl + G_ik delta_jl + G_il delta_jk
               + G_jk delta_il + G_jl delta_ik + G_kl delta_ij)/6.
```

`G` is symmetric, and `m=init_scale` is a declared scalar metric (2 at the
target shape). This replaces the E14 metric `B B.T` by `m I`; it is an
approximation even if `G=B Gout B.T` is computed exactly. Keep F21's separate
scalar K4 account unchanged in rule and order. Do not subtract the scalar
account again at preactivation.

The extra mean/power slices, off-diagonal where indicated, are

```
q4_i = m G_ii
q22_ij = m (G_ii+G_jj)/6       i != j
q31_ij = m G_ij/2             i != j.
```

Define `U(y,S)_ijk=y_i S_jk+y_j S_ik+y_k S_ij`, with symmetric `S`.
For all three coordinates distinct, `Q_iijk=m G_jk/6`. Therefore the declared
one-quadratic/two-linear feed is, on that support,

```
J_ijk = m/12 [a_i b_j b_k G_jk + a_j b_i b_k G_ik
                                    + a_k b_i b_j G_ij].
Jred = U((m/12)*a, diag(b) G diag(b)).
```

This is sufficient for the **entire next K3 tensor**, not just its readouts.
Let `D(T)=from_dstensor(rep3(T),rep21(T))`. At birth this operator retains
exactly the entries with any repeated coordinate and is zero on all-distinct
entries. Hence `J-Jred=D(J-Jred)` and

```
J-D(J) = Jred-D(Jred).
Tnew = Tcarry+Tpath+Jred
       + from_dstensor(K3-rep3(Tcarry+Tpath+Jred),
                       K21-rep21(Tcarry+Tpath+Jred)).
```

This proves equality of the full new tensor in real arithmetic. Subsequent
linear maps and Wick scalings preserve that equality. Do not separately
append the full `J` as well. The discarded pieces here cancel against the birth
correction; this is an exact cancellation, unlike dropping a K4 projection
residual. On a diagonal `G`, `Jred` itself has only repeated support, so it can
be omitted entirely **after** this birth identity is applied.

A manual width-three witness fixes the normalization. Take `G_23=G_32=t`,
other entries zero. Then
`Q_1123=mt/6`, `J_123=ma_1 b_2 b_3 t/12`, and `Jred_123` is the same.
Every `from_dstensor` term is zero at `(1,2,3)`. Changing 1/12 to 1/6 doubles
this surviving entry; no repeated correction can hide that mutation.

## 2. A complete affordable first variant: arbitrary G, scalar metric

Keep the E14 four-product connected-tree H calculation, its symmetric K22
anchor, and its scalar-order rule. Set

```
h=trace(H)/n
Gout=6*(H-h I)/(n+4).
Gpre=B Gout B.T
Qpre=m L_I(Gpre).
```

Regenerate at each nonfinal output; preserve every old K3 feed. Store one
symmetric dense `S=diag(b)Gpre diag(b)` and one vector `y=(m/12)a` per feed
birth. Apply each subsequent map `A` as `S <- A S A.T`, `y <- A y`.
Queries are `D3=3*y*diag(S)` and `D21_ij=2*y_i*S_ij+y_j*S_ii`.
Subtract their birth slices once as above. This replaces five dense matrices
per E14 birth by one matrix and a vector. Peak feed payload at 14 births is
`14*(n^2+n)` numbers, about 56.055 MiB in f32, before baseline and live scratch.

At depth16, the four-product regeneration costs60 square products; two-sided
G transport with final diagonal-only pruning costs29. Feed transport costs
`2*105-14=196` square products; Wick scaling, vector carries and readouts are
quadratic. Total285 extra square products, compared with E14's1069. Using
the same ideal depth4 tariff `S4(1024)=1,301,696,512`, baseline plus square
arithmetic is `1,299,441,143,398`. The corresponding ordinary-f32-square
sum is `1,540,191,632,998`. Both are below `2^41` **before omitted overhead**;
neither is a complete counted ledger. The ideal-Strassen number is1.39957
times F21's count, requiring more than28.55% lower raw error to break even
before overhead. Thus being affordable alone is a weak reason to build it.

This variant has a precise dense oracle and no unknown rank or fitted parameter.
It is preferable to an unpriced exact two-metric construction if preserving the
full tree-generated matrix is the intended scientific question. It still has
the tree truncation, fourth-tensor projection, scalar-metric approximation and
finite-hyperedge response risks. No gain is predicted as established fact.

## 3. A cheaper complete proposal sharing the existing F21 tower

For a bounded accuracy hypothesis, impose one further named approximation:
the extra core's off-diagonal part aligns with current covariance. Define the
output core by

```
Cout_s=(Cout+Cout.T)/2
O=offdiag(Cout_s)
g=6*(diag(H)-h)/(n+4)
lambda=6/(n+4) * sum(offdiag(H)*O)/sum(O*O)
Gout=diag(g)+lambda*O.
```

If the denominator is exactly zero, define lambda=0. Nonfinite quantities fail
the candidate; no fitted clipping/epsilon or PSD repair is implicit. This is
the real-arithmetic Frobenius projection of the accepted tree-derived core's
off-diagonal part onto covariance, and its diagonal is unchanged. It is a
label-free coefficient definition, not a learned or public-fitted table. This
projection may be scientifically poor; that is the hypothesis to test. A
constant theoretical lambda or a freshly DESIGN-fitted lambda would instead
be a different intervention with its own contract.

At the next preactivation use

```
dG = (B.*B)@(g-lambda*diag(Cout_s)) + lambda*diag(Cpre)
Ghatpre = diag(dG) + lambda*offdiag((Cpre+Cpre.T)/2)
Qpre = m L_I(Ghatpre).
```

In real arithmetic `dG` is the exact diagonal of `B Gout B.T`; the off-diagonal
contribution from `B diag(g-lambda*diag(Cout_s)) B.T` is explicitly discarded.
There are therefore two different approximations here: a scalar metric and a
covariance-aligned off-diagonal core. This is not exact transport disguised by
notation. Floating-point discrepancies of the displayed diagonal identity
must be recorded, not asserted bit-identical to another association.

The diagonal of `Ghatpre` produces no all-distinct feed. Let `C11` be F21's
unmodified zero-diagonal preactivation covariance and use its existing birth
factor `A0=diag(b) C11`, birth map `P0=I`. Let `r=b` be fixed at birth. Then
the full persistent reduced feed is

```
y0=(m*lambda/12)*a
X0=A0 diag(r)
S0=(X0+X0.T)/2
Jred=U(y0,S0).
```

For a cumulative later linear/Wick map `P`, F21 already carries `P` as that
source's `M` and `A=P A0` as its `A_L`. Consequently

```
X=A diag(r) P.T
S=(X+X.T)/2
y=P y0
t=diag(X)=rowsum((A.*P)*r)
D3_feed=3*y*t
D21_feed=diag(y)*(X+X.T)+outer(t,y).
```

Only the transported vector `y` and static vector `r` are extra per source;
there is no extra dense carry, rank approximation or age dropping. Store all
14 feed births. Their additional persistent payload is `28*n` numbers,
112 KiB in f32, apart from optional packed/vector scratch. Birth queries have
`t=0`, `D3=0`, and
`D21_ij=y_i*b_i*b_j*(C11_ij+C11_ji)`; subtract that once in S21. The zero
birth D3 follows from the zero-diagonal C11, not from deleting a later D3.

An unfused implementation computes one X per old birth at a nonfinal query,
then its transpose and the displayed cheap operations. There are91 such
products (the extra feed is zero before the first nonlinearity and no final
D21 is needed). The final D3 uses t without X. Together with60 regeneration
products this gives151 extra square equivalents. Baseline plus ideal square
arithmetic is `1,125,013,810,790`, or1.21170 times F21; the raw-error reduction
needed merely to break even is17.47% before overhead. This is a useful
numerically straightforward reference, not yet a metered candidate.

### Exact real-arithmetic fusion, including finite-precision caveat

For one source write `D=A_D`, `P=M`, `A=A_L`, `w=w2b`. F21's existing D21
numerator can be evaluated with the following three products, now including
the feed (all column scales below use the appropriate source's fixed vector):

```
L1=2*(D.*P)+3*(A.*(A*w)) + 3*(y[:,None]*A)*r
L3=6*(A.*P)*w             + 3*(y[:,None]*P)*r
acc=L1@P.T + (P.*P)@D.T + L3@A.T
D21=offdiag(acc/3 + outer(t,y)).
D3=D3_F21 + 3*y*t.
```

Concatenate L1/P and L3/A across all births just as F21 packs its tower; the
outer products can be one `(n,k)@(k,n)` with k<=14. Sum the y*t vectors.
This adds no square matmul and removes no source age. It uses the exact
symmetrized S, so small covariance asymmetry does not define a nonsymmetric
fourth or third tensor.

**This moves `w2b` from the right operand to the left of F21's third product.**
It is a new floating-point spelling. The active fused version must be compared
with the unfused active reference under the unchanged numerical allowance.
When the extra branch is disabled, call frozen `_dslice21` directly rather
than reassociating the baseline and calling an algebraic zero a parity pass.
No widening of failed B4 tolerances or compressed decoder is required.

With fusion, the only added square products are the60 tree-regeneration
products. Baseline plus ideal square arithmetic is `1,006,559,428,198`,
1.08412 times F21, requiring7.76% lower raw error before overhead. These
are feasibility estimates, not a complete official count. Meter the complete
candidate including the changed multiplication inputs.

The full remaining tariff contains: 15 generations of four tree products,
their diagonal scales and two matrix-vector products; symmetric averages of
new C/H/anchors; tree pointwise assembly; K22 row sums and scalar projection;
lambda's inner products and zero-denominator branch; 15 next-layer dG
matrix-vector queries; all new marginal/pair hyperedge terms; unchanged
centering on corrected moments; 14 birth vector/scalar preparations and
repeated-slice corrections; 105 feed vector carries; 91 nonfinal feed
readouts/fusion operand additions and final14 D3 queries; all old-source Wick
scalings; packing, masks, copies, casts, scalar reads and reduction charges.
The explicit dense storage alternative must additionally price X per query.
No cubic PSD projection, hidden eigensolve, rank fit, QR or nonlinear bank
construction is part of this program. Full-width residual operation count,
memory enforcement and server lifecycle headroom remain unmeasured.

## 4. Every consumer and source insertion point

Use the existing SIMPLE term manifest and F21 centering. For each p=1..4,
before centering add `delta a_p=q4*w(4,p)/24`. For each pair `(p,q)` in
`(1,1),(2,1),(2,2)`, add, for i!=j,

```
delta P_pq(i,j)=q31_ij*w_i(3,p)*w_j(1,q)/6
               +q22_ij*w_i(2,p)*w_j(2,q)/4
               +q31_ji*w_i(1,p)*w_j(3,q)/6.
```

P is the connected moment of powers, not the raw joint moment. Reconstructing
raw moments would use `R_pq=P_pq+a_p*a_q.T`; do not subtract means twice from
F21's covariance. Run the unchanged pK-to-K centering using updated a1 in
every term. This updates mean, marginal variance, K3, K21, K4, K22 and
off-diagonal covariance consistently with the declared finite response.
No products of the extra hyperedge with incoming K3, extra K4 or covariance
edges are added implicitly. This finite manifest is an approximation, not a
complete Edgeworth response around a correlated Gaussian.

Relevant frozen insertion points in
[F21](../../estimator_k3_strassen_L4_f21.py): `_advance` around604 for extra y
transport; `_wick_tower` around616 for old-y Wick scaling; `_dslice3` and
`_dslice21` around626/638 for active feed queries; `_layer` around767 for
preactivation core/use terms before K centering; birth construction around886
and subtraction around933; scalar update after the birth; and final-layer
mean-only branch. The newborn y must not receive the just-finished Wick a
second time. The existing A/M state is already post-Wick for old sources.

After corrected K4/K22 are available, compute H anchors, the new g/lambda and
the **original-order** scalar `lap2_d4` projection. Since trace(g)=0 only in
real arithmetic, record finite-precision trace drift; do not silently zero
or transfer it. Initialize extra state to zero for Gaussian input; regeneration
after the first nonlinearity can be nonzero. Preserve the width-two bypass,
all generic shapes and inherited safety behavior. The final layer computes
only extra q4, old-feed D3, and corrected mean; no final birth, lambda or H.

## 5. Cheapest discriminating contract, proposed only

The immediate task should freeze **only the supplied scalar-metric feed and
shared/fused-U identity**, not select or implement a regeneration rule. Supply
explicit G, covariance, lambda and derivative fixtures to that identity oracle.
After its result and independent review, synthesize before choosing between the
specified closure variants. The aligned-tree rule is a concrete later candidate
proposal; its definition above avoids leaving an unspecified closure dependency.

1. Independently enumerate the six lift terms at fixed n=3 and4, with explicit
   signed nonconstant symmetric G, unequal derivative vectors and off-diagonal
   covariance. Enumerate the full J and `from_dstensor` tensor independently;
   compare `J-D(J)` with `Jred-D(Jred)` at every index and after one nontrivial
   dense map. Include a diagonal G and the manual `(1,1,2,3)` witness above.
   Compare the complete dense next K3 tensor through two subsequent nonlinear
   layers, including all consumer updates and births; matching repeated
   readouts alone is insufficient.
2. If a later closure contract selects the tree-alignment proposal, independently
   enumerate the sixteen trees, then the anchored H and specified
   covariance projection. Test zero denominator, signed lambda and asymmetric
   roundoff input symmetrization. Compare every marginal, connected pair,
   centered moment, q31 orientation, scalar account and next full K3 tensor.
3. Compare dense U transport, shared A/P/y transport, and fused/unfused D3/D21
   at every age. Force nonzero extra state in tiny shapes. Mutations must catch
   1/6 versus1/12, missing y transport, missing old-y Wick scaling, new-y Wick
   applied twice, omitted transpose, forgotten outer(t,y), diagonal-G feed
   retained without subtraction, omitted birth subtraction and double scalar
   subtraction. Disabled-extra must preserve frozen F21 output arithmetic.
4. Register actual-f32 fused-versus-unfused checks separately from real-arithmetic
   algebra. Test the existing fixed full-depth negative-variance fixture and
   finite/covariance/moment-matrix admissibility checks. A failed covariance or
   derivative-amplification screen stops this recipe; selecting damping from
   that failure is a new intervention. Tiny equality is not a width1024 bound.
5. Only after these gates, meter the complete source bytes and all-age ledger.
   Cost-only/full-shape fixtures need no protected truth. A useful counted price
   and admissible trajectory can justify a separately registered fresh paired
   accuracy comparison, with Samuel's new predictions and canonical arithmetic
   score gate. Neither exposed T1 nor the external public table may fit lambda.

The cheapest falsifier is step1: a handful of n3 tensor entries can disprove
the only feed simplification on which the cheaper design depends. Step3 is the
next decisive check because fusion changes floating-point association; perform
these identity checks before the conditional tree work in step2. There is no
reason to debug either issue with a new truth block. This is a proposed task
order, not an experiment run or authorization to skip the written prediction,
independent Skeptic or applicable preflight.

## Decision boundary

Exact sharing alone also improves the original E14 count: its generic first
factor is a transported birth diagonal and can share that generation's existing
F21 M with a fixed column scale. That saves91 additional carries beyond E14's
newborn pruning, reducing1069 products to978. The same square-only total is
`2,201,516,826,214`, still `2,493,570,662` above budget before overhead. This
is a scoped improvement, not a full exact-sharing lower bound; other sharing
may exist. Merely finding two more products to delete would leave essentially
no overhead or score headroom and does not make this the preferred next task.

The proposed scalar-metric route answers a more useful bounded question: can
a complete added fourth-order response improve working F21 while purchasing
its own cost? It reuses the proven tower, retains persistent feed, specifies
its extra approximations openly, and provides an independent tiny falsifier.
Its accuracy benefit is SPECULATIVE. The tensor cancellation and sharing are
real-arithmetic identities pending independent implementation checks. Existing
E14 and T1 failures remain unchanged and do not become positive evidence.
