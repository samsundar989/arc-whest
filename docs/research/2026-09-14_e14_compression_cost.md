# E14 all-age compression cost and architecture screen

**Static IMPORTANT slice, preregistration `8f600fb`.** The literal mixed-f64
bridge is rejected as a cost-saving implementation. A specified direct-query
variant with dense f64 residuals and ordinary matmuls also fails a necessary
cost screen. These conclusions reject the two concrete spellings below;
**the shared-basis family remains open**. No reduced-rank operation, trajectory,
additional K4, labels or new data were measured here.

The primary control is frozen F21 (`d77dab0e2c740cc9f6cd7d9172adacc8da731cc5b40ca87798f3fb006e475a94`),
with measured graded cost `F0=928,457,637,478`. The plan's proposed 20% cost-saving
screen requires `F <= 0.8 F0 = 742,766,109,982.4`; this engineering screen is not
a promotion gate. Phase2 remains `C=F`, residual a 0.4-s failure cliff, wall120s
and memory8,192MiB. Static algebra supplies no wall/residual or process-memory
certificate.

## 1. Representation and exact operator boundary

For source age j write its two transported legs as
`A_j=Q F_Aj`, `P_j=Q F_Pj`. Keep its residual `D_j=Z_j L_j^T` at full residual
rank n. The current K3-only/scalar-K4 family has `s=e=0`, no additional K4 feed,
and can represent the residual with `L_j=I`, `Z_j=D_j`. The common-basis rank r
and residual rank n are different parameters. Reducing r does not delete an
age, diagonalize D or reduce the residual rank.

The complete nonlinear boundary must retain F21's table, all consumed Wick
terms, corrected mean, raw-to-centered transformation, scalar K4 projection,
`p111_3/p111_21` source subtraction, newborn `A=w1*C11`,
`D=diag(S3)+3*S21.T`, and final mean-only pruning. Projection is a representation
approximation when r<n; it cannot silently change these definitions.

The youngest source's P is **symbolically I** in F21's packed tower. A faithful
bridge to that layout keeps that newest source exact/dense until it ages, with
all older sources in the common basis. It must not reconstruct a projected
rank-r youngest P and then discard it in favor of I without declaring a
different approximation. At full rank this distinction is mathematically
invisible but its rounding was explicitly captured by E14. All ages remain.

The mixed-f64 control in
[`p2_compact_mixed_qr_e14.py`](../../local_sim/p2_compact_mixed_qr_e14.py)
is a **dense wrapper**: encode the complete source tower, reconstruct it, cast it
to f32, then call the unchanged reference `_layer`. Four fixed outputs match,
while exact source/zero recovery does not. That result is a useful boundary
control, not an autonomous compact implementation or a license to skip failed
f32 fullrank QR conformance.

## 2. Price conventions and exact literal-wrapper inventory

The installed flopscope0.12.1 pure cost functions and packaged tariffs give,
for ordinary dense untagged products,

`M32(a,b,c) = 2*a*b*c-a*c`,
`M64(a,b,c) = 2*M32(a,b,c)`.

A f32/f64 cast costs `C(E)=2E` for E elements in either direction. A dense f64
concatenation costs `2E`; a unit pointwise operation costs `2E`; a f64 identity
allocation is billed on its diagonal (`2n`), although its buffer holds n²
values. Zero allocation costs zero FLOPs and still consumes memory. Explicit
reduced QR of an n-by-r matrix costs
`Q64(n,r)=4*(2*n*r*r-floor(2*r^3/3))` for r<=n. Transposes/slices are views in
this spelling. Do not substitute a FLOP count for copy/allocator time.

Sources: installed `flopscope/_flops.py::matmul_cost`,
`numpy/linalg/_decompositions.py::qr_cost`, `_array_ops.py::{astype,concatenate,eye}`
and `data/default_weights.json`. These were read or evaluated as pure integer
cost functions, not by running a network. Strassen is not priced as an ordinary
matmul: the unchanged F21 bill already includes its exact executed L4 spelling.

For the **literal, already-tested fullrank** wrapper let k>=1 be incoming source
count, `M=M64(n,n,n)`, `Q=Q64(n,n)`. The following is its additional inventory:

| Component | Required operations / counted cost |
|---|---|
| Source import | Cast A,P,D,w2: `2*(3*k*n²+k*n)`; create one youngest f32 identity P, shared f64 identity L and one vector: `5n` combined |
| Complete weighted Gram | Per age, two n×n matmuls, two column scales, two additions: `2*k*M+8*k*n²` |
| Explicit QR | `Q`; Q and R/workspace exist even though only Q is retained |
| Direct factor projection | `Q.T@A`, `Q.T@P` for every age: `2*k*M` |
| Carried Gram | Two products, two column scales, two additions per age: `2*k*M+8*k*n²` |
| Leg reconstruction | `Q@FA`, `Q@FP`: `2*k*M` |
| Full residual materialization | Literal `P*s + 3*A*e + Z@L.T`, including zero s/e and identity L: `k*M+10*k*n²` |
| Packed copies and return casts | A_D,A_L,M,w2 have `(3k-1)n²+kn` elements: concatenate f64 and cast f32, `4*((3k-1)*n²+k*n)` |
| Transport, Wick carry, births, truncation checks, nonlinear table, scalar K4 and readout | The entire unchanged F21 operator remains; these are **not removed** by encoding |
| Basis rotation / separate compact transport | None in this literal wrapper: it rebuilds the complete Gram and projects every age anew; adding cached rotation on top would add work |

Thus exact additional costs under this operation spelling are

`Encode(k,n) = 6*k*M + Q + 22*k*n² + (2k+5)n`,

`Decode(k,n) = 3*k*M + (22k-4)n² + 4kn`.

Both are zero at k=0. These formulas reproduce the **already-recorded** tiny
charges exactly: n8/d6 gives208,020 encode+106,720 decode; n4/d4 gives11,316+
6,048. This is a check against existing artifacts, not a new trajectory.

At depth d, incoming k runs0..d-1, so K=`d(d-1)/2` and the modeled extra bill is

`9*K*M + (d-1)*Q + (44K-4(d-1))*n² + (6K+5(d-1))*n`.

At n1024/d16 it is `4,727,673,482,280` additional FLOPs; adding unchanged F21
would give `5,656,131,119,758`, about2.5721 times the2^41 budget. This is the
static program bill under the named primitives, **not an executed fullshape
ledger**. The rejection needs no extrapolated total: unchanged F21 alone already
exceeds the20%-saving ceiling. Even the two ordinary-f64 fullwidth leg
reconstructions alone at r=n cost `1,030,288,834,560` across120 ages, above that
ceiling before residuals or the operator.

Known identity/zero products can be removed when their equivalence is proved;
those redundant terms are priced here because the literal program executes
them. Removing all wrapper overhead still leaves unchanged F21 and fails the
cost-saving screen. Compression that always decodes the full tower and then
runs every unchanged packed contraction cannot obtain contraction savings.

## 3. Concrete direct-query alternative, and its necessary-cost refusal

A real-arithmetic alternative queries the compressed state without decoding it
into F21's entire packed tower. For each advanced age, with `L=I`, define
`A=Q FA`, `P=Q FP`, `R=Z`, and w its carried birth second derivative. Then

`D3 = sum_ages sum_columns[3*A*A*P*w + R*P*P]`,

`LA = 2*(A*P)*w`,

`LP = (A*A)*w + (2/3)*(R*P)`,

`D21 = [sum_ages(LA@FA.T + LP@FP.T)]@Q.T
       + (1/3)*sum_ages[(P*P)@R.T]`, followed by the off-diagonal mask.

These are exactly the existing compact contractions specialized to s=e=0 and
full identity residual L; distributivity gives the common Q return. They retain
the residual term, coefficient1/3 and final-layer D21 omission. They are not
floating-point operation-order equivalence to F21. B4 already demonstrates why
that distinction matters.

The table below uses B4/F21's **post-Wick stored-state convention**: after the
current nonlinear consumption, old Q and Z are row-scaled by current w1 before
the next state is formed. The next layer therefore transports those stored
arrays as `W@Q` and `W@Z`. If an implementation instead stores pre-Wick arrays,
the equivalent transports are `W@(wick[:,None]*Qraw)` and
`W@(wick[:,None]*Zraw)`, with exactly one Wick application. The newest birth is
appended after old-source Wick scaling and must not inherit that preceding
Wick. A scale after W is not an equivalent operation.

For a concrete all-f64 source/basis implementation using **ordinary** matmuls,
let m be older shared ages and reserve one newest dense exact source. Its full
work inventory must include the following, in addition to unchanged mean/full
covariance propagation and the complete F21 nonlinear table:

| Stage | Required products and remaining work |
|---|---|
| Common-basis transport | One `M64(n,n,r)` for WQ; current Wick row scaling `2nr`; cast W to f64 once per layer (`2n²`) |
| Full residual transport | One `M64(n,n,n)` **per old age**, plus Wick row scaling `2mn²`; keep residual rank n |
| Source leg values for queries | Two `M64(n,r,n)` per old age for A,P; materialize no k-wide tower but still two dense n² legs for the current age |
| D3 | Per age, six pointwise products/scales, one addition, n reductions over n columns, then age accumulation; retain final D3 query |
| D21 shared-coordinate accumulation | Per old age, `2*M64(n,n,r)` for LA@FA.T and LP@FP.T; after ages, one `M64(n,r,n)` shared return |
| D21 full-residual return | Per old age, `M64(n,n,n)` for `(P*P)@Z.T`; its scale/add and the final off-mask remain |
| D21 pointwise work | AP, PP, A², ZP, carried-w scales, LA/LP combination and coordinate/residual accumulations: O(mn²+mnr+n²) explicitly materialized work; no assumption it is free |
| Newest exact age | Separate dense transports and F21-compatible dense query; P's input identity is symbolic. Its birth/advance/query cost is additional, excluded from the lower bound below |
| Nonlinearity, scalar K4 and births | Cast consumed fields to the declared f32 boundary; execute all Wick/raw/centered/scalar-K4/subtraction operations; form dense newborn A,D and scalar/vector state; cast inserted fields back to f64 |
| One weighted-Gram admission/join | For a general dense source, four `M64(n,n,r)` products to apply its A/P Gram to Omega; old-state sketch adds `M64(r,n,r)+M64(r,r,r)+M64(n,r,r)`; identity newborn P permits a proved special case but is not free by assertion |
| QR / projection | `Q64(n,r)`; two `M64(r,n,n)` for source projection; deterministic sketch Omega must be constructed/stored/charged |
| Rotation of retained factors | `M64(r,n,r)` for T=Qnew.T@Qold, then `2m*M64(r,r,n)` for all FA/FP; no age can disappear |
| Carried Gram rotation/update | Two `M64(r,r,r)` for TGT.T; two `M64(r,n,r)` for new factor Gram; their column scaling/addition terms and copies |
| Final readout / state publication | D3 and required scalar-K4/Wick terms, final output cast/stack; no final birth, QR or D21; all immutable replacements and concatenations must be charged |

This is an operation inventory for a concrete direct-query spelling, not an
accepted complete candidate. The lower-bound rejection deliberately grants
**zero cost to every remaining row**, including all pointwise/copy/QR work, so
unspecified fusion of those terms cannot rescue this spelling.

At layer l (zero based), k=l incoming ages and m=max(l-1,0) older ages. Across
n1024/d16, older full-residual transports number105, and older D21 residual
returns on nonfinal layers number91. Their sum196 ordinary-f64 dense products
alone costs

`196*M64(1024,1024,1024) = 841,402,548,224`.

That is90.62% of F21, **already98,636,438,241.6 above the80%-cost ceiling**.
No newest source, basis work, nonlinear backbone, scalar K4, QR, rotation,
readout, cast or copy is included. The lower bound is therefore sufficient to
reject this ordinary-f64/full-residual direct-query spelling as well.

It is not a lower bound for all possible algorithms. A fully costed Strassen
residual kernel, a different precision boundary, exploitable residual structure,
or a genuinely different query identity could alter these mandatory products.
No such alternative was implemented or demonstrated here. In particular,
storing Z in f32 does not by itself halve a product with f64 PP: NumPy promotion
still makes that product f64. A cheaper mixed-query policy must specify both
arithmetic operands and its numerical conformance; it cannot borrow the
reference-wrapper result.

## 4. Storage and co-live-buffer obligations

For the exact fullrank remedy, k>=1 source representation payload is

`P64(n,k)=8*((3k+3)n²+(k+2)n)` bytes; `P64(n,0)=0`.

This counts Q, Gram and shared identity L; FA/FP/Z for every source; each w2;
and shared zero/one vectors. It gives9,664 bytes for n8/k5 and1,696 for n4/k3.
At n1024/k15 it is402,792,448 bytes (~384.13MiB), **representation only**.
It excludes the rest of the process and is not a memory-cap result.

For r<=n, a corresponding all-shared representation has payload

`8*(nr+r²+2krn+kn²+n²+(k+2)n)`.

For the faithful youngest-exact convention use k=m older ages in that expression
(0 when m=0), plus the youngest f32 A,D,w2 payload `4*(2n²+n)`; P is symbolic.
The full-rank residual term kn² persists even as r shrinks.

The reconstructed f32 packed tower occupies
`T32(n,k)=4*((3k-1)n²+kn)` for k>=1. A streamed implementation must still account
for simultaneous buffers, rather than quote P64 as peak memory:

- **Encode:** input T32, cast source A/P buffers `16kn²`, residuals retained in
  P64, complete Gram/sketch, Q/R and QR workspace, projected factors and carried
  Gram during construction. The literal dense Gram is n² even if a later basis
  is narrow.
- **Decode:** retained P64, A/P reconstructions `16kn²`, materialized D list
  `8kn²`, a largest f64 concatenation up to `8kn²`, new T32, old T32 until it is
  released, and pointwise/cast scratch. A conservative array-only stage inventory
  is `P64+2*T32+8*(4k+2)n²`, plus the non-tower baseline buffers; it is not a
  measured process bound.
- **Direct queries:** dense current-age A/P and pointwise matrices, coord n×r,
  residual n×n and D3 n; transport replacement Z buffers can coexist with the
  old Z list. Keep both old/new Q, factor rotations, T, Gram and QR R/workspace
  during a join. An immutable list-building implementation may temporarily hold
  all `mn²` new residual entries and `2mrn` rotated-factor entries beside old
  state; no implicit in-place aliasing is allowed.
- **Every route:** incoming f64 weight list `8dn²`, intended f32 weight copies,
  any additional W64 view/copy, mean/covariance, dense newborns, the nonlinear
  table's live fields, final output, Python/flopscope metadata, BLAS/QR workspace
  and actual allocator behavior. Archive/library memory is not covered by an
  array payload formula.

Release/reuse scheduling can lower simultaneous payload without changing the
math, but it must be implemented and verified. E14 trace retention is diagnostic
and must not be counted as a production optimization or used as a process-cap
certificate.

## 5. Smallest unresolved implementation and stop point

Stop the literal decoder wrapper and the specified ordinary-f64/full-residual
direct-query design before rank or labels. The next useful artifact is a small,
independent **fullrank production-precision query operator**, not another rank
sweep or dense wrapper. It must:

1. Keep F21's complete nonlinear/scalar-K4/birth conventions and exact youngest
   source semantics; fix the arithmetic backend and representation precision.
2. Replace the dense residual transport/query bottleneck with one explicit
   kernel/identity whose full work is below the residual-only obstruction; include
   copies/casts, all retained ages and residual rank n. If using Strassen, execute
   and price its complete glue and numerical boundary, not a `(7/8)^L` discount.
3. Compare its fullrank reconstructed contractions and complete trajectories with
   the independent f64 oracle and a compatible F21 arithmetic control, preserving
   original failures and the declared output allowance. No free dense fallback.
4. Produce a complete executed ledger and liveness inventory before any new
   reduced-rank contract. Only then is changing common-basis rank a separable,
   interpretable approximation.

The static obstruction is specific enough to guide that implementation, but
available repository evidence does not yet provide the missing kernel or prove
it can meet the screen. Do not convert this implementation rejection into a
scientific closure of shared-basis or full-residual methods.
