# E14 Stage 3: fourth-order accuracy derivation and implementation stop

**Status: bounded derivation, no mathematical probe or accuracy measurement.**
No candidate, instrument, seed ledger, protected truth, vault or frozen archive
was changed. This note is not an independent Skeptic acceptance, a claim of
stability, or a new promotion protocol. Any future mathematical test needs its
written prediction/protocol committed first. A score experiment additionally
needs Samuel's predictions and a fresh registered decision block.

**Decision: park the extra-K4 implementation.** A complete inexpensive,
well-defined regeneration rule on the working F21 recurrence has not been
derived. The missing object is particularly precise: off-diagonal entries of
the fourth-cumulant contraction `H_ab = sum_c K4_abcc`, together with a declared
approximation to its unretained residual after each ReLU. F21 does not compute
these entries. Replacing them by a multiple of covariance is the previously
failed implementation, not a new derivation. This does not reject the whole
family of covariance-aligned approximations. Below are the consumer, centering,
overlap and K3-feed equations a distinct proposal must satisfy. They identify
what a future tiny implementation would have to test without claiming one is
ready now.

The subsequent bounded [connected-diagram derivation](2026-09-14_e14_connected_k4.md)
defines one explicit approximate regeneration rule. It supersedes the missing-
definition status for that particular approximation; it supplies no stability,
accuracy or economical implementation result. Exact nonuniqueness is not an
objection to specifying and testing a coherent approximation.

## Evidence and motivation

CONFIRMED source facts: the working operator is
`estimator_k3_strassen_L4_f21.py::_layer`, with its frozen SIMPLE table
`experiments/2026-09-02_wp14_fnp/k3_coefs_simple.json`. It retains the whole K3
tower, its nonlinear birth subtraction, and scalar radial K4. The old compact
off arm is a different nonlinear recurrence and cannot be the control here.

The external motivation is the *dated repository review* of the pinned 504aldo
author ablations: consistent K4 use plus K3 feed was more useful there than use
without feed, while diagonal regeneration was harmful. See
[frontier review](2026-09-14_frontier_research.md) and its pinned primary-source
links. Those observations motivate checking a complete connected-moment
extension of F21; they supply neither its missing coefficients nor its
regeneration rule. No new external search was performed for this note. The
local T1 comparison concerns coupled **8n factorable AUGMENT**, not isolated
matrix K4 or full unrestricted AUGMENT.

Relevant scoped negative evidence remains intact:

- [Frozen-source derivation](../../experiments/2026-09-11_stage2_compact_p07/source_derivation.md)
  and the [actual executable trace review](../../experiments/2026-09-11_stage2_compact_p07/source_cumulant/executable_trace/skeptic_verdict.md)
  show why central fourth moments and invented independent mixed cumulants
  cannot be interpreted as a physical K4 source.
- [Dense source review](../../experiments/2026-09-11_stage2_compact_p07/source_dense/skeptic_verdict.md)
  accepts supplied-tensor transport and the declared quadratic feed only. It
  explicitly does not accept a complete nonlinear closure.
- [Integrated n8 diagnostic](../../experiments/2026-09-11_stage2_compact_p07/integrated_first_source/n8_variance_diagnostic.json)
  and [shortcut review](../../experiments/2026-09-11_stage2_compact_p07/integrated_first_source/shortcut_skeptic_verdict.md)
  preserve negative variance despite algebraic conformity. The synthetic bank
  was carried through raw linear maps while the surrounding state changed
  nonlinearly. Removing storage duplication preserved that failure.

INFERRED: consistent raw-power handling on F21 would eliminate those particular
consumer mismatches. SPECULATIVE: any cheap extra-K4 closure thereby improves
score or stays numerically admissible. The latter does not follow from the
former.

## 1. Fix the convention before adding equations

Use F21's orientation `B = W.T`, preactivation mean `m = B mu`, covariance
`C = B cov B.T`, and the same variance/saturation policy. Let

`w_i(k,p) = E[(d/dz)^k ReLU(z)^p]`, for `z ~ N(m_i,C_ii)`.

This derivative notation describes the formal unsaturated algebra. An
executable F21 extension must use the same clipped-alpha and variance-floor
semantics as the frozen operator; in their active regions its coefficients
need not be derivatives of this one Gaussian expectation. It must not silently
replace those policies with an unsaturated implementation.

The existing finite Wick table determines which products of covariance and K3
are retained. They remain exactly unchanged. Let `Q_abcd` be the proposed
*additional*, symmetric fourth cumulant at this preactivation. It is a tensor
of units activation-to-the-fourth, not a standardized excess kurtosis or a
central fourth moment. Define, before off-diagonal masking,

`q4_i = Q_iiii`, `q22_ij = Q_iijj`, `q31_ij = Q_iiij`.

There is an important source-level trap: F21's pair `pK[(p,q)]` fields are
**connected moments of powers**, not complete joint raw moments. Write them
as `P_pq(i,j) = Cov(Y_i^p,Y_j^q)`, with `Y=ReLU(z)` under the selected formal
closure; write `a_p(i)` for its one-coordinate raw moments. Thus

`R_pq(i,j) = P_pq(i,j) + a_p(i) a_q(j)`.

This relation is a definition of the formal raw moments reconstructed from
the selected approximation; it does not assert existence of a positive law.
In particular, F21 sets `K11=P11`, not `P11-a1 a1.T`. Subtracting means again
would corrupt the working covariance. Conversely, updating only the mean
while keeping Gaussian marginal second moments would corrupt its diagonal.

## 2. A precisely bounded extra-hyperedge response

Here is one *specified response truncation*, conditional on receiving `Q`.
Add one fourth-cumulant hyperedge and no products of that edge with incoming
K3, another extra-K4 edge, or cross-covariance edges. This is a formal extension
of the frozen SIMPLE term manifest, not a full Edgeworth expansion around the
correlated Gaussian. Those omitted mixed diagrams must be named, rather than
silently described as all fourth-order physics.

For each `p=1,2,3,4`, add

`delta a_p(i) = q4_i w_i(4,p) / 24`.

For each existing pair consumer `(p,q)=(1,1),(2,1),(2,2)`, add

```
delta P_pq(i,j) = q31_ij w_i(3,p) w_j(1,q) / 6
                 + q22_ij w_i(2,p) w_j(2,q) / 4
                 + q31_ji w_i(1,p) w_j(3,q) / 6,  i != j.
```

The coefficients are the direct derivative-placement factorials
`1/(3!1!)`, `1/(2!2!)`, `1/(1!3!)`. The missing `(3,2)` coefficient already
exists in the frozen Wick table; no higher derivative than four is required
for these current consumers. The `(1,2)` field is the transpose of `(2,1)`.
Symmetrize equal-power fields only after assembling both oriented terms.
In particular `(q31+q31.T)*outer(w3,w1)` is generally wrong.

Pure marginal terms `q4_i w_i(4,p) w_j(0,q)/24` belong to the reconstructed
raw `R_pq`, not this connected `P_pq`. They cancel from the connected response
at the independent-marginal hyperedge order. Extra-K4/cross-covariance product
diagrams would require a different declared manifest.

All scalar and pair raw-power changes then go through the **unchanged** F21
centering polynomial, using the same updated `a1` everywhere:

```
mu_i = a1_i
V_i = a2_i - a1_i^2
K11_ij = P11_ij
K3_i = a3_i - 3 a1_i a2_i + 2 a1_i^3
K21_ij = P21_ij - 2 a1_i P11_ij
K4_i = a4_i - 4 a1_i a3_i - 3 a2_i^2
       + 12 a1_i^2 a2_i - 6 a1_i^4
K22_ij = P22_ij - 2 a1_j P21_ij - 2 a1_i P12_ij
         + 4 a1_i a1_j P11_ij - 2 P11_ij^2
cov_out = diag(V) + offdiag(P11).
```

These equations retain the nonlinear products generated by the working
centering convention. Linearizing centering in `Q` would be another change.
They also specify actual extra raw moments via `R_pq=P_pq+a_p a_q`. On the last
layer only the updated mean and its `q4` contraction are needed.

## 3. K3 births require information absent from current pair readouts

The associated one-quadratic/two-linear extra-K4 feed is the symmetric tensor

```
J_ijk = (w_i(2,1) w_j(1,1) w_k(1,1) Q_iijk
         + w_i(1,1) w_j(2,1) w_k(1,1) Q_ijjk
         + w_i(1,1) w_j(1,1) w_k(2,1) Q_ijkk) / 2.
```

This is the already-supported declared feed truncation, not the exact full
ReLU response. Let `Tcarry` be F21's old tower after linear transport and
`w(1,1)` scaling, and `Tpath` its existing two-leg Gaussian path block. A
complete implementation must append the separate `J` source and use

```
S3  = K3  - rep3(Tcarry + Tpath + J)
S21 = K21 - rep21(Tcarry + Tpath + J)
Tnew = Tcarry + Tpath + J + from_dstensor(S3,S21).
```

Here `from_dstensor` is F21's `A_D=diag(S3)+3*S21.T`, `M=I` representation;
the masks and symmetrization factors stay unchanged. Consequently `Tnew`
has exactly the computed repeated output slices, and its all-distinct entries
retain `J` once. Updating `S21` alone discards that all-distinct feed; adding `J`
without its repeated subtraction double-counts it. No K3 ages may be dropped.

A direct algebraic obstruction is visible already at width three. Define a
symmetric `U` equal to one on the permutations of `(0,0,1,2)` and zero
elsewhere. Then **all** its `q4`, `q22` and `q31` are zero, but

`sum_c U_12cc = 1`, and `J_012 = w_0(2,1) w_1(1,1) w_2(1,1)/2`.

Thus agreement of every current K4 mean/covariance slice does not determine
either the all-distinct K3 birth or the next matrix-core contraction. This is
an index identity, not a numerical test or a claim about real challenge MLPs.

## 4. Remove scalar overlap in the declared coordinate metric

Let `R(I)_abcd = (delta_ab delta_cd + delta_ac delta_bd
+ delta_ad delta_bc)/3`. It has `R(I)_iiii=1` and, for distinct indices,
`R(I)_iijj=1/3`, `R(I)_iiij=0`. F21's preactivation scalar component is
`c4 * rho^2 * R(I)`, where `rho=init_scale=2` at the target shape. This is its
existing radial approximation; it is not exact transport of an arbitrary
`W`-dependent metric.

For any proposed total *output-coordinate* fourth tensor `T`, define

```
s(T) = 3 * sum_ab T_aabb / (n*(n+2))
Qout = T - s(T)*R(I).
```

Then `sum_ab Qout_aabb=0`. For `n != 2`, the scalar update implemented by F21 is
`3*(sum_i K4_i + sum_{i!=j} K22_ij)/(n*(n+2))`; this follows from its
`lap2_d4` coefficients and projection coefficient `1/(8*n*(n+2))`.
The frozen width-two projection has an explicit zero fallback instead; preserve
it and do not claim the displayed formula is its executable behavior.
Retaining the applicable update with the corrected moments retains its rule, even
though its value can change through the extension.

If an extra source is specified at preactivation rather than at output,
subtract the component **actually already included** there:
`Textra - c_extra*rho^2*R(I)`, with an explicit definition of `c_extra`.
Blindly subtracting the whole F21 `c4` from a tensor that never contained it
would remove unrelated baseline information.

For a transported residual `B^tensor4 Qout`, its Euclidean double trace need
not remain zero. That newly induced trace is not automatically duplicate
scalar information. Either retain it in the extra tensor at consumption, or
transfer it explicitly to the scalar account and subtract the same amount
from the residual. Discarding it silently is another approximation.

## 5. The exact unresolved regeneration problem

A common matrix lift has the convention

```
L_I(G)_abcd = (G_ab delta_cd + G_ac delta_bd + G_ad delta_bc
              + G_bc delta_ad + G_bd delta_ac + G_cd delta_ab)/6.
```

For `H_ab=sum_c T_abcc`, the unique lift matching this contraction is

`G = 6 H/(n+4) - 3 trace(H) I/((n+2)*(n+4))`.

For a lift multiplied by a scalar metric `rho`, divide this `G` by `rho`.
After scalar subtraction, the trace-free part is simpler, but one still needs
the **off-diagonal** `H_ab`. F21's `K4_i` and `K22_ij` determine only
`H_ii=K4_i+sum_{j!=i} K22_ij`. Its current table does not supply
`T_abcc` for three distinct coordinates. Section 3 shows why those missing
entries cannot be reconstructed uniquely from its existing consumers.

One could *define* `G_off = lambda*cov_off` or a diagonal-only reset. Such a
choice is an additional approximation, not the contraction above. The former
returns to the previous failed scalar-core implementation; the latter loses mixed nonlinear
fourth structure and has separate adverse external motivation. Neither is
presented as this note's new intervention.

Two complete families are mathematically definable but lack the required
cheap, stable realization here:

1. **Regenerate:** specify a full finite connected-diagram rule for output
   `T_abcd` (including the needed three-coordinate terms), compute its `H`,
   keep the scalar and projected matrix part, and declare what residual is
   discarded. The required connected power terms and their cheap contraction
   into `H` have not been derived on the F21 state. Fitting `lambda` cannot
   replace this missing derivation.
2. **Transport plus innovations:** choose a declared nonlinear transport
   `A=diag(w1)*B`, and define
   `Qnext = output_projection(A^tensor4 Qold + Delta_new)` with a matched
   scalar account. The actual innovation `Delta_new` must come from a stated
   output cumulant closure and use the same corrected covariance. Neither a
   raw `B` carry nor a bank initialized once supplies this innovation. Exact
   pushforward of a supplied signed law is a different, already investigated
   and much larger construction; four moments are not sufficient statistics
   for an exact ReLU pushforward.

There is also no general positivity theorem for the finite polynomial
response in section 2. Positive diagonal variances are weaker than PSD, and
PSD covariance is weaker than positivity of the full fourth-order moment
matrix. An eigenvalue repair, damping, or standardized-correction clipping
would define another intervention whose bias and cost must be specified
before testing. No such repair has been selected or retrospectively justified.

## 6. What can be priced, and what cannot

Conditional on supplied `q4/q22/q31`, the four marginal and three pair
consumer updates, reconstruction/centering, scalar subtraction and repeated
birth subtraction require `O(n^2)` arithmetic and payload. This is an
incremental consumer bill only. It excludes constructing `Q`, `J`, transporting
their factors, and feeding all new K3 sources through every later layer.

For clarity, a supplied signed CP representation
`Q=sum_{s=1}^r d_s v_s^tensor4` would give

```
q4 = (V^4) d
q22 = (V^2 diag(d)) (V^2).T
q31 = (V^3 diag(d)) V.T
P = diag(w1) V
Z = 1.5 * diag(w2) (V^2) diag(d)
J = SymAverage(Z,P,P).
```

The factor `1.5` compensates for the tower's three-term *average*, reproducing
the `1/2` sum in section 3. Scalar-overlap subtraction adds its own structured
radial tensor and feed; it cannot be dropped from a CP-only bill. The signed
bank representation also contains covariance-product negative terms, so its
positive CP part alone is not the complete cumulant.

Let `M(a,b,c)=a*c*(2*b-1)` be the conventional f32 matrix-product count,
`r_l` the supplied extra rank at layer `l`, and `h_l` the total retained rank
of its induced older K3 births. A minimal **partial** per-layer skeleton is

```
one extra-factor linear carry             M(n,n,r_l)
two nonfinal K4 slice products             2*M(n,r_l,n)
two persistent extra-K3 factor carries     2*M(n,n,h_l)
extra-K3 D21 products for a Z,P,P source   2*M(n,h_l,n)
pointwise slices, feed and consumers       O(n*(r_l+h_l)+n^2).
```

This uses the two-product repeated-slice identity for one symmetric `ZPP`
source; F21's existing blocks retain their own three-product spelling. Final
mean-only pruning removes unused pair queries and births. A complete price
must add scalar/radial negative factors, the unknown regeneration function
`F_regen(n,l,state)`, joins/recompression if present, precision stabilization,
copies/casts, kernels/synchronizations, the baseline bill and its real Strassen
tariffs. Peak storage must include co-live input/output and temporary factors,
not just `n*(r_l+h_l)`. Since `F_regen` and the rank of its induced births are
unknown, **there is no complete parameterized candidate cost yet**. Neither a
zero regeneration cost nor an invented rank is a defensible placeholder.
The known supplied-core and integrated-first-source bills cannot fill this gap.

## 7. A finite next discriminator, conditional on a new derivation

Do not launch another parameter or accuracy sweep. If a future proposal
specifies the missing closure, preregister a tiny forced-active test against
an independent index-enumerated tensor interpreter first:

- Use two layers at width three or four, explicit signed nonconstant matrices,
  the `U_(0012)` tensor above, a scalar-only tensor, zero extra tensor, and a
  diagonal independent-coordinate tensor. Force both regeneration and F21's
  target contraction path; tiny off-shape fallback is not coverage.
- Independently enumerate the derivative placements and ordinary raw-moment
  partitions. Compare all `a_p`, reconstructed `R_pq`, connected `P_pq`,
  covariance, centered K3/K4 slices, scalar overlap, full `J` and next transported
  K3 tensor. The zero-extra arm must reproduce the working F21 interpreter.
- Include mutations that reverse only one D31 orientation, omit `J`, omit its
  repeated subtraction, and subtract scalar overlap twice. Each must have a
  predeclared nonzero discriminating fixture.
- Preserve a fixed float64 algebra bar (a proposal is `1e-10` relative with a
  separately specified absolute near-zero bar) and independently register
  actual-production precision checks. This note does not enact either gate.
- Then replay the existing negative-variance n8 fixture and fixed full-depth
  finite/variance/moment-matrix checks. Retain failed states and same-state
  covariance component decomposition. Do not choose damping from the failure
  and call the original intervention a pass.

Even a pass would establish conformity only to the declared approximation,
not physical-law accuracy, positivity on every network, complete flopscope
cost, deployment headroom or score improvement. Future Phase-2 comparison
remains paired arithmetic score at `C=F`, with measured truth noise and
residual headroom against 0.4 seconds reported separately.

The justified immediate route is therefore to finish the representation and
numerical diagnosis of the existing F21 recurrence. If that route remains
blocked, use the reviewed plan's already bounded alternative derivations under
their own contracts. The accuracy branch can resume when it has a distinct
equation for regenerated off-diagonal `H` and residual treatment, rather than
another name for a previously failed source.
