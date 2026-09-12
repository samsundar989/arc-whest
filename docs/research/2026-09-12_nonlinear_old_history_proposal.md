# Three-layer, two-source nonlinear old-history fixture (proposal)

This is a truth-free design for a future IMPORTANT contract, not a measured
result, a source closure, or a score claim. No fixture has been run for this
proposal. The accepted exact-axis fixture proves a two-layer signed tensor
identity with a **linear** old carry. The question here is whether a later
ReLU map can respond to an older signed law in a way that transporting its
fourth cumulant by a fourth tensor power misses. The test has three activation
layers and two deliberately separated source ages; it does not read labels,
seed blocks, the estimator, the public split, or the vault.

## Fixed laws and source construction

Use the accepted positive order-48 `rho=1/2` exact-axis bank as the law
`P1` of `X=ReLU(Z)`, with 439 indexed atoms. Freeze

```
B = [[ 0.8, -0.3], [-0.55, 1.4]]
D = [[-0.65, 1.1], [ 0.9, -0.4]]
R_B(x) = ReLU(B x),       R_D(y) = ReLU(D y).
```

Thus `Y=R_B(X)` and `T=R_D(Y)` are the second and third activation layers.
Keep coincident mapped particles indexed until after source construction, so
their positive weights and signed differences remain unambiguous.

Make a fourth-order old perturbation by reweighting `P1` on its own support.
For a positive atom law `p`, let the rows of `M` be the ten two-variable
monomials of total degree at most three, each divided by its `p`-weighted
root-mean-square on the indexed particles. This deterministic scaling
changes the conditioning but not the moment-null space. Set

```
g1(x) = x1*x2^3 / (1 + (x1^2+x2^2)^2)
A = diag(sqrt(p)) M^T
h = diag(sqrt(p)) (I - Q Q^T) diag(sqrt(p)) g1
u = h / max_i |h_i|/p_i,   R1 = P1 + 0.2*u,
O1 = P1 - R1 = -0.2*u,
```

where the columns of `Q` are an orthonormal basis of `A`, computed by a
rank-revealing QR or SVD with a **fixed** relative rank cutoff `1e-12`. The
same formula for `h` may be implemented by a stable projection without
forming `I-QQ^T`. It gives `M u=0`, including mass, so `R1` is a positive
unit-mass reference whose first three raw moments equal `P1`'s. The quotient
in `g1` limits the influence of remote radial nodes while retaining a fixed
quartic directional contrast. No feature or sign is to be selected after
seeing an output.

Let `V=R_B#R1`, `O2=R_B#O1`, and `P2=R_B#P1=V+O2`. On the indexed particles
of `V`, use the same projection with
`g2(y)=y1^2*y2^2/(1+(y1^2+y2^2)^2)` and its positive weights to obtain
`u2`. Define `R2=V+0.2*u2` and `N2=V-R2=-0.2*u2`. This is a *new* second-layer
fourth-order source relative to `V`, with zero raw moments through degree
three at insertion. The two sources are not two copies of one age: `O1`
crosses both nonlinear maps, while `N2` crosses only the last.

The fixed construction is invalid, and the fixture records FAIL without
choosing another contrast, if either weighted degree-three moment matrix has
rank below ten, relative condition number above `1e12`, zero projected
contrast, `max|M u|>1e-11` after normalization, or any reference weight is
nonpositive. The normalized perturbation obeys `|0.2*u_i|<=0.2*p_i`; hence
`R1` and `R2` are positive by construction. The four second-layer laws below
are also positive: on each original indexed particle `R2>=0.8 R1>=0.64 P1`
and `O2>=-0.2 P1`, giving `Q10>=0.44 P1` before any particle merge. Each has
unit mass because both signed sources have zero mass.

## Four counterfactuals and the nonlinear discriminator

At layer two define

```
Q00 = R2                 (neither source)
Q10 = R2 + O2            (old only)
Q01 = R2 + N2 = V        (new only)
Q11 = R2 + O2 + N2 = P2  (both, the original positive law).
```

Apply the *nonlinear* `R_D` separately to all four. At both layers enumerate
their raw moments through degree four and form `K_abcd=cum4(...)` and
`J_abc=cum3(...)` from the general set-partition formula, including repeated
indices. At layer three use `K_ij=cum4(R_D#Qij)` and analogously for `J`.
Then define the conditional source accounting at each layer:

```
new       = K01 - K00
old|new   = K11 - K01
old|none  = K10 - K00
interaction = K11 - K10 - K01 + K00.
```

The same contrasts apply to `J`, all K4 consumer slices, and the declared
K4-to-K3 feed. In particular, `old|new = old|none + interaction`; assigning
the old source last is a stated bookkeeping convention, not an independence
claim. Reversing the order must move precisely the interaction.

The main nonlinear old-history discriminator at layer three is

```
G3 = max_abcd |(K11-K01)_abcd
              - (D^tensor4 (K(Q11)-K(Q01)))_abcd|.
```

Here `K(Qij)` is measured **before** `R_D`. For a truly linear final map,
this gap is exactly zero for any input laws. A nonzero `G3` isolates the
effect of the final ReLU on the older source while the younger source is
present. Also record the analogous layer-two gap between
`cum4(R_B#P1)-cum4(R_B#R1)` and
`B^tensor4(cum4(P1)-cum4(R1))`. Neither gap is an error of tensor algebra:
fourth cumulants transport linearly through a linear map, while `ReLU` must
act on the underlying law. The older perturbation has zero lower moments at
layer one, but after `R_B` it may change moments of orders one through three;
enumerate this **lower-moment leakage** explicitly at layers two and three.
Consequently this fixture tests a conditional signed-**law** response, not a
closed K4-only state. The exact source also contains higher-than-fourth
moment changes at insertion, which a K4 tensor alone cannot encode.

## Independent dense oracle and consumers

Keep the source construction and the checking paths separate. The checker
enumerates each atom's monomials directly, forms cumulants by all set
partitions of three and four labelled slots, and sums the four
counterfactuals independently. It must not call a compact K4 factor,
source-slice, signed-transport, or feed helper from the implementation under
test. Compare every one of the 16 dense K4 entries and eight dense K3 entries
at both layers, not just diagonals. Separately run the accepted order-64
positive bank as a numerical convergence control for the *unreweighted*
full law `P2/P3`; reweighting is defined on the fixed order-48 bank, so a
different bank is not an independent oracle for its artificial sources.

For the full positive `P2/P3` law only, compute a separate continuous
reference from the original correlated Gaussian. Split polar angle at every
zero of the preactivations of `Z`, `B ReLU(Z)`, and
`D ReLU(B ReLU(Z))`. On each sector the nested biasless ReLU map is linear
in radius. Integrate radius analytically: a homogeneous output monomial of
degree `m` has radial factor
`2^(m/2)*Gamma((m+2)/2)*q(theta)^(-(m+2)/2)`, where
`q(theta)=e(theta)^T Sigma^(-1)e(theta)`; integrate the remaining smooth
one-dimensional angle factor independently at fixed scalar tolerances.
This direct-moment oracle does not use the axis-bank moment rule or signed
source constructor. It checks the physical three-layer positive law, not
the counterfactual attribution convention.

For each dense K4 contrast and the full tensor, check its required `D4_i`,
`D22_ij`, and **both oriented** `D31_ij` and `D31_ji`. Freeze the declared
local response coefficients
`w1=(0.7,-0.4)`, `w2=(1.1,0.3)`, `w3=(-0.2,0.9)`,
`w4=(0.6,-0.5)`. Check `w4_i D4_i/24`, the off-diagonal covariance term

```
(w2_i*w2_j/4)*D22_ij
 +(w3_i*w1_j/6)*D31_ij +(w1_i*w3_j/6)*D31_ji,
```

and all eight entries of the *declared quadratic* K4-to-K3 feed

```
F_ijk = [w2_i*w1_j*w1_k*K_iijk
        +w1_i*w2_j*w1_k*K_ijjk
        +w1_i*w1_j*w2_k*K_ijkk]/2.
```

Enumerate the feed from the full dense K4 tensor as the reference. It is a
specified derivative truncation, not the entire true K3 change after ReLU;
report the separately enumerated true `J` contrasts rather than equating
them to `F`. Diagonal output variance needs its own declared expansion and
is outside this covariance identity.

## Proposed gates and controls to freeze in the LOG before execution

1. Both projections pass the fixed rank/condition/positivity checks above.
   All ten moments through degree three agree at insertion within `1e-11`.
   If this fails, do not repair the law after observing the output.
2. Independent atom enumeration agrees with every raw moment, K3, K4,
   source contrast, consumer, oriented covariance term, and declared feed
   within `1e-10` absolute on the same order-48 bank. Inclusion-exclusion
   and both source-assignment orders satisfy the same threshold.
3. Order-48 and order-64 unreweighted positive-law raw moments through
   degree four at layers two and three, and order-48 versus the direct
   continuous reference, agree to `1e-8` material-relative error, with
   denominator `max(1, |reference|)`. Direct scalar integration's reported
   absolute error bound must be below `1e-10` per checked moment.
4. The old-history `G3` above exceeds `1e-6` absolute **and** exceeds
   `1e-3` times `max_abcd |(K(Q11)-K(Q01))_abcd|`; otherwise this chosen law
   failed to discriminate nonlinear carry. Record the layer-two gap and the
   lower-moment leakage, whether zero or not. The mixed term is a diagnostic:
   do not claim nonadditive age interaction unless its absolute maximum
   exceeds `1e-10` and direct enumeration agrees.
5. Zeroing either source collapses its conditional contrasts to zero.
   Replacing `R_D` by the signed linear map `D` collapses `G3` to at most
   `1e-10`; a deliberately linear-carry mutant must fail gate 4 when the
   real nonlinear fixture passes. Swapping `D31_ij` and `D31_ji` in only one
   covariance term must be detected by the dense reference on the frozen
   unequal coefficients. Preserve any failed gate verbatim.

These are proposed gates, not retroactive thresholds. The parent must put
the final exact contract in `experiments/LOG.md` and commit it before a
fixture runs. No decision seed block or paired score statistic is involved:
all laws are fixed synthetic analytic or deterministic discrete laws.

The sample is two-dimensional with at most 583 positive-bank particles and
four counterfactuals, direct moments through degree four, and small dense
tensors. It is a tiny CPU-only calculation with a proposed process-memory
ceiling of 64 MiB for the fixture proper. No width-1024 compression,
flopscope ledger, 8 GiB proof, 0.4-second server residual headroom, score,
or physical full-network K4 closure follows. In particular, retaining a
signed atomic law exactly is the oracle mechanism and is not a feasible
wide-network implementation by itself.
