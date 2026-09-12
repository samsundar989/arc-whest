# K4 source repair architectures (truth-free design memo, 2026-09-12)

Scope: mathematical design following the accepted source-identity defect at LOG
commit `2164401`. The frozen `arm_coherent.py` and its supplied-core dense oracle
are read-only references. This memo is neither a scored arm, a promotion, nor an
independent Skeptic verdict. Here `K` always means a *fourth cumulant*, not a
fourth central moment; `B` is an arbitrary signed linear transport (including
any explicitly chosen diagonal ReLU response), and `b_i` is its row.

## 1. Retain an exact independent-source diagonal component

For independent post-ReLU coordinates `Y_a`, define
`k_a = E[(Y_a-EY_a)^4] - 3 Var(Y_a)^2`. Their fourth tensor is exactly
`K_abcd = k_a 1[a=b=c=d]`. Store it as `(Q,k)`, initially `Q=I`; under `B`,
`Q <- B Q` and `K_ijkl = sum_a k_a Q_ia Q_ja Q_ka Q_la`. This remains an exact
component under any signed, nonorthogonal sequence of **linear** transports.
Its required consumers, with no diagonal masking in the identities, are

```
D4_i   = sum_a k_a Q_ia^4
D22_ij = sum_a k_a Q_ia^2 Q_ja^2
D31_ij = sum_a k_a Q_ia^3 Q_ja       (D31_ji is separately oriented)
```

Thus `D22 = (Q^2 diag(k))(Q^2)^T` is symmetric and
`D31 = (Q^3 diag(k))Q^T` generally is not. At `Q=I`, off-diagonal D22 and
D31 vanish exactly. For an equal-magnitude He row, `D4=4k/n`, including at
`n=1024`; this is the invariant the frozen source fails. For Gaussian
preactivation, `K=0`, so its K4 consumers and K4-induced K3 feed are zero even
when the Gaussian coordinates are correlated. ReLU of correlated Gaussian
coordinates can subsequently create *mixed* fourth cumulants; the diagonal
component alone makes no claim to represent them. In a general source split,
`K = K_diagonal + R_mixed`, the residual must be named rather than replaced by
fabricated off-diagonal values.

At the same derivative order as the frozen mean consumer, the K4 mean term is
`w4_i D4_i/24`; it must consume this D4, not a central fourth moment.

The K4 part of the off-diagonal output covariance in the current derivative
truncation has the symmetric orientation

```
Delta C_ij = (w2_i w2_j/4) D22_ij
           + (w3_i w1_j/6) D31_ij
           + (w1_i w3_j/6) D31_ji,       i != j.
```

It reduces to its transpose for any signed `Q`, unequal derivatives and
nonconstant `k`. Merely symmetrizing the frozen expression after calculation
would not establish these contractions. Its diagonal output variance must be
handled by the same declared approximation order, not inferred from this
off-diagonal formula.

K4-induced K3 must use the **same** tensor. A definition that avoids borrowing
the frozen cubic polynomial is the first-order cumulant response: with Gaussian
reference `Z`, output map `f`, and `L_a=Q_:a dot grad_Z`, take the cubic
coefficient in `t` of
`sum_a (k_a/24) E[L_a^4 exp(t dot f(Z))]/E[exp(t dot f(Z))]`.
This is an identity for the first-order K4 perturbation, not a cheap evaluation
algorithm. If the chosen local closure keeps only one quadratic and two linear
response factors, its *direct* fourth-cumulant feed is

```
B_ijk = 1/2 sum_a k_a [
  w2_i w1_j w1_k Q_ia^2 Q_ja Q_ka
 +w1_i w2_j w1_k Q_ia Q_ja^2 Q_ka
 +w1_i w1_j w2_k Q_ia Q_ja Q_ka^2 ].
B_iij = w2_i w1_i w1_j D31_ij + (w1_i^2 w2_j/2) D22_ij;
B_iii = (3/2) w2_i w1_i^2 D4_i.
```

These last formulas specify that truncation only; higher response/covariance
terms require derivation from the generating expression. In particular, the
frozen `birth3_dense` oracle proves its *defined cubic polynomial* for a
supplied core, not this physical K4-to-K3 source identity. Keep Gaussian
nonlinearity's own K3 birth separate: `K4=0` does not imply total output K3=0.
The three consumer slices alone do not determine `B_ijk` for three distinct
indices. The diagonal representation does: set `A_ia=w2_i Q_ia^2` and
`P_ia=w1_i Q_ia`; its direct birth is
`(1/2) sum_a k_a Sym_3(A_:a,P_:a,P_:a)`, with `Sym_3` here the sum of the
three placements of `A`. Each future linear map transports both `A` and `P`.

At `n=1024`, a float32 `Q` is 4 MiB. Conditionally retaining one rank-`n`
component at each of 16 layers (without claiming later sources independent)
costs at least 64 MiB for 16 `Q` matrices;
transporting all ages takes up to 120 dense `B@Q` products, about 258 billion
conventional multiply-add FLOPs. Computing both dense D22 and D31 takes two
more matrix products per active source, up to about 516 billion additional
FLOPs over those 120 age-layer uses, plus O(n^2) D4 and factor construction.
If each direct K3 birth is retained for all later layers, its two factor
matrices add up to 128 MiB and their two transports per age-layer add up to
about 516 billion more conventional FLOPs. Peak live array payload then
includes the 64 MiB K4 source bank, up to 128 MiB of K3 birth factors, up to
another bank while replacing them, and several 4 MiB `n x n` products/scratch
arrays. These are
incremental algebra estimates, not a flopscope ledger, process-RAM bound or
0.4-second residual proof. Retaining only the first independent source is much
cheaper but does not close later mixed sources or nonlinear propagation.

## 2. Transport a metric core, with a visible residual

Declare `T(G,M)_abcd = [G_ab M_cd + G_ac M_bd + G_ad M_bc +
G_bc M_ad + G_bd M_ac + G_cd M_ab]/6`, with symmetric `G` and positive
semidefinite metric `M`. The frozen supplied-core convention is `M=2I` and
`G=diag(g)+lambda C_off`. Under **linear** `B`, the represented tensor
transports exactly as `G'=B G B^T`, `M'=B M B^T`. Both legs are necessary:
`B=tI` makes `T` scale as `t^4`. Replacing `off(G')` with
`lambda off(B C B^T)` or resetting `M'=2I` is generally false, including for
a nonconstant diagonal `G` and a signed rotation. The slices at the consumer
coordinate are

```
D4_i   = G_ii M_ii
D22_ij = (G_ii M_jj + G_jj M_ii + 4 G_ij M_ij)/6
D31_ij = (G_ii M_ij + G_ij M_ii)/2.
```

These hold for arbitrary signed `B`, nonconstant and off-diagonal `G`, and
zero `G`. The same symmetric covariance rule and generating-function K3 rule
above apply with `K=T(G,M)`; the direct quadratic-response K3 feed is obtained
by substituting this `K` into the three permutations of `K_iijk`. A zero core
must make every *core-attributed* consumer and feed zero, regardless of `M` or
Gaussian covariance. Subsequent K3 birth transport needs an explicitly
factorized cubic representation derived from `T(G,M)`; the frozen birth tower
cannot be assumed to equal the direct cumulant response. One unfused exact
factorization of the stated quadratic-response feed uses
`S(v,H)_ijk=v_i H_jk+v_j H_ik+v_k H_ij` and the sum of six placements
`S6(A_:a,P_:a,R_:a)`:

```
B = [S(w2*diag(G), diag(w1) M diag(w1))
   + S(w2*diag(M), diag(w1) G diag(w1))]/12
   + sum_a S6(A_:a,P_:a,R_:a)/6,
A=diag(w2), P=diag(w1) G, R=diag(w1) M.
```

This follows by substituting
`K_iijk=(G_ii M_jk+M_ii G_jk+2G_ij M_ik+2M_ij G_ik)/6`
in the direct feed. It stores three factor matrices, two quadratic matrices
and two vectors per birth. Future transport sends each factor through `B`,
each quadratic matrix through a dense sandwich and each vector through `B`.
At `n=1024`, that is about 20 MiB per birth, up to 320 MiB for 16 births,
with a possible second bank during transport. There are seven dense matrix
products per active birth-layer, about 15.03 billion conventional FLOPs, or
about 1.80 trillion for 120 age-layer uses, before cubic slice readout and
any fusion.

Reproducible tiny algebra check (no labels or estimator execution): set
`G=[[1,.2,-.4],[.2,-.3,.7],[-.4,.7,2]]`,
`M=[[2,.1,.3],[.1,1.3,-.2],[.3,-.2,.8]]`,
`w1=[.2,.8,-.5]`, `w2=[1.2,-.1,.4]`. In float64, enumerate all 81
`K_abcd` directly from the six-term definition of `T(G,M)`, then all 27
`B_ijk` from the three-placement direct feed in section 1. Independently
evaluate the `S/S6` factorization above and compare every entry with
`abs(direct-factorized) < 1e-14`. The direct/factorized pairs for indices
`(0,0,1)`, `(0,1,2)`, `(2,2,2)` are respectively
`(.04774,.047740000000000005)`,
`(-.04504499999999999,-.04504499999999999)`, and
`(.24,.24000000000000005)`; the maximum absolute difference over all 27
entries in this evaluation was `5.551115123125783e-17`. This checks the
factorization identity only, not a ReLU source or the frozen birth tower.

Source insertion and projection are approximations. For a candidate source
`S`, define `E=S-T(G,M)` and retain either `E` itself or its measured slice
errors. With invertible `M`, metric contraction
`H_ij=sum_ab (M^-1)_ab S_ijab` gives the unique contraction-matching core
`G=6H/(n+4)-3 tr(M^-1 H) M/((n+2)(n+4))`. For singular `M`, the same formula
holds only on its rank-`r` support with `M^+` and `n` replaced by `r`; a
component outside that support remains residual. This matches a contraction,
**not** all fourth-tensor slices. For any later row map `B`, the exact omitted
consumer errors are `E[b_i,b_i,b_i,b_i]`,
`E[b_i,b_i,b_j,b_j]`, and `E[b_i,b_i,b_i,b_j]`. For example, a multilinear
operator-norm bound gives at most `||E|| ||b_i||^4` for D4 and
`||E|| ||b_i||^2 ||b_j||^2` for D22, and
`||E|| ||b_i||^3 ||b_j||` for D31. The direct quadratic K3 feed inherits
these slice errors multiplied by its stated derivative factors; a small
contracted `H` alone gives no such bound. Projecting an independent diagonal
source into one core therefore does not guarantee its exact dense-row D4 value,
even when inverse-width scaling returns. Distinct source ages start with
distinct metrics after transport; merging them into one `(G,M)` is another
projection unless their metrics actually agree.
For example, with `M=2I` and independent equal diagonal cumulants `k`, this
projection gives `G=3k I/[2(n+2)]`. A He row of squared norm 2 then gives
`D4=12k/(n+2)` with both legs transported, versus the exact diagonal-source
`4k/n`. Holding the metric fixed would instead give `6k/(n+2)`; neither is
the exact diagonal tensor.

One float32 `(G,M)` pair at `n=1024` stores 8 MiB. Two dense sandwiches per
layer (four matrix products) cost about 8.59 billion conventional FLOPs,
around 137 billion over 16 layers; O(n^2) slice construction then needs
several additional 4 MiB arrays. Keeping 16 age-specific pairs instead stores
128 MiB and can demand about 1.03 trillion sandwich FLOPs over 120 age-layer
uses, with up to a second bank during transport. Metric
inversion/eigendecomposition for projection is an
additional O(n^3) operation and may be poorly conditioned. No listed number
includes the existing K3 backbone, cubic birth transport/readout (priced
separately above), flopscope operation billing, nonlinear source derivation,
allocator overhead, or server residual.

## Smallest source correction and next truth-free gate

Build a separate tiny reference for `W0` with orthogonal rows, so L1 ReLU
coordinates really are independent. Compute their **cumulants** from exact
Gaussian ReLU moments, insert `Q=I` and no mixed source, transport by an
arbitrary signed/nonconstant `W1`, and compare all D4/D22/D31 entries with an
independently enumerated dense fourth tensor. Check the oriented covariance
formula and the K4-induced K3 response (first the stated quadratic truncation)
against an independent dense derivative expansion. Include correlated Gaussian
preactivation with `K4=0`, zero core, permutations, scaling `B=tI`, and a
nonconstant off-diagonal `G`/nonorthogonal `B` metric-core fixture. This repairs
the first demonstrable source identity *as an explicitly separate component*;
it does not silently turn a diagonal source into the frozen projected-core
hypothesis. Do not use the P07-fitted `lambda=1/3` as evidence for either new
representation. A general mixed-ReLU source, response-order choice, feasible
compression, counted cost, 8 GiB behavior, server residual headroom and any
accuracy benefit remain unknown. They need separate contracts and a fresh
Skeptic before a labeled decision.
