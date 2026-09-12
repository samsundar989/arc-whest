# Causal signed-law reference-reset increments for later nonlinear K4 (design only)

This is a proposed truth-free **IMPORTANT** fixture, not a measured result, an
accepted closure, or a K4 estimator. The accepted full-polar split-bank test
certifies a fixed two-coordinate *artificial reweighting* on a finite law. The
test proposed here asks whether a source obtained from the actual Gaussian-input
network admits exact age accounting through a later ReLU. Its Gaussian reset is
part of the definition; the resulting births are not unique properties of a
network independent of that reference. The failed unsplit nonlinear fixture
remains failed. A LOG contract and Samuel's prediction must precede execution.

## Fixed physical law and exact first anchor

Let `F_W(x)=ReLU(Wx)` and fix

```
Z ~ N(0, S0),     S0 = [[1, 1/2], [1/2, 1]],
X = ReLU(Z),
B = [[ 0.8, -0.3], [-0.55, 1.4]],   Y = F_B(X),
D = [[-0.65, 1.1], [ 0.9, -0.4]],   T = F_D(Y).
```

Write `P1`, `P2`, `P3` for the physical laws of `X`, `Y`, `T`. The matrices
are frozen from the accepted nonlinear toy fixture; no challenge network,
label, seed block, public split, or vault is involved. The first second-raw
moment has the exact analytic value

```
M1 = E[XX^T] = [[1/2, c], [c, 1/2]],
c = 1/6 + sqrt(3)/(4*pi).
```

The identity for `c` is the bivariate standard-normal ReLU product formula
at correlation `1/2`. It supplies an independent, exact construction check.
Set `S2=B M1 B^T`, the **second raw preactivation moment**, not the centered
covariance. Its entries, with the displayed decimal matrices interpreted
exactly as rationals, are

```
S2[0,0] = 0.365 - 0.48*c,
S2[0,1] = -0.43 + 1.285*c,
S2[1,1] = 1.13125 - 1.54*c.
```

`det(B)=0.955`, so `S2` is positive definite. Independently compute
`M2=E[YY^T]` from the physical continuous law and set `S3=D M2 D^T`.
`S3` too must be positive definite or this fixed construction fails. Define
the reference probability laws

```
R1 = P1,
R2 = ReLU#N(0,S2),
R3 = ReLU#N(0,S3).
```

The zero mean and second-raw matching are intentional, fixed reference
choices. In particular, `R2` need not match `P2`'s mean or lower output
moments. Centered-covariance matching would define a different experiment.

## Causal signed measure recurrence and conditional cumulants

For a general layer with physical `P_l=F_l#P_(l-1)`, let
`S_l=E_(P_(l-1))[(W_l x)(W_l x)^T]` and
`R_l=ReLU#N(0,S_l)`. Construct at that layer, using only its current weight
and current physical/reference laws,

```
Delta_l = F_l#R_(l-1) - R_l,       l >= 2.
```

`#` denotes pushforward of a measure. Each `Delta_l` has mass zero but may
have nonzero raw moments of degrees one, two, and three. Pushforward is linear
on **measures**, so induction gives the exact identity

```
P_l = R_l + sum_(a=2..l) (F_l# ... #F_(a+1)#Delta_a).
```

For this three-layer fixture, put

```
Delta2 = P2 - R2,
Q3     = F_D#R2,
Delta3 = Q3 - R3,
O3     = F_D#Delta2 = P3 - Q3.
```

Then `P3=R3+Delta3+O3` exactly. `P3-R3` is the total physical deviation
from the layer-three Gaussian reference; `Delta3` is the new reference-reset
source and `O3` is the older deviation after the actual nonlinear map.
`Delta3` uses `S3` computed from the physical `P2`, so source attribution is
conditional on this stated reference convention. A later *whole* fourth
cumulant cannot be inserted as another age source.

For every unit-mass law `Q` and mass-zero signed measure `delta` for which
the moments exist, use the set-partition definition of the fourth cumulant.
With `m_Q(I_B)` and `m_delta(I_B)` denoting raw moments for the labelled
slots in block `B`, the exact conditional difference is

```
kappa4(Q+delta; I) - kappa4(Q; I)
  = sum_(partitions pi of four labelled slots)
      (-1)^(|pi|-1) (|pi|-1)!
      [product_(B in pi) (m_Q(I_B)+m_delta(I_B))
       - product_(B in pi) m_Q(I_B)].
```

This formula explicitly retains all first-, second-, and third-moment
leakage and their cross terms. A mass-zero signed measure has no standalone
probability cumulant `kappa4(delta)` to add. For the fixed fixture define

```
K_new       = kappa4(Q3) - kappa4(R3),
K_old|new   = kappa4(P3) - kappa4(Q3),
kappa4(P3)  = kappa4(R3) + K_new + K_old|new.
```

Each difference compares two positive unit-mass laws. Report the 16 oriented
tensor entries, the eight third-cumulant entries, and every raw degree-one
through degree-three moment difference for `Delta2`, `Delta3`, and `O3`.
Compare true third-cumulant differences with any declared derivative-truncated
K4-to-K3 feed; do not identify them.

## Independent exact tiny reference and finite-bank diagnostic

There is a one-dimensional scalar continuous reference for every needed
raw moment. Let `f` be any of `ReLU`, `F_B o ReLU`,
`F_D o F_B o ReLU`, or `F_D o ReLU`, let `S` be its applicable Gaussian
matrix (`S0`, `S2`, or `S3`), put `e(theta)=(cos theta,sin theta)`,
`q_S=e^T S^(-1)e`, and `k=a+b<=4`. Homogeneity and exact radial integration
give

```
E[f(G)_0^a f(G)_1^b]
 = [2^(k/2) Gamma(1+k/2) / (2*pi*sqrt(det S))]
   * integral_0^(2*pi)
       f(e(theta))_0^a f(e(theta))_1^b
       q_S(theta)^(-(k+2)/2) dtheta,       G ~ N(0,S).
```

Use `f=F_B o ReLU,S0` for `P2`; `f=ReLU,S2` for `R2`;
`f=F_D o ReLU,S2` for `Q3`; `f=F_D o F_B o ReLU,S0` for `P3`;
and `f=ReLU,S3` for `R3`. `M2` and hence `S3` come from the independent
`P2` degree-two integrals. Split each scalar integral at every zero of its
fixed map's intermediate preactivation rows. An adaptive scalar integration
with a reported error bound is the reference; it must not call the bank,
signed-source, or cumulant implementation being checked. Build fourth and
third cumulants separately from all set partitions of labelled slots.

For a finite diagnostic, use a positive three-node Rayleigh moment rule
exact through radial degree five and angular Gauss-Legendre orders 32, 48,
64 on every sign sector. Keep coincident output atoms indexed. Generate
positive banks separately for the five laws, then form the signed
differences on the disjoint union of those indexed supports. Never rescale
mass after construction. The finite bank may split at `D` to resolve a
later ReLU boundary, as in the accepted repair, but that is a **future-aware
toy quadrature**, not a deployable online source. An exact tiny causal
representation instead keeps `Delta2` as two signed symbolic
angle/radius kernels (`P2` and `R2`) without inspecting `D`; when `D` becomes
available at layer three, split those kernels at its new roots. Finite-rank
compression of those kernels is a separate unsolved question.

## Suggested frozen gates for a later LOG contract

The parent must register exact gates before any calculation. Proposed checks:

1. The computed `M1` and `S2` agree with the analytic anchor within `1e-12`;
   independently integrated `M2` and `S3` agree within `1e-10`.
   `S2,S3` are positive definite; all five positive laws have positive
   unrescaled mass-one banks, positive radial nodes and weights, Rayleigh
   degree-0..5 defects at most `1e-12`, and fixed boundary roots agree with
   an independent root finder within `1e-12`.
2. For all 15 two-variable monomials of total degree at most four, each
   positive law's order-32/48, order-48/64, and order-48/continuous gaps
   satisfy `|difference|/max(1,|reference|) <= 1e-8`. Each adaptive scalar
   integral reports an absolute error bound below `1e-10`.
3. On the same finite bank, the measure identities for `Delta2`, `Delta3`,
   `O3` and `P3=R3+Delta3+O3` hold for all raw moments through degree four
   within `1e-10`. Independent direct atom enumeration and set-partition
   cumulants match every dense K3/K4 conditional contrast within `1e-10`.
   The conditional K4 telescoping identity has the same gate.
4. The old nonlinear discriminator
   `G=||K_old|new - D^(tensor 4)[kappa4(P2)-kappa4(R2)]||_max`
   exceeds `1e-6` absolute and `1e-3` times
   `||kappa4(P2)-kappa4(R2)||_max`; otherwise this chosen fixture fails to
   discriminate nonlinear carry. Replacing only `F_D` by the signed linear
   map `x -> D x` makes the analogous gap at most `1e-10`.
5. To claim lower-moment leakage matters here, require at least one
   degree-one through degree-three raw moment of `O3` to exceed `1e-8`
   absolute, and require
   `||K_old|new - [m4(P3)-m4(Q3)]||_max > 1e-6`, where `m4` is the dense
   fourth raw-moment tensor. A failure of either check preserves the
   exact identities but defeats this fixture's lower-leakage discriminator.
   Zeroing `Delta2` or `Delta3` in its corresponding conditional contrast
   must give zero within `1e-10`.

These are deterministic algebra/numerical gates, not a paired score test and
not a Phase-2 promotion gate. A tiny CPU-only fixture should be run first;
its raw moments, source hashes, failures, and all gate values belong in an
evidence packet and need a fresh independent Skeptic before belief.

The construction proves, if those gates pass, only exact conditional
signed-law accounting for this fixed physical two-coordinate cascade and
its Gaussian-reset convention. It does not establish a K4-only nonlinear
closure: the causal symbolic kernels retain the entire distribution, and
fourth-order moments do not determine their future ReLU response. It gives
no width-1024 source algorithm, compact rank, residual bound, counted FLOPs,
8 GiB proof, score improvement, or submission candidate. The second-raw
matrix needed for each reset is exact in this tiny oracle; estimating it
cheaply and causally in a deployed estimator is separate work.
