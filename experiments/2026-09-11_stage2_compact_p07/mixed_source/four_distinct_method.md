# Truth-free n=4 correlated-ReLU fourth-source reference: design only

2026-09-12. This memo prices a future independent source calculation; it records no new cumulant values, experiment outcome, score, gate change, seed-block use, or Skeptic verdict. The accepted n=2 Gaussian-pair source and linear transport are in `evidence.md` and `skeptic_verdict.md`; `cost_options.md` identifies `K_iijk` and all-distinct `K_ijkl` as the missing source classes. The already registered signed-cubature n=2 contract in `experiments/LOG.md` is separate. Before executing the n=4 calculation, register its own fixed numerical gate and resource request in LOG and obtain the required human prediction. A numerical fixture is not a competition promotion decision.

## Fixed Gaussian fixture and quantities

Take four independent standard normals `U,e0,e1,e2,e3`, fix

```
a = (0.4, 0.5, 0.6, 0.7)
Z_i = a_i U + sqrt(1-a_i^2) e_i
R = Cov(Z) =
    [[1,    .20, .24, .28],
     [.20, 1,   .30, .35],
     [.24, .30, 1,   .42],
     [.28, .35, .42, 1  ]].
Y_i = max(0,Z_i).
```

The distinct positive off-diagonal entries avoid exchangeability while keeping the quadrature moderate. This matrix is provably positive definite: `R=diag(1-a_i^2)+aa^T`, hence `v^T R v >= 0.51 ||v||^2` for every `v`. There is no random draw or challenge network. Record `K_0012` and all-distinct `K_0123` at minimum; preferably enumerate all 12 `K_iijk` classes (`i` repeated, `j<k` different from `i`) and the one all-distinct class. Index order is zero-based. Repeated-index orientation and all 24 permutations of `K_0123` should agree; do not infer the four-distinct class from pairwise entries.

For any index multiset `I=(i,j,k,l)`, let `m(A)=E[prod_{r in A} Y_r]`, with `m(empty)=1`, retaining repeated indices as powers. Compute the fourth cumulant independently by the set-partition identity

```
K_I = sum_{partitions pi of {0,1,2,3}}
      (-1)^(|pi|-1) (|pi|-1)! prod_{block B in pi} m(I_B).
```

As a separate algebra check on the *same* raw moments, form `X_i=Y_i-m_i` and evaluate `E[X_i X_j X_k X_l] - C_ij C_kl - C_ik C_jl - C_il C_jk`. Keep both reconstruction codes independent. These formulas include covariance subtraction; the fourth raw or central moment alone is not a cumulant.

## Low-RAM reference: one-dimensional latent integral

Conditional on `U=u`, the four `Z_i` are independent normals with mean `b_i=a_i u` and standard deviation `s_i=sqrt(1-a_i^2)`. Thus every raw moment with total degree at most four is just one scalar integral:

```
m_p = integral_{-infinity}^{infinity} phi(u)
      prod_{i:p_i>0} h_{p_i}(b_i,s_i) du,
h_q(b,s) = E[max(0,N(b,s^2))^q].
```

Use `h_0=1` when an exponent is absent. For positive powers, set `t=b/s`, `q_0=Phi(t)`, `q_1=b Phi(t)+s phi(t)`, and `q_r=b q_{r-1}+(r-1)s^2 q_{r-2}` for `r=2,3,4`; then `h_r=q_r`. The distinction between absent exponent `h_0=1` and the recurrence's positive-half-line mass `q_0=Phi(t)` matters. Compute `Phi(t)` stably through `erfc(-t/sqrt(2))/2` and evaluate raw moments using adaptive Gauss-Kronrod integration on split intervals `(-infinity,0)` and `(0,infinity)` with absolute and relative error requests around `1e-13`. Integrate all `binom(4+4,4)=70` multiindices with total degree `0..4`, though only a subset is needed for the selected cumulants. A scalar or small fixed-vector integrand needs kilobytes, and the full moment table has 70 doubles. This conditional-independence reduction is exact for the fixed Gaussian fixture, not a closure for later non-Gaussian layers.

As an internal convergence check, repeat the one-dimensional reference with a different integration scheme (for example 48/72/96-point Gauss-Legendre on each of `[0,2]`, `[2,6]`, `[6,12]` and their reflected negative intervals) and tighter adaptive tolerances. Do not treat agreement between two cumulant formulas fed the same moments as an independent integration check.

## Independent direct marginal-density cross-check

Independently integrate raw moments over the **positive orthant of the coordinates whose exponents are nonzero**. For support `S={i:p_i>0}`, use its `d x d` principal submatrix `R_S` and the marginal Gaussian density

```
m_p = integral_[0,infinity)^d (prod_{i in S} x_i^p_i)
      exp(-x^T R_S^-1 x/2) / ((2*pi)^(d/2) sqrt(det R_S)) dx.
```

This is a continuous raw-moment integral. Integrating a zero-power coordinate over its positive half alone would wrongly discard ReLU axis/origin mass. The marginal-density route integrates `K_0012`'s highest-order raw moment over three dimensions and `K_0123`'s over four; all lower-order moments use their own one-, two-, or three-dimensional marginals. The matrix inverse/determinant and quadrature are independent of the latent-factor integrand. Use Gauss-Legendre nodes on `[0,12]`, initially orders 32, 48, 64 *per active dimension*. Build one-dimensional nodes and weights once, and stream the product into fixed-size tiles, evaluating the quadratic form from a precomputed inverse. No `meshgrid` or full `N^4` tensor. Keep float64 accumulators, preferably compensated or pairwise summed over tiles. At order 64, the all-distinct highest raw moment has `64^4=16,777,216` nodes; the 12 three-distinct `K_iijk` highest moments together have at most `12*64^3=3,145,728` nodes. Lower moments and lower orders add work but do not change the fourth-power dominant term. The 64-node result must agree with order 48 within the separately registered numeric tolerance; otherwise report nonconvergence and price order 96 (`96^4=84,934,656` nodes) through workerq before using the value.

The `[0,12]` truncation has a conservative distributional bound. For any raw product of total degree at most four, Cauchy-Schwarz and Holder give `E[prod |Z_i|^p_i 1{max_i Z_i>12}] <= sqrt(105*4*Phi(-12))`, about `9e-16`, since each standardized marginal has eighth absolute moment 105. This bound is deliberately loose and covers all four coordinates, including a lower-dimensional marginal. Check `[0,10]` versus `[0,12]` as a diagnostic; use the analytic bound, not that difference alone, to justify the tail. Propagate each raw-moment quadrature error through the partition polynomial using interval bounds or a conservative sum of absolute term sensitivities. Cumulants can involve cancellation, so report absolute errors alongside relative errors only when the term is materially nonzero. Cross-check the direct marginal raw moments and both cumulant reconstructions against the independent one-dimensional latent reference. If they disagree, preserve both results and diagnose quadrature or algebra before claiming a source identity.

For a separate linear-consumer identity, use normalized rows `b_plus=(1,1,1,1)/2` and `b_minus=(1,1,1,-1)/2`. Compute the fourth cumulant of `b·Y` both by contracting the complete `4^4` tensor and by reconstructing the scalar's first four raw moments from the raw-moment table. Holding all other tensor entries fixed, the all-distinct entry contributes exactly `+1.5 K_0123` to the first contraction and `-1.5 K_0123` to the second, because it has 24 permutations. This checks indexing and signed transport but does not replace the independent density integration.

## Resource and interpretation limits

The streamed order-64 four-dimensional check evaluates about 16.8 million four-variable densities; selected three-dimensional checks add about 3.1 million, plus lower-order grids. A scalar density uses roughly 50-150 conventional arithmetic operations and one exponential per node, so this is order `1e9` to `3e9` arithmetic operations plus about 20 million exponentials, excluding quadrature setup and repeated convergence orders. This is a planning estimate, **not** a measured flopscope ledger or local wall-time claim. Budget 0.5-2 CPU-hours for the multi-order Python/NumPy implementation and benchmark the actual first queued run; 96^4 is about 5.1 times the 64^4 dominant node count. Stream at most 65,536 four-dimensional points per tile: each float64 working vector is 0.5 MiB, so even twenty concurrent vectors are 10 MiB; target under 128 MiB array payload and under 1 GiB total process RAM. The 4^4 output tensor is only 2 KiB. Submit any substantial direct check through `workerq` with a conservative `--ram 2 --cpus 2 --vram 0` declaration; inspect measured peak and revise upward if needed. These estimates do not certify the challenge's 8 GiB memory cap, counted FLOPs, 0.4 s residual headroom, or a width-1024 representation.

The reference establishes, if it later passes its registered numerical checks, only the first-layer correlated-Gaussian ReLU fourth source for this fixed four-variable fixture. It does not show that `K_iijk` or `K_ijkl` can be stored or transported cheaply at width 1024, that a finite cubature rank is adequate, or that the resulting estimator improves score. No protected truth, seed registry, frozen arm, shipped estimator, or existing gate is involved in this design.
