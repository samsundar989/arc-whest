# First correlated-Gaussian ReLU K4 source: transport and cost options

Truth-free design/cost memo, 2026-09-12. This is neither an independent Skeptic verdict nor a proposed scored arm. It uses the accepted selected linear-tensor oracle in `source_dense/` and the residual counterexample in `repair_architecture.md`. The current n=2 correlated-source quadrature contract in `experiments/LOG.md` is the next source-identity test; none of the options below is established by that test yet.

## The object that has to be represented

Let `Z ~ N(0,C)`, `Y_i=ReLU(Z_i)`, and `X_i=Y_i-E[Y_i]` at the *first* correlated Gaussian preactivation. For all indices, including repeats, the source is

```
K_ijkl = E[X_i X_j X_k X_l]
       - Cov(X_i,X_j) Cov(X_k,X_l)
       - Cov(X_i,X_k) Cov(X_j,X_l)
       - Cov(X_i,X_l) Cov(X_j,X_k).
```

Each raw moment in this formula is a Gaussian positive-orthant integral of at most four distinct variables, determined by its submatrix of `C`. The classes `K_iiij`, `K_iijj`, `K_iijk`, and `K_ijkl` (four distinct indices in the last class) must all be considered. The first two are obtainable from bivariate marginals; they do not determine the latter two. At zero correlation, independent-coordinate mixed cumulants vanish; at nonzero correlation they generally need not. The accepted independent-source `(Q,k)` result therefore proves an exact *component*, not this source.

For a later signed linear map `B`, exact transport is `K' = B^{\otimes 4} K`. A stored source coordinate matrix `Q` updates as `Q <- B Q`. Every claimed compressed representation needs to reproduce the future `D4_i=K_iiii`, `D22_ij=K_iijj`, `D31_ij=K_iiij`, **and** the declared K4-induced K3 birth, including three-distinct-index entries needed by later transport. Current slices alone do not suffice: the accepted all-distinct residual has zero current slices and future D4 `1.5`. These are linear transport statements; later ReLU layers create new, generally non-Gaussian sources and require their own closure.

## Transportable choices

| Source representation | What is exact | Missing approximation or bottleneck |
|---|---|---|
| Implicit Gaussian integral: retain `C` and source-coordinate `Q`; evaluate any `K_ijkl` from its 1–4D marginal integral | Exact specification of the first ReLU-Gaussian source and exact transport by replacing each queried future row with `Q^T b` | Compact storage does not make the contractions cheap. Streaming each future `D22_ij` independently sums four source indices. A full tensor materialization is far over the memory cap. No cheap K3 birth follows merely from storing `C`. |
| Full symmetric tensor | Exact source if all integrals are evaluated; exact four-mode linear transport, slices and direct K3 feed | 4 TiB unsymmetrized float32, or 171.67 GiB even with ideal symmetric packing at n=1024. This is a tiny-reference control only. |
| Signed rank-`R` quartic factors `sum_r a_r q_r^{\otimes4}`, or a declared tensor network | Exact linear transport of the **chosen** tensor (`q_r <- B q_r`) | A finite fit/projection of the true source is approximate. No useful bound on `R` follows from n=2 pair tests or present-slice agreement; generic ranks/bonds may erase savings. Residual must be transported or bounded in a norm that controls future rows. |
| Weighted cubature/particle cumulant: centered columns `V_:s`, normalized weights `sum_s p_s=1`, `C_p=V diag(p)V^T`, `K_p=sum_s p_s V_:s^{\otimes4}-3T(C_p,C_p)` | This is the **exact fourth cumulant of the discrete weighted measure** and remains exact for that measure after `V <- B V`, `C_p <- B C_p B^T`. It includes its own mixed cumulants, including all-distinct terms. | Gaussian quadrature or Monte Carlo with finite `S` is an approximation to the true ReLU-Gaussian source. The covariance subtraction must use the same weighted measure; inserting the analytic covariance would no longer give its cumulant. Sampling produces noisy mixed terms even at true independence. The negative metric-core term and its K3 birth have real costs. |
| Pairwise-only source coefficients plus transported `Q` | Exact linear transport of the tensor **after** `K_iijk` and all-distinct `K_ijkl` are set to zero | Source truncation is unproved. Even `K_iijj` contributes crossed products to all-pair D22 after dense transport; an `O(n^2)` coefficient bank does not imply `O(n^2)` readout. The hidden-residual example forbids certifying this choice from current slices. |
| A projected `T(G,M)` plus explicit `E=K-T(G,M)` | Both legs of `T` transport exactly (`G,M <- B(G,M)B^T`), as does a carried `E` | Projecting alone loses tensor information. The metric-contraction projection in `repair_architecture.md` matches one contraction, not all future rows. If `E` is dropped, this is an approximation; if `E` is dense, the original memory problem returns. Merging age-specific metrics is another projection unless they coincide. |
| Finite Hermite/diagram expansion in the Gaussian correlation matrix | A chosen polynomial truncation has a well-defined tensor and can be transported by its source-coordinate map | ReLU has an infinite Hermite series. The degree cutoff and induced K3 terms are approximations. Explicit degree-2 chaos has `n * binom(r+1,2)` coefficients for latent rank `r`; at `n=r=1024`, float32 is about 2.00 GiB **per source** before scratch or history. Implicit graph formulas might lower storage, but their all-pair readout ledger has not been derived. |

The cubature identity uses `T(C,C)=(C_ij C_kl+C_ik C_jl+C_il C_jk)/3` under the memo's six-term `T` convention. The `-3T(C_p,C_p)` term is essential: a raw fourth moment is not a cumulant. Its core-attributed off-diagonal covariance must use both orientations of D31, as in `repair_architecture.md`.

## Conventional arithmetic and payload ledger, n=1024, depth=16

These are **not** flopscope bills or wall-time predictions. Count a conventional dense `n x n` matrix product as `2n^3=2,147,483,648` multiply/add FLOPs, a `B @ V` with `V` width `S` as `2n^2S`, and a float32 `n x n` array as 4 MiB. One L1 source has at most 15 subsequent transports; 16 separate age sources have up to `15+...+0=120` active age-layer uses. The latter is a scaling illustration, **not** a claim that later preactivations are Gaussian or that the same source construction is valid there. Source construction also matters: one fully generic source has 523,776 distinct pairs, 178,433,024 distinct triples and 45,545,029,376 distinct quadruples, before repeated-index patterns and quadrature nodes per marginal. This is why an implicit exact Gaussian-integral description is small in bytes but not presently an executable width-1024 source algorithm.

| Work | One L1 source, 15 uses | 16 age sources, 120 uses | Payload / qualification |
|---|---:|---:|---|
| Full dense tensor, four mode products per use (`8n^5`) | `1.3511e17` FLOPs | `1.0809e18` FLOPs | 4 TiB per source without symmetry; already infeasible. |
| One `(Q,k)` rank-n diagonal source: one `B@Q` per use | 32.21 billion | 257.70 billion | 4 MiB per source, 64 MiB for 16 Qs. Applies to the independent component only. |
| Its two dense slice Grams, D22 and D31 | 64.42 billion | 515.40 billion | Additional to transport; its direct K3 birth factors add two more `B@factor` products per use, 64.42/515.40 billion. |
| One age-specific `(G,M)` pair: two sandwiches, four products per use | 128.85 billion | 1,030.79 billion | 8 MiB per source, 128 MiB for 16 pairs; excludes source construction, residual and K3 feed. |
| Unfused metric-core quadratic K3 birth from `repair_architecture.md`: seven products per use | 225.49 billion | 1,803.89 billion | About 20 MiB per birth, 320 MiB for 16; optimization is possible but unproved. |

For the weighted cubature factors, the following *unfused illustrative* loop charges, per active use, one `B@V` (`2n^2S`), rebuilding the **same-measure** covariance `C_p` as a weighted Gram (`2n^2S`), two CP slice Grams for D22/D31 (`4n^2S`), and two CP K3-birth-factor transports (`4n^2S`). This totals `12n^2S` per use for one carried birth factor set per source age. D4 and pointwise factors add `O(nS)`/`O(n^2S)` work depending on materialization; source sampling, Gaussian mapping, the negative `3T(C_p,C_p)` slices and K3 birth, ordinary K3 backbone, joins, casts and scratch are **excluded**. If each nonlinear step creates additional K4-induced K3 births that persist, their separate age-layer uses must be added; the table is not an all-history tower bill. The CP K3 terms are only the declared one-quadratic/two-linear truncation, not a full ReLU response.

| `S` | CP-only four-item subtotal, one source | CP-only subtotal, 120 uses | 16 stored `V` matrices | Interpretation |
|---:|---:|---:|---:|---|
| 64 | 12.08 billion | 96.64 billion | 4 MiB | Cubature/source error unknown. |
| 256 | 48.32 billion | 386.55 billion | 16 MiB | Still excludes the negative core's K3 tower. |
| 1024 | 193.27 billion | 1,546.19 billion | 64 MiB | Storage looks modest while arithmetic exceeds the existing compact ledger. |

At `S=256`, generating one first-layer Gaussian sample matrix via `W0 @ X` adds about `0.537` billion conventional FLOPs before ReLU and reductions; generating all 16 Gaussian surrogates would not be justified without a separate model. Rebuilding `C_p` from `V` may be much cheaper than two full-rank sandwiches at small `S`, but a *valid* low-rank factorization and transport/readout of the negative core's K3 birth is still required. Using the unfused seven-product birth above would add 225.49 billion for one age or 1.804 trillion for 120 uses; one must either pay this, derive a cheaper exact factorization for `C_p`'s rank, or explicitly label a dropped feed. D22/D31 source coefficients built only from bivariate integrations have `O(n^2)` storage but may require `O(n^4)` crossed readout without a proven factorization. Their 523,776 distinct pairs per 1024-wide source also make numerical source integration a priced step.

For comparison, the accepted *supplied-core* compact reference measured `818,494,496,296` flopscope FLOPs over 10,052 operations and about 1,014 MiB observed process RAM; the current exact arm measured 1,079 MiB peak. None includes this corrected mixed-source pipeline. Phase 2 charges counted `C=F` while residual must stay below its 0.4 s failure cliff. The official server's `+2^24` FLOP offset and unknown residual for a new candidate prevent converting this arithmetic table into a score or deployment claim. Even a small float32 factor bank must be measured with temporary arrays and complete lifecycle under the 8 GiB limit.

## Smallest implementation discriminator

Finish the already registered n=2 `rho=0` / `rho=0.5` Gaussian-ReLU quadrature contract first. Its independent raw-moment cumulant reconstruction and signed future-row D4 check establish whether `K_1112`/`K_1122` are material and whether diagonal-only omission is visible; they do **not** test three- or four-distinct mixed entries. Its **positive-quadrant** grid integrates moments with positive powers in both coordinates; it omits the probability mass mapping to either ReLU axis and the origin. It therefore cannot be reused as `(V,p,C_p)` for a discrete ReLU measure.

The smallest representation follow-up is a separate tiny **full signed Gaussian** product cubature, with every node mapped through ReLU and its full weight retained, including nodes that land on an axis or the origin. Build `(V,p,C_p)` from those rectified nodes; compute all n=2 K4 entries from `sum p V^4 - 3T(C_p,C_p)` and compare with an independently coded raw-moment cumulant of *that same discrete measure*. Then compare signed two-map transport, all three slice families, oriented covariance and the declared quadratic K3 feed with its separate dense tensor. Use the registered higher-order positive-quadrant integration only as a distinct approximation-to-Gaussian reference, with its own convergence check. This separates a transport/factor identity from source quadrature accuracy.

An n=4 correlated Gaussian fixture with `K_iijk` and all-distinct `K_ijkl` is a **later separately priced gate** before complete mixed-source coverage. A full product grid at 64 or 96 nodes per dimension has `64^4=16,777,216` or `96^4=84,934,656` nodes; it must be streamed, checked for convergence against an independent method, and queued if its RAM/CPU demand is substantial. The n=2 identity and factor checks do not make that four-variable integration cheap or validate it. No protected labels, seed block or scored estimator is needed for any of these steps.

The next feasibility gate after tiny algebra is to implement one selected factor/core path as a *label-free* n=1024 x 16 microkernel with an operation-family ledger, all source generations/ages, source construction, direct and K3 readouts, scratch peak, and official-package subprocess lifecycle. Compare that exact executing ledger with the existing compact baseline before any accuracy contract. A tiny algebra pass does not authorize a new protected read, and this memo makes no rank/error or 0.4-second residual claim.
