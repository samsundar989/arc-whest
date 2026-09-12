# Weighted-discrete K4: exact covariance feed and all-history cost

Truth-free design memo, 2026-09-12. This prices one **uncompressed, exact-for-the-chosen-discrete-measure** algorithm. It is neither a Skeptic verdict nor a measured flopscope ledger, scored arm, rank-adequacy finding, or deployment claim. The accepted signed-cubature fixture establishes the n=2 weighted-discrete tensor/consumer identity. The response truncation and covariance convention below are those in `repair_architecture.md`; they do not constitute a complete ReLU closure.

## Source and fourth-order consumers

Let `v_s=V_:s` be centered columns with normalized weights `p_s`. The polynomial identities also permit signed weights, though a probability cubature uses nonnegative ones. Set `C=V diag(p)V^T`, `c=diag(C)`, and `T(C,C)=(C_ij C_kl+C_ik C_jl+C_il C_jk)/3`. Then

```
K = sum_s p_s v_s^{otimes 4} - 3 T(C,C),
K^-_ijkl = -(C_ij C_kl+C_ik C_jl+C_il C_jk).
```

The negative term is an exact fourth-order metric core with two identical rank-at-most-`S` covariance legs. It need not be expanded into a dense fourth tensor or `S(S+1)/2` quartic columns. Under any signed linear map `B`, update `V <- BV`; its same-measure covariance becomes `C'=(BV)diag(p)(BV)^T=BCB^T`, so the entire discrete cumulant transports as `B^{otimes 4}K`. Keep source ages separate: `sum_age T(C_age,C_age)` differs from `T(sum_age C_age,sum_age C_age)` by false cross-age terms.

At a consumer coordinate, with elementwise powers,

```
D4_i   = sum_s p_s V_is^4 - 3 c_i^2,
D22_ij = sum_s p_s V_is^2 V_js^2 - c_i c_j - 2 C_ij^2,
D31_ij = sum_s p_s V_is^3 V_js - 3 c_i C_ij.
```

`D22` is symmetric; `D31` generally is not. The stated off-diagonal output covariance contribution is `(w2_i w2_j/4)D22_ij + (w3_i w1_j/6)D31_ij + (w1_i w3_j/6)D31_ji`, for `i != j`. The stated mean term is `w4_i D4_i/24`. A diagonal output-variance update requires its own declared response order; the off-diagonal formula does not determine it.

## Exact negative K4-to-K3 factorization

The **declared quadratic-response** feed, with one quadratic and two linear response legs, is

```
F_ijk = (1/2)[w2_i w1_j w1_k K_iijk
            + w1_i w2_j w1_k K_ijjk
            + w1_i w1_j w2_k K_ijkk].
```

At birth, put `P=diag(w1)V`, `H=P diag(p)P^T=diag(w1)C diag(w1)`, `a=w2*c`, `A=diag(w2)`, and `Q=diag(w1)C=P diag(p)V^T`. Write `A_:h,Q_:h` for their columns, `Sym3(u,v,v)_ijk=u_i v_j v_k+v_i u_j v_k+v_i v_j u_k`, and `S(a,H)_ijk=a_iH_jk+a_jH_ik+a_kH_ij`. Substituting `K^-_iijk=-(c_i C_jk+2C_ij C_ik)` gives the **exact** identity

```
F^- = -(1/2) S(a,H) - sum_{h=1}^n Sym3(A_:h,Q_:h,Q_:h).
```

For comparison, expanding `C` into particles yields an exact `S(S+1)/2` pair-column representation with `w2*(v_s*v_t)`. The displayed `n`-column representation is smaller when `S` is moderate or large and makes its paired covariance structure explicit. It is exact for the stated feed, not for higher response derivatives. The positive CP part is `(1/2)sum_s p_s Sym3(A^+_s,P_s,P_s)`, with `A^+_s=w2*(v_s*v_s)=A(V_:s^2)`. Thus the whole feed shares `P` between positive and negative terms and can derive `A^+` from `A` and the birth-time `V`.

At every later signed linear map `B`, carry `A <- BA`, `P <- BP`, and `a <- Ba`. Keep the **birth-time** `V` and `p` as coefficient data. Reconstruct `Q=P diag(p)V_birth^T`, `H=P diag(p)P^T`, and `A^+=A(V_birth^2)` at readout. These equal the separately transported birth factors because matrix multiplication is associative; no later derivative is reapplied to an old birth. The same `A` is created once per nonlinear **birth layer** because `w2` is shared by all K4 source ages consumed there. Their `P,a,V_birth,p` remain age-specific. Neither replacing `B(v_s*v_t)` by `(Bv_s)*(Bv_t)` nor replacing `sum_age Q_age^2` by `(sum_age Q_age)^2` is valid.

The negative cubic readouts are

```
D3^-_i   = -(3/2)a_i H_ii - 3 sum_h A_ih Q_ih^2,
D21^-_ij = -(1/2)[a_j H_ii+2a_i H_ij]
           - [(Q^2) A^T]_ij - 2[(A*Q) Q^T]_ij,
```

where `*` and `Q^2` are elementwise. The positive CP readouts are

```
D3^+_i   = (3/2)sum_s p_s A^+_is P_is^2,
D21^+_ij = (1/2)[(P^2 diag(p))(A^+)^T]_ij
           + [(A^+*P diag(p))P^T]_ij.
```

These retain the three-distinct-index cubic tensor through their factors; carrying present `D3/D21` alone would lose future information. A full `D21` readout uses two `n x n` products for the negative column term, one weighted `P` Gram for `H`, one `P`-by-birth-`V` product for `Q`, one `A`-by-`V_birth^2` product for `A^+`, and two width-`S` products for the positive term. At a given readout layer, ages born at the **same** layer have the same transported `A`: their `Q_age^2` matrices can be summed before one product with `A^T`. The `(A*Q_age)Q_age^T` products still require one per age in this algorithm. A final mean-only `D3` needs `Q,A^+` but not the full `H` Gram or any dense `D21` product. Its `H_ii` is a weighted row-square reduction.

This exact sharing is within a birth layer and with the positive CP feed. Different nonlinear birth layers have different `A`; different source ages have different `Q,H,P`. An equivalent already-existing transported dense response factor might allow further reuse, but this memo charges it explicitly. A shared-coordinate sketch/QR, truncating `Q`/`A`, dropping the negative feed, or summing age covariances before products is an approximation or a distinct algorithm requiring its own error and cost account. No lower bound is asserted against other tensor networks.

## Schedule and conventional arithmetic

Take width `n=1024`, depth `d=16`, float32 factors, and hypothetically one rank-`S` additive discrete K4 source born after each nonlinear layer. Only the first 15 sources can affect a later layer. Index layers `0..15`: at layer `l`, `l` prior K4 sources are consumed. That is `sum_{l=1}^{15}l=120` K4 transport/consumer uses: 105 at nonfinal layers require full `D22/D31`, while the final 15 need `D4` only. The 105 nonfinal K4 uses create 105 age-specific K3 feed terms across 14 nonlinear birth layers. Their future uses number `sum_{l=1}^{14}l(15-l)=560`: 455 nonfinal full `D21` readouts and 105 final `D3` readouts. The 14 shared `A` birth-layer matrices undergo `sum_{l=1}^{14}(15-l)=105` future transports. This all-age additive source schedule is a **conditional architecture**, not a result of the n=2 fixture: later non-Gaussian source construction and possible cross-source cumulants are undefined.

Charge an `m x k` by `k x q` product as `2mkq` conventional multiply/add FLOPs; one float32 `n x n` array is 4 MiB. The algorithm carries `V` for each K4 age, rebuilds each full same-measure `C` at nonfinal uses, uses two positive-CP slice Grams, transports shared `A` and age-specific `P,a`, and reconstructs `Q,A^+` at each K3 readout. At the final mean-only layer, `c=diag(C)` is computed in `O(nS)` without a full covariance Gram. A hypothetical source-generation line charges one `n x n` map on `n x S` nodes per source; it does **not** price the distributional model, sampling or quadrature needed to obtain those nodes.

| Dense-product family | Exact count/formula | S=64 | S=256 | S=1,024 |
|---|---:|---:|---:|---:|
| 15 hypothetical source maps | `15*2n^2S` | 0.002 T | 0.008 T | 0.032 T |
| K4 `V` transport | `120*2n^2S` | 0.016 T | 0.064 T | 0.258 T |
| Same-measure `C` Grams, nonfinal | `105*2n^2S` | 0.014 T | 0.056 T | 0.225 T |
| Positive CP `D22,D31` Grams | `105*4n^2S` | 0.028 T | 0.113 T | 0.451 T |
| Shared K3 `A` transport | `105*2n^3` | 0.225 T | 0.225 T | 0.225 T |
| Age-specific K3 `P,a` transport | `560*(2n^2S+2n^2)` | 0.076 T | 0.302 T | 1.204 T |
| K3 `Q` reconstruction | `560*2n^2S` | 0.075 T | 0.301 T | 1.203 T |
| K3 `A^+` reconstruction | `560*2n^2S` | 0.075 T | 0.301 T | 1.203 T |
| K3 full `H` Grams | `455*2n^2S` | 0.061 T | 0.244 T | 0.977 T |
| K3 positive `D21` products | `455*4n^2S` | 0.122 T | 0.489 T | 1.954 T |
| K3 negative `D21` products, shared `A` | `(455+91)*2n^3` | 1.173 T | 1.173 T | 1.173 T |
| **Dense-product subtotal** | sum of unrounded lines | **1.868 T** | **3.276 T** | **8.905 T** |

`T` denotes `10^12` conventional FLOPs, not a tensor. Source centering, nonlinear map/derivative work, factor formation, `D4/D3`, negative-core elementwise slices, age accumulation (including the `Q_age^2` sums before shared-`A` products), casts, allocation and Python/array calls are additional. These matter to actual billing and residual even where dense arithmetic dominates. With only one first-layer source, there are 15 K4 uses, 14 useful K3 births, 105 K3 future uses, 91 full `D21` readouts and 105 shared-`A` transport uses; the corresponding dense-product subtotals are about **0.703/0.963/2.004 T** at `S=64/256/1024`. The earlier `mixed_source/cost_options.md` CP-only subtotal excludes this negative feed and full birth tower.

At the final readout, the exact persistent payload in this layout is 14 shared `A` matrices (`56 MiB`), 105 `P` matrices and 105 birth-time `V` coefficient matrices (`210*4nS` bytes), 105 `a` vectors (0.410 MiB), and 15 K4 `V` matrices (`15*4nS` bytes): **112.66/281.41/956.41 MiB** for `S=64/256/1024`. Normalized weights add negligible `O(105S)` payload. A streaming array plan reserves, in addition, `10n^2+6nS` float32 slots (40 MiB plus 1.5/6/24 MiB) for `Q,H,A^+`, square/Hadamard inputs, covariance and CP slices, two dense-product outputs and the accumulated `D21`. It releases each source/birth scratch before advancing. Immutable transport needs at least one old/new factor overlap (4 MiB for `A`, `4nS` bytes for `P`). Thus even this plan has a rough live array allowance of at least **159/333/1,029 MiB**, rounded upward from the persistent bank, listed scratch and both overlaps; product libraries and expression temporaries can raise it. Building an entire replacement bank before release can nearly double persistent payload. These are array-payload plans, not process-RAM bounds; the source generator and existing K3 backbone can dominate memory.

There is no QR/sketch join in this exact bank: source ages and K3 birth terms remain separate. Concatenating columns may copy the bank, while compressing them must price basis construction, Gram/QR, rebasing, reconstruction, residual storage and approximation error. This ledger also excludes the ordinary K3 backbone, Gaussian-nonlinearity K3 births, actual ReLU source integration/sampling, later non-Gaussian source closure, official-package operation billing, the server FLOP offset, 8 GiB lifecycle behavior and 0.4 s residual headroom. No score or runtime inference follows from these conventional counts. Before treating the factorization as an implementation result, preregister a separate truth-free executable factor/readout check, price the running microkernel and seek a fresh independent Skeptic verdict.
