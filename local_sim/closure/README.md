# local_sim/closure — ReLU moment closure, generated symbolically

An independent derivation of the ReLU nonlinear-step moment closure, written as exact
rationals so that other implementations can be checked against it. It was derived without
reading `external/504aldo/**`, `estimator*.py`, or `k3_coefs_simple.json`.

## Rule

- Operator: `E[F(y)] = exp(sum_v kappa_v/v! d^v) E_0[F(y)]`. `E_0` is taken under
  independent Gaussians `N(mu_i, s_i)`, where `s_i` is the full variance. Each monomial gives
  `coef * prod(kappa) * w(k_i,p)_i [* w(k_j,q)_j]`, where
  `coef = prod 1/(mult! (v!)^mult)`.
- Available blocks: `C=(1,1)`, `D21=(2,1)`, `D12=(1,2)`, `D22=(2,2)`, `D3i=(3,0)`, `D3j`,
  `D4i=(4,0)`, `D4j`.
  **Assumed zero:** `kappa_iiij (3,1)`, `kappa_ijjj (1,3)`, and every cumulant of order 5 or higher.
- Outputs:
  - `pK(p) = E[ReLU^p]` for p = 1..4.
  - `pK(a,b) = Cov(ReLU_i^a, ReLU_j^b)` for (1,1), (2,1), (1,2) and (2,2). Both orientations
    are written out.
  - Only connected monomials (at least one mixed block) appear, because
    `Cov = exp(L_i+L_j)(exp(L_mix)-1) E_0[FG]`. The derivation is in the docstring of
    `generate.py`.
- Truncation:
  - `BUDGET B` keeps a monomial when `sum cost(v) <= B`, with
    `cost(v) = sum(v) - 1 - [all entries even]`. This is the same power counting as upstream
    `vec_block_cost`; `BUDGET B` equals upstream `k_max = B+1`.
  - `PAIR` keeps every monomial with at most 2 factors.

| rule | pK(1..4) each | pK(1,1),(2,1),(1,2),(2,2) each | max derivative |
|---|---|---|---|
| BUDGET 2 | 3 | 5 | 4 |
| BUDGET 3 | 3 | 13 | 5 |
| BUDGET 4 | 6 | 39 | 8 |
| PAIR | 6 | 30 (BUDGET 4 minus the 9 three- and four-factor C-power terms) | 8 |

- `w(k,p) = sigma^(p-k) (P_phi(alpha) phi + P_Phi(alpha) Phi)`, with integer polynomials, is in
  `w_forms.json`:
  - For k <= p it comes from partial Gaussian moments.
  - For k > p it is `p! (-1)^m He_m(alpha) phi sigma^-(m+1)` with `m = k-p-1`.
- The conversion from `pK` to the joint cumulants `K` of the post-ReLU outputs is in each rule
  file under `conversion`. It was verified to hold exactly.

## Run

```
uv run python local_sim/closure/generate.py                 # writes closure_{budget2,budget3,budget4,pair}.json, w_forms.json (deterministic)
.venv-gpu\Scripts\python.exe local_sim/closure/oracles.py   # ~20 s; needs mpmath (torch optional); writes oracle_report.json
```

## What the oracles prove

- The `w` forms match direct quadrature, derivatives of the density, `d/dmu` consistency, and
  upstream `relu_wick_coef`.
- The `pK->K` conversion holds exactly.
- The operator coefficients and the connected projection are exact for polynomial `F` (in
  `Fraction` arithmetic), up to degree 8 and (4,4).
- The term sets and coefficients equal upstream `get_all_terms` once it is filtered to the
  available blocks.
- Swapping i and j gives the same tables.
- Asymptotic orders on specific realizable families (see below).
- Float32 behaviour of the normalised spelling `W = sigma^(k-p) w` and `kappa/sigma^order`.

## What they do NOT prove

- They say nothing about accuracy on actual MLP pre-activations.
- Matching finitely many cumulants does not identify a law. The slope checks cover the stated
  families only.
- On a strongly non-Gaussian fixture (skew 2.7, excess kurtosis 13):
  - The closure is poor at every rule: relative errors are 2%–260%.
  - A higher BUDGET can be worse. BUDGET 4 gives a mean error of 260%, against 35% at BUDGET 2.
- Because the cost-3 and cost-4 cumulants `kappa_(3,1)` and `kappa_5` are dropped, BUDGET 3 and
  4 cannot improve the order past the scaling of those cumulants. On family B this was
  measured: pK(1,1) stays at order 3 and the mean at order 4. With all blocks the orders rise
  to 3.9/5.0 and 6.
- With a fixed O(1) correlation, the power series in C leaves a floor that does not go to zero
  as the non-Gaussianity goes to zero.
