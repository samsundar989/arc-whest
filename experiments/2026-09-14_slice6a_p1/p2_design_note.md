# P2, the 7→4 identity: the target, priced and pinned

*2026-09-15. Design note for the implementation, not a result. Label-free.*

## The identity, confirmed three ways

Our own derivation, our own measurement on F21 state, and the oracle all agree:

    e     = w(1,2) − 2·pK1·w(1,1)
    S_sep = zero_diag( e_i · C_off[i,j] · w1_j )
    S21   = S_sep + R_res
    A_D   = diag(S3) + 3·S21ᵀ = diag(S3) + 3·a_b·d(e) + 3·R_res ᵀ

The middle term is **free**: `a_b = fac1 = w1[:,None]·C11` is a leg F21 already
carries, so `3·S_sepᵀ` is a column scale on it. That is the whole mechanism.

Oracle check: `external/504aldo/estimators/estimator_v16b.py:558-569` has
`e_b = W_all[:, _i12] − 2·pk1v·w1` and
`S_sep = _zero_diag(e_b[:,None] · C_off · w1[None,:])`, and states the birth as
`M_b = diag(S3c) + 3 S21ᵀ = diag(S3c) + 3 a_b diag(e_b) + 3 Rr Lrᵀ`. Identical
to what we derived independently before cloning.

## The rank, and what it costs in accuracy

**They ship rank 32, not 16.** Their own F63 row, `estimator_v16b.py:280`:

| rank | their measured raw-MSE cost vs a dense `M` |
|---:|---|
| 16 | +1.6% |
| **32** | **+0.9%** |
| 64 | +0.4% |

The handoff's "rank 16" is their *cheapest* row, not their shipped choice.

Our own Frobenius measurement at `n=128` (`p2_split_n128.json`) found rank-16
leaves **1.7–7.7%** of `‖S21‖` unrepresented — an order of magnitude more than
their +1.6% end-to-end MSE cost. That is the expected direction: a contraction
forgives Frobenius error. It also means **our Frobenius screen cannot set the
rank**; only an end-to-end score can, and theirs already did.

## The price, modelled

Per layer with `k = l+1` lineages, F21 spends 3 dense transports and 4 dense
`(2,1)` contractions. P2 drops **one transport** (`A_D`) and **one contraction**
(`(M∘M)@A_Dᵀ`), and adds a thin rank-`r` leg.

| | modelled total | saving | share of the tower |
|---|---|---|---|
| F21 tower | 0.92946 B | — | — |
| P2 rank 16 | 0.66830 B | 0.26117 B | 28.1% |
| **P2 rank 32** | **0.67269 B** | **0.25677 B** | **27.6%** |
| P2 rank 64 | 0.68148 B | 0.24799 B | 26.7% |

**Read the RELATIVE figure, not the absolute.** This model prices plain matmuls
and F21 runs Strassen depth 4, so it overprices F21's tower by about 2.2×
(0.929 B modelled against 0.422 B measured). Strassen applies to both arms, so
the ~27.6% share is the transferable number; the absolute is not.

**Applied to F21's measured 0.422 B**, rank 32 projects to roughly **0.305 B** —
a 1.38× cost improvement at +0.9% raw. Score would move about
`1.009 × 0.723 ≈ 0.73`, i.e. `1.596e-8 → ~1.16e-8`.

**That is real and it is not enough on its own.** 504aldo sit at 5.4e-9. P2 is
one of four cost mechanisms, and the note exists so the next implementation
starts from a number rather than a hope.

## What the implementation has to touch

`_Tower` gains `Z`/`L` residual legs; `_advance` transports `Z` instead of
`A_D`; `_dslice3` and `_dslice21` read the reconstructed `M = P·d(s) + 3·A·d(e)
+ Z·Lᵀ`; the birth emits `(a_b, 3·Bᵀ, Q, S3c, e_b)` instead of a dense `A_D`.
Their `_dslices` at `estimator_v16b.py:583-620` is the shape to match, and the
factor 3 on the residual is folded into `Z` at birth.

The rank compression is a **randomized range finder with one power iteration**,
which is `fnp`-expressible. It introduces randomness, so it must derive from
`mlp.seed` — hard rule 2 — and their "no randomness" README wording is theirs,
not ours.

## Not established here

That P2 preserves *our* score. Their +0.9% is measured on their chain, not
F21's, and the two have different term sets. Our own end-to-end screen decides
it, and the rank is theirs to suggest and ours to verify.
