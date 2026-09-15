# The missing `q31` consumer is a term-table extension, not a missing line

*2026-09-15. Read against the oracle, `external/504aldo/estimators/estimator_v17.py`.
Label-free; nothing run.*

## Why this matters

P1 delivers **1.2371×** where 504aldo report **1.97×** from the same mechanism.
Half their gain is missing, and the two candidates for where it went were the
absent `q31` consumer and λ. The λ sweep is measuring the second. This is the
first.

## What `q31` actually is

Their line 452:

    wk431 = C_off * (0.5 * METRIC_C * lam_prev)

is a third consumer of the κ4 core — an `(n,n)` slice, the `(3,1)` one. It is
not consumed at one site. It feeds **four** term rows (lines 247–250):

| destination | Wick pair | scalar |
|---|---|---|
| `K[(1,1)]` | `(1,1)` × `(3,1)` | 1/3 |
| `K[(2,1)]` | `(1,2)` × `(3,1)` | 1/6 |
| `K[(2,1)]` | `(3,2)` × `(1,1)` | 1/6 |
| `K[(2,2)]` | `(1,2)` × `(3,2)` | 1/3 |

Their own header calls it *"the (3,1) slice … feeding 4 extra n² terms"*.

## Why it cannot simply be added to F21

F21's term table is **upstream's SIMPLE `k_max=3` enumeration**, frozen with a
pinned sha256 (`k3_terms_simple.json`,
`824358c01b891ce15d3d45755746d5459181abd303ebf9e9ad56af807c75b2a3`) and
re-checkable against the upstream checkout by `p2_k3_fnp.py --check-terms`.
That checkability is the whole reason the port was bounded: *"an open-ended
re-derivation in a restricted dialect"* became *"implement 25 named
contractions"*.

**The `(3,1)` slice is not in that enumeration.** It belongs to the augmented
closure, and 504aldo append these four rows to their own `TERM_SPECS` after
building the base set. So adding them to F21 means the table is no longer
upstream's SIMPLE list, and `--check-terms` would either fail or have to be
taught that four rows are ours.

That is a **deliberate divergence from the pinned reference**, and this
repository has been careful about exactly that kind of divergence since the
width-2 projection hole. It needs its own contract, its own statement of what
the extended table is checked *against* — upstream cannot check it, because
upstream does not contain it — and its own tiny-shape parity evidence.

## Cost

Their header says the κ4→κ3 feed costs **zero extra n³**, and the `q31` terms
are `n²`. So this is cheap in FLOPs. The expense is entirely in provenance:
a frozen, externally-checkable table stops being frozen and externally
checkable.

## What this changes about the "missing half"

It stops being an oversight and becomes a **scoped piece of work with a known
shape**: four rows, one new slice, `O(n²)`, and a contract about diverging from
the pinned enumeration. Whether it recovers the gap is untested; their ablations
are on their chain, not F21's.

**Order of operations.** The λ sweep decides first. If λ is carrying most of the
gain — and the first arm says the bare core is only 1.070× of the 1.237× — then
fitting λ end to end is the cheaper lever and should come before a table
divergence.
