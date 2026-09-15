# P1b does not exist for this representation — the feed is already there

*Checked 2026-09-14 against F21's own frozen Wick table. Label-free: no truth,
no network minted, no block spent.*

## The question

`docs/CLAUDE_CODE_HANDOFF_2026-09-14.md` §2.2 lists a κ4→κ3 feed as part of P1:

> κ4→κ3 feed `X3 = A·d(λw1) + P·d(w1²dG)` (E14's `_feed_slices` carries only the
> first term — add the second; both O(n²) per lineage)

and `docs/GAME_PLAN.md` has insisted since 2026-09-10 that a κ4 closure must
derive **both its consumers and the third-order births it induces from one
state**, because *"use-only is not the proposed replacement"* — that was the
lesson of the failed T1 deletion, which removed births while keeping consumers
derived for a different algorithm.

So: does `estimator_k3_v30.py` have the births, or only the consumers?

## The answer: it has both, and it always did

F21's frozen term table already routes the fourth-order slices into the
third-order ones. Enumerated directly from `_tables()["terms"]`:

| birth | terms | source slices consumed | of which fourth-order |
|---|---|---|---|
| `K[(3,)]` | 3 | `(3,)`, **`(4,)`** | **1** — blocks `[(4,)]`, scalar `1/24`, Wick `[[4, 3]]` |
| `K[(2,1)]` | 5 | `(1,1)`, `(1,2)`, `(2,1)`, **`(2,2)`** | **1** — blocks `[(2,2)]`, scalar `1/4`, Wick `[[2,2],[2,1]]` |

And the chain's tower birth is

    S3  = K[(3,)]  - p111_3
    S21 = K[(2,1)] - p111_21
    out_tw = _append(adv, diag(S3) + 3 * S21.T, fac1, w2b_vec, n)

so `S[(4,)]` reaches `S3` and `S[(2,2)]` reaches `S21`, every layer.

**v30 makes `S[(4,)]` and `S[(2,2)]` per-neuron.** The induced κ3 births
therefore become per-neuron automatically, *from the same `dG`* that the
consumers read. One state, both uses. That is exactly the coherence GAME_PLAN
demanded, obtained for free because the term table was always doing the routing.

The `1/24` in the table is the same `1/24` the handoff writes in
`pK[(p,)] += q4·w(4,p)/24`, which is a small corroboration that the two
descriptions are of the same object.

## Why the handoff asks for `X3` anyway

That term belongs to **504aldo's compact representation**, where the tower is
carried as explicit `A` and `P` legs and the frozen term table is not used at
all. In that architecture the feed has to be written by hand onto the legs,
because nothing else will do it. F21 keeps the term table, so it gets the feed
from the table.

It is the same physics reached two ways, and porting `X3` onto F21 would
**double count** it.

## What this changes

- **P1b is struck from the plan for this candidate.** There is no second
  mechanism to build; `estimator_k3_v30.py` is P1 complete.
- If a future candidate abandons the term table for the compact leg
  representation — which is what P5 eventually implies — the feed comes back as
  real work, and this note is why.

## What it does not establish

That the births are *correct*, only that they exist and are driven by the same
state. Whether the per-neuron core improves the score is the design screen's
question, not this one. And the `X3` mapping onto F21's legs was **not**
derived here: E14's `_feed_slices` turned out to be a different mechanism
altogether (its tree-core all-age `U` query over a carried `Y`/`R` state), so
no mapping was attempted and none is claimed.
