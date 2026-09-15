# P2 repriced: 32.3% of the tower

> **CORRECTED AGAIN, 2026-09-15, and this is the one to use.** The 41.5% below
> assumed F21 spends **seven** dense products per layer. It spends **six**:
> `_dslice21` already fuses its two `M`-right-factor terms into one matmul, so
> the baseline is 3 transports + 3 contractions, not 3 + 4. P2 gives
> 2 + 2, so the count is **6→4** and the saving is **32.3%**, giving
> `C/B 0.4222 → 0.2959`.
>
> I have now priced this three times — 27.6%, 41.5%, 32.3% — and only the last
> was read off the code. The first two were read off the plan and the mechanism
> name. **Count the products in the source before quoting a saving.**
>
> The identity, the regrouping, the rank and every caveat below stand
> unchanged; only the arithmetic of the baseline moved.

*2026-09-15. Corrects `p2_design_note.md`. Label-free; arithmetic on measured
shapes, nothing run.*

## What I had wrong

The design note priced P2 as dropping **one of three** dense transports and
**one of four** dense contractions, for a 27.6% saving. That understated it,
because I had not read how the oracle restructures the `(2,1)` slice.

## What it actually does

`M = P·d(s) + 3·A·d(e) + Z·Lᵀ` is substituted on the **right** of the
contractions, and each of its three pieces lands somewhere cheap:

| piece of `M` | where it goes |
|---|---|
| `P·d(s)` | folds into the **P-group**, as a column scale |
| `3·A·d(e)` | folds into the **A-group**, as a column scale |
| `Z·Lᵀ` | the only new term, and it is **thin**: `(P∘P) @ L` then `@ Zᵀ`, both `O(n²r)` |

So the whole `(2,1)` slice collapses to **two** dense contractions with right
factors `A` and `P`, plus `O(n²r)` thin work. Their `_dslices` writes it as

    D21 = (LA) Aᵀ + (LP) Pᵀ + [(P∘P) L] Zᵀ / 3
    LA  = 2·A∘P∘w2 + P∘P∘e
    LP  = A∘A∘w2 + P∘P∘(s/3) + (2/3)·M∘P

**Per layer, seven dense products become four**: three dense transports plus
four dense contractions, against two plus two. *That* is what "7→4" names — I
had been reading it as a term count.

## The corrected price

| | modelled | C/B |
|---|---|---|
| F21 tower | 2,043,909,505,024 | 0.92946 |
| P2 dense | 1,167,912,337,408 | 0.53111 |
| P2 thin, rank 32 | 28,285,861,888 | 0.01286 |
| **P2 total** | **1,196,198,199,296** | **0.54397** |

**Saving: 41.5% of the tower.** The thin overhead is 3.2% of the dense saving,
so the rank is nearly free at this scale — consistent with the oracle's header
claim of *zero extra n³*.

Only the **relative** figure transfers: the model prices plain matmuls and F21
runs Strassen depth 4, which is why it puts the tower at 0.929 B against a
measured 0.422 B total. The share is the transferable quantity.

## Applied to the measured file

F21's live-shape total is `928,457,637,478` FLOPs. Its f32 K2 backbone anchor is
`68,802,510,336`, so the **tower is 92.6%** of the bill and P2 acts on that:

    928,457,637,478  ->  571,700,759,714        C/B 0.4222 -> 0.2600

## What that would be worth, stacked

Each mechanism is measured **separately**; this stacks them arithmetically and
is a projection, not a measurement of the pair:

| | raw MSE | C/B | score |
|---|---|---|---|
| F21, graded | 3.780e-08 | 0.4222 | **1.596e-08** |
| + P1, at its measured 1.2373× on raw | 3.055e-08 | 0.4222 | 1.290e-08 |
| + P2, at the oracle's +0.9% raw | 3.082e-08 | 0.2600 | **8.013e-09** |

For scale: 504aldo sit at 5.4e-9, their clones at 4.2e-9, the leader at 2.5e-9.

So P1 and P2 together would move us from rank 62 to somewhere in the twenties —
**real, and still not the top tier.** The remaining mechanisms (P3 feedback, P4
Strassen/pooling, P5 age-gated basis) are what the rest of the gap needs.

## Caveats that belong next to those numbers

- The stacking assumes the two mechanisms are independent. They touch different
  parts of the chain, which makes it plausible and not proven.
- P2's +0.9% raw is **the oracle's** figure on **their** chain at rank 32. F21
  has a different term set and our own screen has to measure it.
- P1's 1.2373× is a design-tier measurement under an independent Skeptic's
  **PROVISIONAL**, with about two thirds of it attributable to a λ table whose
  value at width 1024 is undetermined.
- Nothing here is a public-board number.
