# Independent Skeptic verdict: scoped ACCEPT for the algebra and conventional ledger; architecture INCOMPLETE

2026-09-12. This is a truth-free design review of `lowrank_feed_cost.md` at
`809b65b`, against `repair_architecture.md` and the accepted n=2
`signed_cubature/skeptic_verdict.md`. It is not an executable factor check, a
flopscope measurement, a source-closure result, a score decision, or deployment
acceptance.

## Strongest counterexample and implementation consequence

The costed all-age tower requires **additive fourth-cumulant increments** at
later nonlinear layers. The accepted cubature construction instead gives the
*whole* fourth cumulant of one chosen weighted discrete law. Those are different
objects. In one coordinate, let an existing source have fourth cumulant `k`,
transport it through `B=1`, and leave the output law unchanged. A new cubature
of that full output law also has cumulant `k`. Inserting it as a second source
would make the bank's readout `2k`, though the true cumulant remains `k`; the
new increment is zero. More generally, the fourth cumulant of dependent
contributions contains mixed cumulants, so it is not their separate fourth
cumulants summed. Defining a later source by subtracting the transported old
tensor avoids double counting, but does not establish that this residual has a
small nonnegative weighted-discrete representation or the memo's `n x S` cost.
This does not falsify the memo's explicitly conditional schedule. It does
change the next implementation choice: establish a two-layer, truth-free
increment/source identity (including its covariance convention and mixed terms)
before building all 105 K3 feeds. A first-source-only microkernel can test the
factor/readout implementation without assuming later-source closure.

The conventional negative `D21` family is the immediate cost risk even in that
first-source path. At `S=64`, `(455+91) 2n^3 = 1.172526 T` of the full
`1.868277 T` dense-product subtotal comes from this family; it is independent
of `S`. For one first-layer source, its negative family is `182*2n^3 =
0.390842 T` within `0.703253 T`. A rank-only experiment therefore cannot
remove the dominant full-history negative readout. The next microkernel should
measure this family under the actual immutable array API and residual cliff,
while separately testing any proposed fusion or approximation. These are
conventional multiply/add counts, not counted flopscope FLOPs or a score bound.

## Independent algebra check

Write `C=V diag(p)V^T`, `c_i=C_ii`, `P=diag(w1)V`,
`H=P diag(p)P^T`, `Q=diag(w1)C`, `A=diag(w2)`, and `a_i=w2_i c_i` at
birth. In the negative fourth tensor,
`K^-_iijk=-(c_i C_jk+2C_ij C_ik)`. Substitution into the three
quadratic-response placements gives `-S(a,H)/2` from the first pairing.
For each placement, the other two pairings have the same free covariance
column `h`, giving `-sum_h Sym3(A_:h,Q_:h,Q_:h)`. The displayed factorization
therefore has the right minus sign and no missing factor of two. The positive
part is `sum_s p_s Sym3(w2*V_s^2,w1*V_s,w1*V_s)/2`.

At a future map `B`, transport `A,P,a` once and retain the birth-time
`V,p`. Then `Q=P diag(p)V_birth^T`, `H=P diag(p)P^T`, and
`A^+=A(V_birth^2)` reconstruct the linearly transported birth factors.
Reapplying `w2` or squaring a transported `V` would generally fail. The same
`A` belongs to all K4 ages consumed at one birth layer; `a,P,Q,H` remain
age-specific. Thus `sum_age Q_age^2` may precede one product with the shared
`A^T`, but the separate `(A*Q_age)Q_age^T` products cannot be replaced by a
single product of summed `Q` without cross-age terms.

For `D21_ij=F_iij`, the negative placements are
`-[a_j H_ii+2a_i H_ij]/2 - sum_h[Q_ih^2 A_jh+
2A_ih Q_ih Q_jh]`, matching the two oriented matrix products in the memo.
Setting `j=i` gives `D3^-_i=-3a_iH_ii/2-3sum_h A_ih Q_ih^2`.
The positive placements give the memo's `D21^+` and `D3^+` with the same
orientation. I separately enumerated all 64 birth tensor entries for a
float64 `n=4,S=5` centered positive-weight example, transported through a
nonconstant signed `4 x 4` matrix, then compared the factor readouts with
direct tensor transport: maximum absolute birth, `D3`, and `D21` errors were
`5.55e-17`, `2.78e-16`, and `1.11e-15`. In this example
`max|D21-D21^T|=2.44`, so silently symmetrizing `D21` is demonstrably wrong.
This is an independent tiny algebra check only; it does not price or validate
a deployed implementation.

## Schedule and arithmetic audit

With layers `0..15`, the prior-source counts at consumers are `1..15`:
`120` K4 uses, split into `1..14=105` nonfinal full consumers and `15`
final mean-only consumers. The 105 nonfinal uses create 105 age-specific K3
feeds at birth layers `1..14`. Their future-use count is
`sum_{l=1}^{14} l(15-l)=560`: `sum l(14-l)=455` full `D21` uses plus
`sum l=105` final `D3` uses. Fourteen birth-layer `A` matrices take
`sum_{l=1}^{14}(15-l)=105` transports. One shared first negative matrix
product per birth/readout group gives `sum_{l=1}^{13}(14-l)=91` products;
the other negative product remains once per full age/readout (`455`). These
independently reproduce the memo's `455+91` term. The first-source-only
schedule also checks: 15 K4 uses, 14 K3 births, 105 K3 future uses, and 91
full `D21` readouts.

Using exactly `2mkq` per matrix product, I recomputed the eleven listed
families without rounding intermediate lines. Their dense-product subtotals
are `1.868277219328 / 3.275550097408 / 8.904641609728 T` at
`S=64/256/1024`; first-source-only subtotals are
`0.703252660224 / 0.963366617088 / 2.003822444544 T`.
The final persistent bank is
`14(4n^2)+105(4nS)+105(4nS)+105(4n)+15(4nS)` bytes, or
`112.66015625 / 281.41015625 / 956.41015625 MiB`.
Adding the stated `10n^2+6nS` float32 scratch slots and one immutable
`A` plus `P` old/new overlap gives
`158.41015625 / 332.41015625 / 1028.41015625 MiB`, consistent with the
memo's rounded-up `159/333/1029 MiB` array allowances. I found no arithmetic
error in these conditional ledger lines. The scratch reservation is a plan,
not a proven peak: source production, library temporaries, the ordinary K3
backbone, Python process memory, and later-layer covariance/mixed-source work
are outside it.

## Boundary and cheapest follow-up

The signed-cubature acceptance supports the one-source weighted-discrete
identity and signed linear transport at `n=2`. The calculations above support
the *declared quadratic-response truncation* and its conditional storage/cost
model. They do not establish the full cumulant response, an additive later
source law, finite rank adequate for challenge width, official counted FLOPs,
the 8 GiB lifecycle, 0.4-second headroom, or accuracy. The strongest next
truth-free check is a small independent dense two-layer cumulant oracle that
contrasts a full later output cumulant with its transported-old-plus-new
increment, followed by an immutable factor/readout microkernel and official
operation ledger. No label, decision seed block, vault, or submitted estimator
is needed for this review.
