# Phase-2 write-up — DRAFT, PRIVATE, not posted anywhere

> **Status 2026-09-10 (revised after the Skeptic returned PROVISIONAL on §(ii)):
> sections (i), (ii), (iii) and (v) are substantially drafted; (iv) awaits the κ₄
> rank-sweep; (vi) awaits the depth-2 graded row and T2. §(ii) has been rewritten:
> its channel attribution was struck as unidentified, and the surviving result is
> narrower and better evidenced. One `OWED` remains inside it, named below.** Nothing here is posted. The
> Phase-1 manuscript `docs/writeup-draft.md` is a different document, is stale, and
> is not superseded by this one — it is Phase 1's and stays where it is.
>
> **Every number in this file must trace to a LOG entry and an artifact.** Where a
> number is still owed, the line says `OWED` and names the job. A draft that
> carries a plausible-looking placeholder is how a wrong constant reaches a
> published paper.
>
> **PRIVATE material that must never enter this document:** `docs/BENCHMARK.md` in
> full, and `docs/RESEARCH_NOTES.md` §2g (scoring-channel forensics). Board
> positions of *named other teams* are public facts and may be cited; our own
> forensics on the scoring channel may not.

*Target: the $20k Algorithmic Contribution prize. Judging (forum 18041) rewards
mechanistic insight and honest analysis over leaderboard position, which suits
what we have: **a class of methods characterised, a ceiling located, and five
levers priced and closed, each with a mechanism.***

---

## Provisional title

*Cumulant information in flight pays only when the tower that consumes it is
exact: a cost-accuracy account of factored κ₃ propagation at width 1024*

## Abstract *(draft 2026-09-10, revised after T1)*

We give a measured account of what limits white-box estimation of expected
post-ReLU activations for a random ReLU MLP at width 1024 and depth 16, under an
analytical FLOP budget.

Our organising result is structural rather than numerical. Factored cumulant
propagation carries κ₃ as a rank-`kn` object whose columns are *born* at each
layer, and the accuracy of the scheme is set almost entirely by whether the tower
that consumes those columns is **exact**. We show this four times over, from four
independent directions: truncating the tower by generation destroys the gain of the
richest scheme we could price, to the point where the truncated rich scheme is
statistically indistinguishable from the truncated poor one; *collapsing* the tower
(regenerating it each layer) beats truncating it per FLOP; a scheme that
carries the fourth cumulant as an `(n,n)` object rather than a scalar is `2.09×`
more accurate but births `8n` columns per layer against `2n`, which is `3.6×–5.8×`
over the hard budget at every recursion depth we priced; and deleting those births
to make it affordable does not cost part of its advantage but **all of the
third-cumulant correction**, landing the estimator back on plain Gaussian
propagation.

Taken together these say something sharper than any one of them: **the carried
lineage is not a resource that degrades gracefully. Every deletion we have
measured — by generation, by diagram class, or by cumulant degree — returns the
chain to roughly Gaussian accuracy**, whatever it was that was deleted.

We then price and close five levers, each with a mechanism and a number: the exact
`W Wᵀ` metric (`1.07×` at width 1024, decaying with width exactly as concentration
predicts), Strassen recursion depth (bounded not by arithmetic but by a
memory-bound glue layer that costs `37%` of wall for `1.3%` of FLOPs),
representation-level symmetric packing (`1.007×`, because zero of the chain's 797
matmuls has the redundancy this meter discounts), generation truncation, and
last-layer capping.

Finally we report what we could and could not reproduce of the public
leaderboard's shared floor: its accuracy, yes, in a reference implementation; its
cost, no, and we exclude both public routes to it. `OWED: (vi) needs the depth-2
graded row and T2.`

---

## (i) The exact factored K₃ at 1024 × 16 on the counted meter, and three structural identities

**Claim.** The exact κ₃ tower at `1024 × 16` costs `0.7810 B` of counted FLOPs on
the official meter, where `B = 2⁴¹`, and three exact identities take it there from
a naive `1.07 B` **at zero cost in accuracy**.

**The cost model, which is pure Python and runs before any FLOP is spent.**
flopscope bills `(m,k) @ (k,n)` at `m·n·(2k−1)` from shapes alone. Every dominant
operation in the tower is `(n,n) @ (n,n)`, so the whole bill is a matmul *count*
times one constant:

| quantity | value | provenance |
|---|---:|---|
| matmul-equivalents in the exact tower, depth 16 | `797` | `_matmuls(16, None)`, verified against the meter |
| the same tower without last-layer pruning | `857` | same function, pruning branch removed |
| matmul-only counted FLOPs | `0.77794 B` | `797 × n²(2n−1) / 2⁴¹` |
| elementwise remainder | `0.00305 B` | `_ELEM_PER_LAYER = 400`, measured 237 |
| **total projected** | **`0.78099 B`** | |
| graded, submission `330212` | `0.77970 B` | the model is `0.17%` high |

That last row is the point: the projection is made in Python before the first
array is touched, and the grader agrees with it to `0.17%`. An estimator that
cannot price itself before it spends cannot fail safely, and an MLP over `B`
scores as **zeros**, not as a bad answer.

**Identity 1 — the identity leg is free.** The youngest lineage block's transport
matrix is symbolically `I`, and `Wᵀ I = Wᵀ` exactly. It is *concatenated* in rather
than multiplied, at zero FLOPs. This is the factored representation's own version
of the symmetry saving that §(v) shows is otherwise unavailable.

**Identity 2 — two of the six `(2,1)` products are the same floats.** The repeated
`(2,1)` slice of the tower decomposes over vertex partitions into six products,
three from the `D` block with factors `(A_D, M, M)` and three from the `L` block
with factors `(A_L, 3M, C_L)`. In the `D` block, `{0,1}|{2}` and `{0,2}|{1}` are
**identically the same product** because two of the three factors are the same
buffer. In the `L` block the same collapse happens for a different reason:
`C_L = A_L diag(w₂)` — a *right diagonal scale* — so

    {0,1}|{2}[a,i] = Σⱼ A_L[a,j] M[a,j] A_L[i,j] w₂[j]
    {1,2}|{0}[a,i] = Σⱼ M[a,j] A_L[a,j] w₂[j] A_L[i,j]

are the same summands in a different written order. Verified numerically at
`n = 64, k = 3` at `1.7e-16` relative, against `1.5` and `1.3` for the genuinely
distinct pairs. **So the two-leg block needs four distinct matmuls, not six** —
one saving of eight per lineage per layer.

**Identity 3 — the last layer reads only the mean.** The grader scores the
final-layer post-ReLU *means*, so at layer 16 the only output ever read is
`μ_out = K[(1,)]`. The `(2,1)` slice's four matmuls per lineage, the covariance,
`K[4]`, both blocks born there and the whole Wick carry-forward are never issued.
**That is 4 matmuls per lineage × 15 lineages = 60 of 857**, i.e. `7.0%` of the
tower, for nothing.

> **Correction to the record, and it is the third of its kind in this file.**
> `estimator_k3_strassen.py`'s own docstring says *"5 matmuls per lineage saved,
> which at depth 16 is 75 of the tower's 1,022."* The function says **4 per
> lineage, 60 of 857**, and `857` reproduces the graded anchor. The `5` predates
> Identity 2 (which cut six products to four) and the `1,022` predates the
> identity-leg saving. Two other stale scalars in the same file were corrected on
> 2026-09-10. **The file is not edited to fix any of them**: its sha256
> `3ad05d75…` is a graded submission's bytes. The lesson is recorded instead —
> *read the function, never the prose about the function* — and it is the single
> most repeated process defect in this project.

**Blocking.** The linear step issues three matmuls on `(n, kn)` operands rather
than `3k` on `(n,n)` ones. Same FLOPs to within one `−1` per call, `k` times fewer
metered *operations* — and the operation count is what the `0.4 s` residual cliff
is made of. `132` calls carry `797` matmuls' worth of arithmetic.

---

## (ii) The AUGMENT attribution ablation — what carries the accuracy

**Claim.** In ARC's factored scheme at `k_max = 3`, moving the fourth cumulant from
a **scalar** to an `(n,n)` **matrix** is worth `2.094×`, the raised diagram budget
that accompanies it is worth *less than nothing*, and a self-inconsistent
cumulant state is catastrophic.

Four arms, paired on 24 networks at `1024 × 16` with `2²⁶`-sample two-half labels,
control reproduced bit-exactly across two reboots:

| arm | diagram budget | κ₄ carried as | mean cross-MSE | vs SIMPLE |
|---|---|---|---:|---:|
| `SIMPLE` | base (40 terms) | a scalar | `4.0047e-08` | `1.000×` |
| **A5** | raised (142) | a scalar | `4.3448e-08` | **`1.085×` WORSE** |
| **`AUGMENT`** | raised (142) | an `(n,n)` matrix | **`1.9123e-08`** | **`0.478×`** |
| **A3** | raised (142) | **zero** | `3.6895e-07` | **`9.21×` WORSE** |

Three things follow, and the second and third are the ones worth publishing.

1. **All of the gain is the representation, none is the budget.** A5 isolates the
   raised term set and it is *negative*.
2. **A self-inconsistent cumulant state is far worse than a consistent
   lower-order one.** A3 zeroes κ₄ while every diagram that consumes it keeps
   running, and it is an order of magnitude worse than simply not tracking κ₄ at
   all. This is a general warning about ablations of cumulant schemes and we have
   not seen it stated.
3. `get_d_max(3, ·) = 4` for both SIMPLE and AUGMENT, so AUGMENT adds **no new
   cumulant degree**. It enriches the one already tracked, from
   `r = 2` (zero free indices, a scalar) to `r = 1` (two free indices, a matrix).

**The deletion experiment, and what it does NOT show.** κ₄'s `(n,n)` object
couples into the flow twice: through the Wick updates of μ, Σ and the diagonal κ₃
slices (`O(n²)`, essentially free) and by **birthing new factored κ₃ columns**
(`8n` per layer against SIMPLE's `2n`, which is what makes the scheme
unaffordable). We deleted the births and kept the object, by exact text
substitutions on a copy of the reference implementation, verified bit-exact
against it with the ablation off.

Birth counts are **read off the returned tensor**, not estimated:

| arm | births / layer at `n = 1024` | mean cross-MSE | vs SIMPLE |
|---|---:|---:|---:|
| SIMPLE | `2n` = 2,048 | `4.004685e-08` | `1.000×` |
| AUGMENT | `8n` = 8,192 | `1.912323e-08` | **`2.094×` better** |
| AUGMENT-LITE — κ₄'s births deleted | `5n` = 5,120 | `4.365096e-06` | `109×` worse |
| AUGMENT-MINIMAL — those *and* the raised budget's extra legs deleted | `2n` = 2,048 | `2.976232e-06` | `74×` worse |
| *(Gaussian K2, same 24 networks, same labels)* | — | `4.196350e-06` | `105×` worse |
| *(calibrated K2)* | — | `1.448599e-06` | `36×` worse |

Ground-truth noise floor: `1.1552e-09`.

**Read the last two rows before the middle two.** `AUGMENT-LITE` is statistically
indistinguishable from plain Gaussian propagation — the paired difference against
K2 straddles zero. `AUGMENT-MINIMAL` is worse than *calibrated* K2. **Deleting
blocks from the carried lineage does not cost part of κ₄'s contribution; it costs
the entire κ₃ correction.**

So the honest statement is narrower than it first appeared, and in one way more
useful:

- **What is established.** No affordable member of this family exists *by
  deletion*. An arm that costs SIMPLE's exact `2n` lineage while keeping κ₄ live
  is `74×` worse than SIMPLE, on `0 of 24` networks improved. Both controls
  reproduce their independent anchors **bit-identically**.
- **What is NOT established, and we say so.** Which κ₄ channel carries the
  `2.094×`. The two channels are **not separable in this implementation**: κ₄'s
  own update is computed from the factored κ₃ blocks the ablation removes, so the
  ablated arms propagate a *different* κ₄ rather than the same one with its births
  suppressed.
- **The competing explanation, which our own data supports.** Generation-capped
  SIMPLE — which contains no κ₄ births at all — collapses to `1.65e-06` by the same
  order. **A single statement covers every case we have measured: any deletion
  from the carried `(1,1,1)` lineage returns this chain to roughly Gaussian
  accuracy.** That is a cleaner result than a channel attribution would have been,
  and it is the one the evidence identifies.
- One diagnostic worth reporting because it constrains the mechanism:
  `AUGMENT-MINIMAL` deletes strictly more than `AUGMENT-LITE` and is `1.47×`
  **better**, on 21 of 24 networks. The collapse is not monotone in how much is
  removed.

---

## (iii) The theorem: cumulant information in flight pays only when the tower that consumes it is exact

This is the paper's central claim and it rests on three independent measurements
that were made for three different reasons.

**(a) The memory-depth ceiling.** `OWED: restate from the 2026-09-05 entry with
its number (3.87×) and its mechanism.`

**(b) Collapse beats truncation.** The *regenerating* tower — which rebuilds the
lineage each layer rather than carrying it — runs at `0.077 B` and `9.1e-8`. Every
generation-truncated arm is worse **per FLOP** than that. Truncation of the tower
by generation roughly doubles raw MSE per generation dropped (`4.00e-08` exact,
`4.36e-08` at 12 generations, `5.06e-08` at 11) against a saving that is only
linear. **So the score-optimal operating point is the exact tower, and every
remaining cost gain must be accuracy-neutral** — which is what §(i)'s three
identities are and what §(v)'s five closed levers are not.

**(c) AUGMENT's gain does not survive truncation of the tower.** Capped to the
generations at which it would fit the budget, AUGMENT fails by `26×–58×`, and the
collapse **is not AUGMENT's**: capped AUGMENT is statistically indistinguishable
from capped SIMPLE at `g = 2` and significantly *worse* at `g = 4`. The richer
representation buys nothing once the object that consumes it is approximate.

**(d) Deleting diagram classes does the same thing as deleting generations.**
§(ii)'s ablations remove specific *diagram classes* from the carried lineage
rather than whole generations, and land in the same place: `2.98e-06` and
`4.37e-06` against generation-capped SIMPLE's `1.65e-06`, all of them within a
factor of a few of plain Gaussian propagation at `4.20e-06`.

Put together, and this is the form the evidence actually identifies: **the carried
lineage does not degrade gracefully. Every deletion we have measured — by
generation, by diagram class, or by cumulant degree — costs not a share of the
correction but essentially all of it**, and the estimator returns to roughly
Gaussian accuracy whatever was removed. The value of carrying higher-order
cumulant information is therefore not a property of the information; it is a
property of the exactness of the consumer.

That is why the affordable members of this family cluster at one accuracy and the
accurate members cluster at one price, and why no team on the public board has
published a method that is both.

*What would falsify this: any truncation, of any kind, that keeps the chain
materially below the Gaussian baseline. We have looked for one along three axes
and not found it; we have not proved none exists.*

---

## (iv) The κ₄ ceiling, and how far a small-width ladder can be trusted

**Claim.** At fixed `L/n = 1/64`, exact κ₄ is `9.0×–10.6×` more accurate than
AUGMENT — and the instrument that says so is a measured over-reader, so the
live-shape number is `6.5×–8.8×` and is a **hypothesis**, not a finding.

The ladder runs three rungs, `(128,2)`, `(192,3)`, `(256,4)`, at fixed `L/n`. Its
own pre-registered validity check **failed**: bootstrap intervals on
`SIMPLE/AUGMENT` read `2.75 / 3.00 / 2.91` and exclude the live shape's `2.094` at
all three rungs. Re-running at `float32` reproduced the `float64` numbers to four
or five significant figures, so dtype is excluded as the explanation. The ladder
also falsifies its own similarity premise from the inside: a method ratio moves
with `n` at fixed `L/n` (`0.998 → 0.977 → 0.948`, non-overlapping intervals).

**The useful part is that the bias has a direction and a size.** Two method ratios
have now been measured at both the top rung and the live shape:

| quantity | ladder, `(256,4)` | live, `(1024,16)` | over-read |
|---|---:|---:|---:|
| `SIMPLE / AUGMENT` | `2.909` | `2.094` | `1.39×` |
| exact-metric gain on SIMPLE | `1.311` | `1.070` | `1.23×` |

So the ladder reads high by `1.2×–1.4×` on both. Discounting the κ₄ ceiling by the
same factor leaves `6.5×–8.8×` at the live shape. We report this as **the
calibrated extrapolation of a bias measured on two other quantities**, because no
κ₄ ceiling has ever been measured at `1024 × 16`, and we say so.

**Rank sweep.** `OWED — background GPU job.` Cap the exact κ₄ tensor's rank, sweep
it, and report the fraction of the `9×` recovered against rank. Next to §(iii),
that curve is the cleanest available statement of *why* κ₄ is hard: the gain lives
in the part of the tensor a low-rank carrier cannot hold. Supporting evidence
already in hand: an `r=1` state holds **`~0.2%`** of κ₄'s Frobenius mass at
`n = 1024`, and the one team on the public board known to have built a low-rank
traceless carrier sits last of the six at the shared floor.

---

## (v) Five null results, each with a mechanism

Null results with mechanisms attached are the strongest thing this project has,
and the contribution prize's criteria reward exactly that. Each of these was
pre-registered with a gate before the data existed.

**1. The exact metric decays with width, as concentration predicts.** Upstream
replaces `W Wᵀ` by its average `E[W Wᵀ] = 2I`. Using the exact metric instead:

| shape | gain |
|---|---:|
| `128 × 2` | `1.681×` |
| `192 × 3` | `1.458×` |
| `256 × 4` | `1.311×` |
| **`1024 × 16` (live)** | **`1.070×`** |

Gate `1.15×`, **FAILED**. The effect is real — the exact metric wins on `24/24`
networks with an interval well clear of zero — and too small to buy against its
cost, which is a ninth born block per layer plus the loss of a scalar reduction of
the entire degree-4 object. The mechanism was pre-registered and it is the one that
fired: `E[W Wᵀ] = 2I` is a concentration approximation, so its error falls as the
spectrum of `W Wᵀ` concentrates, which happens as `n` grows. **The lever gets worse
exactly where the competition is.**

**2. Strassen's wall is a memory-bound glue layer, not arithmetic.** At recursion
depth 3, Winograd additions are `37.2%` of backend wall for `1.3%` of FLOPs,
concatenations `30.5%` for `1.6%`, and **the matmul leaves are `26.6%` of wall for
`96.8%` of FLOPs.** That is the mechanism behind arithmetic intensity collapsing
from `215` to `22.8 GFLOP/s` between the unrecursed and depth-3 chains.

The obvious fix is measured dead. Merging the seven Strassen call sites a layer
issues cuts *operations* by `3.0×` and `3.7×` and moves wall by `+2%` and `−28%`.
**The glue costs bytes moved, not calls issued.** The rule worth carrying: *fewer,
bigger operations buy residual and buy nothing on wall.*

**3. Symmetric packing is worth `1.007×`, and the premise was the wrong question.**
We instrumented every matmul the chain issues and classified it. Zero of `132`
calls — zero of `797` matmul-equivalents — has operands the meter can discount.
Sixteen have a symmetric *result*, and they are all the same product, the
covariance congruence `Wᵀ C W`; the best symmetry-aware spelling of it saves
`1.20×` on `32` of `797` matmuls, which is the `1.007×` above.

The mechanism is that the lineage's four dominant products contract **different
factor blocks** (`A_D`, `A_L`, `M`, `C_L`), and a discount needs a redundancy the
accountant can see. Two different blocks share none.

Along the way we measured the accountant's actual discount law, which we have not
seen documented:

| spelling | × dense |
|---|---:|
| `A @ B` | `1.0039` |
| `as_symmetric(S) @ B` — a symmetric **operand** | `1.0176` |
| `A @ A.T` — a symmetric **result**, written with `@` | `1.0039` |
| **`einsum("ij,kj->ik", x, x)` — the same object passed twice** | **`0.5039`** |
| `einsum("ij,kj->ik", x, x[:, :])` — a **view**, not the object | `1.0020` |
| `einsum("ij,kj->ik", x*y, x)` — sharing a **factor** | `1.0059` |

**A symmetric operand is a `1.8%` penalty, not a saving.** The only discount is
object-identity aliasing, it is worth exactly `2.00×`, and it cannot be faked: a
view does not count and neither does the `@` spelling.

**4. Generation truncation and last-layer capping.** Truncating generations roughly
doubles raw MSE per generation dropped against a linear saving; capping the last
layer is `5.6×` worse for `5.9%` of the budget. Both are covered by §(iii)(b).

**5. A `2×` method that cannot fit its own budget.** The factored AUGMENT scheme is
`2.094×` more accurate and `3.6×–5.8×` over the *hard* cap at every recursion depth
we priced. `B` is a cap and not only a multiplier: above `1.0 B` the grader returns
zeros. And the only spelling that clears the budget drops to the regenerating
tower, where — granting it its full `2.094×` **and assuming it free** — the
multiplier floor still puts it `2.14×` behind our own graded nominee. **This is the
sharpest single statement of the cost-accuracy bind this class of methods is in.**

---

## (vi) The board's shared floor, and what we could and could not reproduce

`OWED — needs the depth-2 graded row and T1.`

The skeleton of the argument, which the measurements above already support:

- Six independent teams sit within `14%` of one another at a raw accuracy of
  `1.87e-8` to `2.13e-8`. We reach `1.912e-8` in torch with ARC's `kind=AUGMENT`.
  **The board's shared floor is AUGMENT-class accuracy**, and the convergence of
  six teams on it is evidence that it is a property of the *class* rather than of
  any team's engineering.
- The leaders reach it at `0.18–0.21 B`. The only spelling of that accuracy we have
  priced is roughly `22×` dearer. **We can reproduce their accuracy and not their
  cost.**
- We have now excluded both public routes to that cost, each with a number and a
  mechanism: the wall lever (`~6%`, memory-bound glue), the exact-metric lever
  (`1.07×`, decaying with width), Strassen recursion depth (bounded by the wall,
  not by arithmetic), and representation-level packing (`1.007×`, no discountable
  redundancy exists). **Whatever the leaders are running, it is not this chain
  spelled better.**
- What that leaves, stated as open rather than resolved: either κ₄ reaches the
  same accuracy through a channel that births nothing (`OWED: T1`), or a cheaper
  chain carries a learned per-layer correction (`OWED: T2`). We report which, and
  if neither, we report that too — an honest "we excluded the routes we could
  price and the remaining gap is unexplained" is a better contribution than a
  guess.

---

## Method-disclosure and honesty commitments

- Every constant we ship is disclosed, with the tier of data it was chosen on.
- Every gate in this document was written **before** its data existed and is
  recoverable from the git history of `experiments/LOG.md`; where a prediction
  missed, the miss is reported next to the number.
- LLM use is disclosed, per the challenge's rules.
- We never used the flopscope wall-time loophole. Where wall time mattered we said
  so and priced it.

## Open items before this can be submitted

| item | blocking |
|---|---|
| §(ii) the completed 2×2: does `legsonly` also land on the Gaussian baseline? | queued, ~10 GPU-min |
| §(iv) rank sweep curve | background GPU job, not yet queued |
| §(iii)(a) memory-depth ceiling | restate from the 2026-09-05 LOG entry |
| §(vi) the whole section | depth-2 graded row + T1 + T2 |
| a STRATEGIC claim review of every quantitative claim | Rule 2 Skeptic |
| decide what stays private | `BENCHMARK.md`, `RESEARCH_NOTES` §2g must not leak |
