# Strategist Packet — 2026-09-10: the AUGMENT route is dead on every axis. Where does 2.5× of cost come from, and is there an accuracy ceiling worth climbing?

*Self-contained. Assumes no repository access and no shared chat history.
Supersedes `STRATEGIST_PACKET_2026-09-09.md`, whose plan is **void** — its
central deliverable was measured impossible. Every number is measured unless
marked PROJECTED, ESTIMATE or HYPOTHESIS.*

---

## 1. The contest, in one page

**Task.** Given the weights of a randomly initialised He-normal ReLU MLP — Phase 2
is **1024 wide × 16 deep, no biases** — predict every neuron's expected post-ReLU
activation under `x ~ N(0, I)`, as a `(depth, width)` array. Any compliant method
is allowed; the organisers expect the best to be mechanistic.

**Score, per MLP, lower is better:**

    score = final_layer_MSE × max(0.1, FLOPs / 2^41)

FLOPs are counted analytically by `flopscope`, a numpy drop-in. λ is zero this
phase, so wall time is not billed — but there is a **120 s per-MLP wall cap**, an
**8 GiB memory cap**, and, decisively for this packet, **`B = 2^41` is a HARD
per-MLP FLOP CAP, not merely a multiplier**: the grader wraps `predict` in a
budget context and returns **zeros at multiplier 1.0** on exhaustion. There is no
"expensive but it still ships" regime. The leaderboard is the mean over a hidden
100-MLP suite.

**Prizes.** `$50k/$20k/$10k` for best score, plus a **co-equal `$20k` for best
algorithmic contribution**, judged on a write-up together with score.

**Decided by a private re-run** of up to two designated submissions on fresh,
unseen MLPs. Public board rank is not the target.

**Dates.** Submissions close **2026-10-17**. Write-up due **2026-10-24**. Today is
2026-09-10: **~5.3 weeks of submission runway.**

---

## 2. Where we stand

**Public board:** #1 is **`3.80e-9`** adjusted. A cluster of **six teams sits at
`1.87–2.13e-8` raw** final-layer MSE and **`0.18–0.21 B`** multiplier.

**Our three graded submissions** (all `0/100` failures):

| id | what | adjusted | raw | `B` | ops | worst wall |
|---|---|---:|---:|---:|---:|---:|
| `330212` | exact factorized-K3 chain | `2.9467e-8` | `3.7793e-8` | `0.7797` | 3,249 | `9.54 s` |
| **`330327`** | **+ batched Strassen L=3** | **`2.031e-8`** | `3.7793e-8` | `0.53745` | 12,357 | `61.93 s` |
| `330290` | + batched Strassen L=4 | `1.836e-8` | `3.7793e-8` | `0.48578` | 15,393 | `114.68 s` |

Rank ≈ **#27**. Strassen is exact — raw is unchanged to `+0.0036%`.

**Nominee A = `330327`**, because `330290`'s worst row is `114.68 s` against a
`120 s` cap and we have measured a **`1.60×` spread in graded wall on identical
bytes**. Nominee B = `330212`.

**Our chain.** An exact factorized third-cumulant flow, ported from ARC's own
research code (`kind=SIMPLE`) into the counted-FLOP dialect. It carries a
*lineage*: each layer adds `2n` columns to a factored rank-`R` third-cumulant
tensor, and at depth 16 the tower is 15 generations deep. That tower is `96%` of
our FLOP bill.

---

## 3. What the last two days established, and it inverts the previous plan

The Sep 9 plan said: *"port `kind=AUGMENT` first, unconditionally — a measured
`2×` with zero research risk; if the delta is `+60%` it still ships."* **AUGMENT
is ARC's augmented variant, which carries the fourth cumulant in flight as an
`(n,n)` object, and it was measured at `1.912e-8` against SIMPLE's `4.005e-8` on
24 networks — exactly the leaders' floor.** The plan was to port it.

### 3.1 The port is `3.6×`–`5.8×` over the HARD budget

Nobody had priced the lineage. Measured by reading the factored tensor's column
count off ARC's own flow at two shapes:

    kind=SIMPLE    births 2n columns per layer
    kind=AUGMENT   births 8n columns per layer

`5` leg blocks `+ 2` live degree-4 blocks `+ 1` from-dense block, against SIMPLE's
`1 + 1`. Over 16 layers that is `885` block-layers against `240`, ratio `3.6875`.
ARC's own FLOP model independently gives `3.68` on the two tower components, and
adds **two rank-`R` contractions that have no SIMPLE counterpart at all** (~half
the AUGMENT bill). Transferred onto our graded-anchored total:

| Strassen depth | ours | AUGMENT | most generous floor |
|---:|---:|---:|---:|
| 0 | `0.780 B` | `5.76 B` | `1.79 B` |
| 2 | `0.604 B` | `4.46 B` | `1.40 B` |
| 3 | `0.537 B` | `3.97 B` | `1.25 B` |
| 4 | `0.486 B` | `3.62 B` | `1.13 B` |

The "floor" column applies two structural savings a Skeptic confirmed from source
(three leg blocks merge exactly; one degree-4 block has zero all-distinct content
and is dead work) **and makes both new contractions free**. It is still over the
cap everywhere.

### 3.2 AUGMENT's gain does not survive any approximation of the tower

The only way to fit is to shrink the tower. Measured on the same 24 nets, capping
carried generations:

| arm | mean MSE | vs uncapped SIMPLE |
|---|---:|---:|
| SIMPLE, exact tower | `4.005e-8` | 1.00× |
| AUGMENT, exact tower | `1.912e-8` | 2.09× better |
| AUGMENT, 4 generations | `7.42e-7` | 18.5× worse |
| AUGMENT, 2 generations | `1.65e-6` | 41× worse |

And the collapse is **not AUGMENT's**: at the same cap, capped AUGMENT is
statistically indistinguishable from capped SIMPLE (`t = 0.50` at `g=2`) and
significantly *worse* at `g=4` while carrying four times the columns. **AUGMENT's
`2.09×` is a property of the exact tower and vanishes under truncation.**

### 3.3 The one spelling that fits loses on arithmetic alone

Only `g=1` clears the budget, and there our own fallback logic stops truncating
and drops to a **regenerating** tower (a rank-`n` projection of all generations,
which we measured beats every truncation arm). That closes with no new data:

    regenerating SIMPLE, measured        raw 9.09e-7 at 0.0767 B
    grant AUGMENT its FULL 2.094x gain   raw 4.34e-7
    x the 0.1 multiplier FLOOR (assume it is FREE)
    best conceivable adjusted            4.34e-8
    nominee A, graded                    2.031e-8
    -> 2.14x WORSE, in the most generous case reachable

**Cost is not assumed at all** — the bound uses the multiplier floor, so it holds
for a free estimator. **AUGMENT is deprioritized on every axis.**

### 3.4 Cost engineering: the wall lever is closed too

We profiled where the Strassen wall goes at depth 3:

| class | % of operations | % of FLOPs | % of wall |
|---|---:|---:|---:|
| Winograd add/sub | 57.8% | 1.3% | **37.2%** |
| concatenate | 16.5% | 1.6% | **30.5%** |
| **matmul leaves** | **1.1%** | **96.8%** | **26.6%** |

The arithmetic Strassen exists to save is a quarter of the wall. The obvious fix —
merge the seven Strassen call sites per layer — cuts operations `3.0×` and `3.7×`
and moves wall by `+2%` and `−28%`. **The glue is memory-bound: it costs bytes
moved, not calls issued.** What remains is Winograd's 15-add form (`~6%`) and an
unpriced hybrid. Recommend closing at `6%`; its whole prize was making depth 4
nominee-grade, worth `9.5%` of score.

### 3.5 One clean negative that cost 41 seconds

ARC's flow approximates the metric `W Wᵀ` by its average `2I`. Turning that off is
worth `1.68×`, `1.46×`, `1.31×` at widths 128, 192, 256 — and **`1.070×` at the
live 1024**, below our pre-registered `1.15×` gate. Real on 24/24 nets, too small
to buy. Closed.

### 3.6 A small-width ladder, and why it is now calibrated rather than trusted

We built a ladder at fixed depth/width `= 1/64` — `(128,2)`, `(192,3)`, `(256,4)`
— to answer lane questions without a 1024-wide oracle. **Its pre-registered
validity check failed**: it reads `SIMPLE/AUGMENT` at `2.75/3.00/2.91` where the
live shape is `2.094`, with bootstrap intervals excluding the live value at every
rung. It also falsifies its own premise from inside — one method ratio provably
moves with width at fixed depth/width. Re-running at float32 to match the anchor's
dtype reproduced the float64 numbers to 4–5 significant figures, so dtype is not
the explanation.

**But §3.5 gave a second calibration point, and both point the same way:**

| quantity | ladder, top rung | live shape | over-read |
|---|---:|---:|---:|
| `SIMPLE / AUGMENT` | `2.909` | `2.094` | `1.39×` |
| exact-metric gain | `1.311` | `1.070` | `1.23×` |

**The ladder reads method ratios high by `1.2×`–`1.4×`.** That is a bias with a
direction and a rough size, which is more useful than "cannot extrapolate".

---

## 4. The one result that is still live, and it is large

On the same ladder, **full `K4` — the exact fourth cumulant — is `9.0×`–`10.6×`
more accurate than AUGMENT at all three rungs**, with no downward trend. A Skeptic
verified from source that this arm really is the exact-κ₄ ceiling (factoring drops
none of its 171 terms; its rank never truncates) and that it shares an **identical
diagram budget** with AUGMENT, so this is a clean *representation* comparison and
not a raised-budget one. Signal is 7.7–11.5 standard errors above zero.

Discounted by the ladder's measured `1.2×`–`1.4×` over-read, the live-shape
ceiling would be **`6.5×`–`8.8×`. HYPOTHESIS, not a finding** — no κ₄ ceiling has
ever been measured at 1024×16.

**This matters because it contradicts the working assumption of both previous
packets.** We have been treating `1.9e-8` as a floor that six teams have reached
and that we must undercut by cost engineering alone. If there is a `6×`–`9×`
accuracy ceiling above AUGMENT, then **accuracy is not exhausted**, and the board's
shared floor is a property of one representation rather than of carrying κ₄ in
flight as a class.

The counter-argument, which we take seriously: the team whose entire published
contribution was a low-rank traceless-κ₄ carrier sits at `2.13e-8`, the **worst**
of the six. If that structure bought a large factor at this shape, they would show
it.

---

## 5. The strategic problem, stated plainly

**Our raw accuracy is `3.78e-8` at `0.537 B`. The leaders are at `1.87–2.13e-8`
at `0.18–0.21 B`.** They are simultaneously `~2×` more accurate and `~2.7×`
cheaper. Two facts constrain what they can be doing:

1. **They are not running a 15-generation exact lineage.** Strassen depth alone
   cannot take our structure to `0.2 B`: `(7/8)^6` on `0.78 B` is `0.35 B`, and
   depth is already wall-limited at 3.
2. **They are not running ARC's factored AUGMENT either.** We have now priced it
   at `3.6×`–`5.8×` over the hard cap — roughly **`22×` the leaders' multiplier**
   for the accuracy they have.

Their telemetry (scraped, Sep 5): **25–35k operations per MLP** against our
12,340; **5–10 GFLOP/s** against our 22.8; **70–119 s wall**, i.e. they run *to*
the cap. A deeper, op-heavier, lower-intensity spelling at `2.7×` fewer billed
FLOPs.

### The remaining inventory, with status

| lever | status |
|---|---|
| Strassen depth | wall-limited at 3; depth 4 is `1.05×` margin, not nominee-grade |
| Strassen glue re-spelling | **closed at `~6%`** — memory-bound, §3.4 |
| dead/cold-column pruning | measured `0.961×` free; `0.845×` needs a compensation recovering `139×` our MSE |
| generation truncation | measured a losing trade at every depth |
| last-layer skip | already banked |
| AUGMENT port | **dead on every axis**, §3.1–3.3 |
| exact metric | **closed**, `1.07×`, §3.5 |
| **symmetric-packed spelling** | **UNPRICED. The only representation-level cost lever never tried.** ARC's own paper counts symmetric-tensor FLOPs at `α ≈ 1/6` for order 3; our two dominant operations are dense `Θ(n²R)`. If `flopscope` bills the packed shape, this is the `2.5×`. If it bills the unpacking, it is nothing. |
| **κ₄ ceiling above AUGMENT** | **UNMEASURED at the live shape**, `6.5×`–`8.8×` projected, §4 |
| learned corrector | not started; offline and unmetered; board evidence caps it near `1.1×` |

---

## 6. What we would like judged

**Q1. Given that AUGMENT is dead, is the remaining cost lane worth the slice?**
The symmetric-packed spelling is the only unpriced representation-level lever and
it is binary: `flopscope` bills by shape, so either a contraction spelled on the
`n(n+1)(n+2)/6` unique entries is billed at that size, or the packing/unpacking is
billed and it is worthless. It is `~5` days to find out. If it fails, cost is
closed at `~0.5 B` and the realistic finish is top-10 to top-15 on score. **Is
that the right use of the next week, or should it go to §4?**

**Q2. Is the κ₄ ceiling worth pursuing with 5 weeks left?** The honest framing:
we have a `6.5×`–`8.8×` projected ceiling and **no idea what a cheap carrier
recovers of it**. Our own Skeptic's proposal is to measure *the climb, not the
ceiling* — cap the exact-κ₄ tensor's rank, sweep it, and ask what fraction of the
gain a rank that fits inside `0.1 B` recovers. That is one job. **But every κ₄
carrier we have priced has been catastrophically expensive, and the one team who
built one is last of the six.** Is this a lane or a trap?

**Q3. What is the most likely explanation of the leaders' `0.18–0.21 B` at
`1.9e-8`?** This is the question we cannot answer from inside. Their telemetry
says many small operations at low arithmetic intensity, running to the wall cap.
Our regenerating collapse — a rank-`n` projection of all generations rather than a
carried tower — costs `0.077 B` and gives raw `9.09e-7`, i.e. `24×` worse than
our exact tower for `7×` less cost. **Is the leaders' architecture plausibly a
cheap regenerating or low-rank flow plus a strong accuracy layer, rather than an
exact tower plus cost engineering?** If so we have been optimising the wrong
chassis for three slices.

**Q4. Nominees.** Nominee A is `330327` at `2.031e-8` with `1.21×` wall margin
under measured contention; we adopted a `≥2.0×` margin rule, which A does not
meet, and the only thing that does is Strassen depth 2 at `0.604 B` (a `12.4%`
score cost for `2.28×` margin). **With no better candidate now in prospect, should
nominee A be moved to the safer depth?** A single timed-out row on the fresh 100
costs `~9e-3` on the mean; break-even timeout probability is `~3e-7`.

**Q5. Effort split for the remaining 5.3 weeks**, given the write-up is co-equal
prize money and its spine is now unusually strong: the exact factorized chain on
the counted meter, the AUGMENT attribution ablation, the memory-depth ceiling,
**the null results in §3.2–3.5 (an approximation that vanishes with width; a
cost lever that is memory-bound not arithmetic-bound; a `2×` method whose gain
does not survive its own budget)**, and the ladder's measured over-read bias. That
is a mechanistic paper regardless of rank.

---

## 7. Two things about this evidence a reader should weigh

**Tier.** Every accuracy number above comes from a 24-network DESIGN asset that
has now been re-read seven times. Its own manifest says in writing that nothing
measured on it may close a lane, so §3.2, §3.3 and §3.5 are stated as
*deprioritized* rather than *disproven*. The margins (`2.14×`, `26×`–`58×`) make
that distinction operationally irrelevant for five weeks, but not rhetorically
free.

**Reliability of this packet's author.** Three independent Skeptics ran on this
slice. All three found real defects and two changed a conclusion: the port was
first reported as dying on the wall cap when it dies on the FLOP cap; the capped
result was first attributed to AUGMENT when it is a property of truncation; a
ratio was reported with its sign inverted; and a load-bearing constant was read
from a stale docstring instead of from the function. **The measurements have held
up under adversarial re-checking — one Skeptic reproduced the capping seam
against an independent implementation to `3e-5` — but first readings of them have
not.** Weight the §4 hypothesis accordingly: it is the one live claim that has
*not* yet survived a live-shape measurement.
