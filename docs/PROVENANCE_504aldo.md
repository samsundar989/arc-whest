# PROVENANCE — what we took from team 504aldo, and what we did not

*Required by `docs/CLAUDE_CODE_HANDOFF_2026-09-14.md` §1 before any Slice 6
mechanism is ported. Opened 2026-09-14. **This file is maintained, not written
once**: every mechanism that reaches one of our files gets a row, on the commit
that introduces it.*

## The short version

Team 504aldo published a working Phase-2 estimator under the **MIT licence** on
2026-09-10 and invited reuse. Their public submission sits at adjusted
**5.4e-9**; ours (F21, `#330950`) sits at **1.60e-8**. They are three times
better than us and they showed their working.

**We re-implement their mechanisms in our own code. We do not ship their
bytes.** Their repository is a *parity oracle*: something to check our
arithmetic against at tiny shapes, and nothing else. No file of theirs is, or
will be, submitted.

## The eligibility position, stated plainly

Rules §6 says a submission must not be *"derived from another Participant's
submission."* An MIT licence does not settle that, and neither does rewriting
the code: the rule is about derivation, not about copyright. Those are
different questions and this repository has been careful not to conflate them
since 2026-09-10.

**Samuel's decision, 2026-09-14:** re-implement in our own files; v29 is a local
parity oracle only; never submit v29 bytes; and put the neutral question to the
organizers — **with his approval of the exact wording before anything is sent.**

*(Earlier, on 2026-09-10, his decision was to hold off on the question
entirely. The 2026-09-14 decision supersedes it. Both are recorded because a
changed decision is a fact about the project, not an embarrassment.)*

**The question is drafted below and has NOT been sent.** No organizer, forum or
Discord has been contacted. Until it is answered, the standing rule is: own
re-implementation only, this file maintained, no v29 bytes submitted.

### The draft, awaiting Samuel's approval of the exact wording

> Team 504aldo released their Phase-2 estimator under MIT and invited reuse;
> several accounts have since submitted it. For prize verification under §6
> ("not derived from another Participant's submission"), is building on that
> code with attribution and substantial own changes acceptable, and where is
> the line?

Neutral by construction: it asks where the line is, it does not argue for a
position, and it does not name or disparage the accounts that submitted the
published bytes.

## The clone — what was taken from it, and what was not

**Cloned 2026-09-15 with Samuel's explicit approval**, into `external/504aldo/`,
which is **gitignored**: their code is never vendored into our history, never
edited, and never submitted. HEAD is exactly the commit this file already
pinned, `18c17e2d7a9aeacd399cfc2c6b571e4e16dbfb45`.

It is used as a **parity oracle** and nothing else. Every consultation gets a
row here, naming the file, the line and what question it answered.

| date | their file | what was checked | outcome |
|---|---|---|---|
| 2026-09-15 | `estimators/estimator_v17.py:679` | which harmonic projection the per-neuron κ4 core is | **r = 1 matrix core**, `g = (K4 + rowsum K22)·6/(n+4) − (ΣK4 + Σrowsum K22)·3/((n+2)(n+4))`. F21's `_proj_coef_d4_r2` is the **r = 2** scalar — a *different projection*, not a rescaling. This was the bug behind a 9.2× design-screen regression |
| 2026-09-15 | `estimator_v17.py:447–450` | the transport operand and the consumer coefficients | `WW = W*W`, `dG = WW @ (g_prev − var_prev·λ) + var·λ`; `D4 = metric·dG`, `D22 = (metric/6)(dG_i+dG_j)`, `METRIC_C = 2.0`. **Our implementation already matched all of these** — only the birth was wrong |
| 2026-09-15 | `estimator_v17.py:585–592` | the κ4→κ3 feed | Confirmed it belongs to their **compact leg representation**. F21's frozen term table already routes κ4 into the κ3 births, so porting it would double count. **P1b stays struck** |
| 2026-09-15 | `estimator_v16b.py:558–569` | P2's separable split | Identical to what we derived and measured independently BEFORE cloning: `e = w(1,2) − 2·pK1·w(1,1)`, `S_sep = zero_diag(e_i·C_off·w1_j)`, birth `diag(S3) + 3·a_b·d(e) + 3·R·Lᵀ`. No correction needed |
| 2026-09-15 | `estimator_v17.py:247–250, 452` | where the missing `q31` consumer feeds | It is **four term rows** plus a new `(3,1)` slice, not one line. F21's table is upstream's frozen SIMPLE enumeration and does not contain `(3,1)`, so adding it is a deliberate divergence from the pinned reference needing its own contract |
| 2026-09-15 | `estimator_v16b.py:280` | P2's residual rank | They ship **32**, not the handoff's 16. Their F63 row gives raw-MSE cost +1.6% / +0.9% / +0.4% at rank 16 / 32 / 64. A *suggestion* for our rank, to be verified end to end on our chain |
| 2026-09-15 | `estimator_v16b.py:558-569` | the range finder, whole | **`_p2_compress` is their function with one line changed**, not merely their sketch: same single power iteration, same two QRs, same `Bm = Q^T R` and `(Bm.T * 3.0, Q)` return with the factor 3 folded into `Z`, same narrow-shape rank guard. The one line we changed is the sketch, and we changed it because **our invariant suite rejected theirs**: `Om = w32[:, :r]` is deterministic but not permutation-equivariant, and `inv_permutation` went from v30's `1e-13` to **1.07% of raw MSE at 128x8, FAILED** against its 1% bar. We select the residual's own strongest columns instead -- equally deterministic, and equivariant (`5.4e+05x` the control's float32 floor versus `1.2x`, independently reproduced). **That divergence is genuinely ours and it corrects their published estimator**; the rest of the function is not |
| 2026-09-15 | `estimator_v16b.py:583-620` | whether the dense residual reconstruction is avoidable | It is **not**, and they do not avoid it either: `MP = einsum("kiq,kjq->kij", Z, L)` forms it densely at `O(k n^2 r)`. Confirms our own reading that `LP` genuinely needs `A_D . M`. Their tower is 3-D throughout, so they pay no layout shuffle; F21's is 2-D and Strassen-wide, so ours does -- **that difference is ours to measure, not theirs to answer** |

**What was NOT taken:** no code was copied. The coefficients above are
mathematical facts about a harmonic projection — `6/(n+4)` and
`3/((n+2)(n+4))` are properties of the degree-4 tensor at width `n`, derivable
from ARC's own `harmonic.proj_coef`, and the oracle was used to identify
*which* projection applies, not to supply a fitted number. Their fitted λ table
remains unimported.

## Source, pinned

| what | value |
|---|---|
| repository | `https://github.com/504aldo/whest-p2-cumulant-k3` |
| licence | MIT |
| commit read | `18c17e2d7a9aeacd399cfc2c6b571e4e16dbfb45` (2026-09-10 03:24:28 UTC) |
| published | forum post 2026-09-09, repository 2026-09-10 |
| their submission | `#330093` V29, adjusted 5.40e-9, raw 2.13e-8, C/B 0.2526 |
| our control | F21 `#330950`, adjusted 1.5957e-8, raw 3.7795e-8, C/B 0.4222 |

## The mechanisms, one row each

Status is one of **NOT STARTED** / **RE-IMPLEMENTED** / **REJECTED**. A row only
becomes RE-IMPLEMENTED on the commit that lands our own version of it, with its
own tiny-shape parity evidence.

| # | mechanism | their finding | our file | status |
|---|---|---|---|---|
| P1 | κ4 regeneration `G = diag(g) + λ·C_off`, with consumers and the κ4→κ3 feed derived from one state | F68 | `estimator_k3_v30.py` | **RE-IMPLEMENTED**, own code from F21 lineage via `local_sim/p2_build_v30.py`. Birth projection identified against the oracle; consumers and transport were already ours and already correct. **1.2371x on 24/24 design nets** at unchanged cost (t 95% `[+3.36e-9, +4.39e-9]`, sign p 0.0000). Their reported gain is 1.97x, so ~half is missing: the `q31` consumer is absent and λ is an initialiser. Skeptic pending |
| P2 | the 7→4 leg identity, `S21 = S_sep + R_res` with a thin transported residual | F63 | `estimator_k3_v31.py` | **RE-IMPLEMENTED, and PART OF IT IS THEIRS.** See *The P2 timeline* below, which an independent Skeptic reconstructed from git and filesystem timestamps on 2026-09-15 after this row claimed more independence than the record supports. In short: the **reconstruction identity is ours** (committed 1h47m before the clone); the **`(2,1)` regrouping and the range finder are theirs**, matched deliberately from `estimator_v16b.py:558-620`. Our own contributions are the 2-D Strassen-wide layout, its shuttles, and the equivariant sketch. Measured at 1024x16: **`0.7128x` of v30's cost, a 28.7% saving, C/B `0.4222 -> 0.3010`**. **Accuracy not yet measured** -- contract `experiments/LOG.md` 2026-09-15, screen job `1364` |
| P3 | rank-16 D21→birth feedback | F69 | — | **NOT STARTED** |
| P4 | Strassen depth/pooling/block-symmetric `C_pre`/final-layer trim | F86 (cost anatomy) | — | **NOT STARTED** |
| P5 | age-gated shared basis, 384 at age 4 nested to 224 at age 7 | F72/F73 | — | **NOT STARTED** |

## The P2 timeline — what preceded the clone and what did not

*Added 2026-09-15 after an independent Skeptic **REJECTED** this file's earlier
claim that "the algebra is ours and the oracle only confirmed the sketch and
the rank". That claim was written by the agent that did the work, from memory
of its own reasoning. The Skeptic read git and the filesystem instead. **The
timestamps win.** This section exists because the §6 eligibility question turns
on exactly this distinction, and a document that overstates our independence is
worse than useless for answering it.*

| when (UTC) | what |
|---|---|
| **2026-09-14 22:24:39** | `cfb13bd` commits `local_sim/p2_p2_split_probe.py` and `p2_split_n128.json`, already carrying `e = w(1,2) − 2·pK1·w(1,1)` and `(3·S_sep)ᵀ = 3·fac1·d(e)` |
| **2026-09-15 00:11:18** | **the clone** — `external/504aldo/.git` |
| 2026-09-15 00:47:51 | `221a671`, the design note |
| 2026-09-15 02:10:48 | `9c69c68`, the reconstruction identity test |
| **2026-09-15 11:26:21** | `e61b7d8`, the **`(2,1)` regrouping** proved — **11h15m after the clone** |

### What is ours

**The reconstruction identity** — `A_D = M·d(s) + 3·A_L·d(e) + Z·Lᵀ`, and the
observation that `3 S_sepᵀ` is a free column scale on a leg the tower already
carries. Derived, implemented and committed **1h47m before the clone existed**,
and the commit proves it.

**The 2-D Strassen-wide layout and its shuttles.** Their tower is `(k,n,n)`
throughout; F21's is `(n, k·n)` so that one wide matmul feeds Strassen depth 4.
Everything in `_p2_recon` that moves between the two layouts is ours, exists
only because our layout differs from theirs, and is a genuinely different cost
regime.

**The equivariant sketch**, which corrects a real defect in their published
estimator — found by running an instrument, not by reading their code.

### What is theirs, and was matched deliberately

**The `(2,1)` regrouping.** Our own design note, written 36 minutes after the
clone, states the plan as *"Their `_dslices` at `estimator_v16b.py:583-620` is
the shape to match."* It was matched: same decomposition, the same variable
names `LA` / `LP` / `MP` / `PPL`, the same `(2/3)` and `(1/3)` placement, the
same right-factor convention, and the birth tuple `(a_b, Bm.T * 3.0, Q, S3c,
e_b)` verbatim from their `:571`.

*In fairness, and stated as the weaker argument it is:* the regrouping is a
forced consequence of substituting the reconstruction into F21's own
contraction, so an independent derivation would plausibly have reached the same
grouping. That is an argument about what could have happened. The artifacts
record what did.

**The range finder** — see the `:558-569` row above. Ours is their function
with the sketch line changed.

**The rank**, 32, taken from their F63 row.

### Why this correction was made rather than argued with

This file's own *Attribution, for the write-up* section already said the right
thing: *"the κ4 regeneration law, the **7→4 identity**, the D21 feedback and
the age-gated basis are **504aldo's published work**."* The mechanism row
contradicted it. The row was wrong and the attribution was right, so the row
moved.

Nothing here changes the standing rules: own re-implementation only, no v29 or
v16b bytes ever submitted, the §6 question drafted and **unsent** pending
Samuel's approval of the exact wording. It does change what an honest answer to
that question would have to say, and Samuel should see this section before the
question goes anywhere.

### What we are deliberately NOT taking

- **Their fitted λ table** (`lam_prefit_perdump.npy`, 0.95× mean). It was fitted
  on eight **public** MLPs. Our validation policy fits on fresh mints only, and
  importing a public-fitted constant would breach it whoever fitted it. We fit
  our own — `local_sim/p2_lambda_fit.py`, on 24 freshly minted P03 design
  networks, label-free.
- **Their mutable-buffer/`out=` patterns.** Our contract says grader arrays are
  immutable, and a public submission passing is not evidence about what the
  official participant shim permits. Unresolved, and stated as unresolved.
- **Their local-runner advice and package-version range.** External code is a
  reference, not a policy.
- **Their range-finder sketch.** `W[:, :r]` is deterministic but not
  permutation-equivariant, and our invariant suite fails it. We select the
  residual's strongest columns instead. This is the first place our
  re-implementation is *better* than the source rather than merely
  independent of it, and it was found by an instrument, not by reading.

## What we found on our own, that their table does not give us

Two measurements from 2026-09-14, both ours:

1. **E14's label-free Gaussian-tree λ is a different object from theirs.**
   Opposite sign from layer 4 on, an order of magnitude too small, 1.7× too
   large at layer 0, R² ≤ 0.2, depth-profile Pearson r = −0.01. Their κ4 gain
   comes from κ3-*sourced* κ4 production, which a Gaussian-tree diagram does not
   contain. `experiments/2026-09-14_exp_lambda/`.
2. **The unfactored AUGMENT chain cannot be run at the live width.** Its κ3
   state is a dense `(n,n,n)`, so the **retained state alone is ≥133 GiB at
   1024×16 — over 4× a 32 GiB card**, before any transient. So "fit λ from the
   AUGMENT chain at 1024×16" is not a tuning problem.
   *(Three peak figures quoted here earlier were withdrawn on Skeptic review:
   they came from an ad-hoc run never saved as an artifact, and an OOM at
   n=192 is consistent with any exponent ≥ 2.33, so it does not test the law.
   The retained-state figure needs none of them.)*
   And λ's own value at 1024 is **undetermined** — the leave-one-out window
   spread is 8.89× — so `lam_own.npz` is an initialiser, never a constant.
   `local_sim/p2_lambda_fit.py`, `local_sim/p2_lambda_extrapolate.py`.

## Attribution, for the write-up

Any contribution claim must state plainly: the κ4 regeneration law, the 7→4
identity, the D21 feedback and the age-gated basis are **504aldo's published
work**, credited to their 2026-09-09 publication. What would be ours is the
independent λ fit, the width-stability result, whatever mechanism we add beyond
their catalogue, and our negative results. Replication is not contribution, and
this repository does not get to blur that.
