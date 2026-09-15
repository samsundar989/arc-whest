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
| 2026-09-15 | `estimator_v16b.py:561` | how the range finder gets its sketch without randomness | `Om = w32[:, :r]` -- the first `r` columns of the layer's OWN weight matrix, which at He init is i.i.d. Gaussian. We copied it, and **our own invariant suite rejected it**: relabelling the hidden neurons permutes those columns, so the sketch picks a different subspace and the estimator stops being permutation-equivariant -- `inv_permutation` went from v30's `1e-13` to **1.07% of raw MSE at 128x8, FAILED** against its 1% bar. We ship a different sketch: **the residual's own strongest columns by norm**, which is equally deterministic and equivariant besides (`8.8e-01` vs `3.2e-07` relative, measured side by side). **A DIVERGENCE FROM THE ORACLE, and one that corrects it** -- their published estimator has this property too, and we did not inherit it |
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
| P2 | the 7→4 leg identity, `S21 = S_sep + R_res` with a thin transported residual | F63 | `estimator_k3_v31.py` | **RE-IMPLEMENTED**, own code from the v30/F21 lineage via `local_sim/p2_build_v31.py`. Both identities were **proved before any estimator code existed** (`p2_p2_identity_test.py`: reconstruction to 3.7 float32 ulps on real F21 state, regrouping to 6.8e-16 with an arbitrary residual), so the algebra is ours and the oracle only confirmed the sketch and the rank. Tiny-shape parity: control == v30 bitwise, dense arm 2.2e-7, rank-32 arm 2.5e-4. Priced at **31.3%** of the tower, counted off the code. **Not yet measured end to end** -- contract `experiments/LOG.md` 2026-09-15, screen job `1361` |
| P3 | rank-16 D21→birth feedback | F69 | — | **NOT STARTED** |
| P4 | Strassen depth/pooling/block-symmetric `C_pre`/final-layer trim | F86 (cost anatomy) | — | **NOT STARTED** |
| P5 | age-gated shared basis, 384 at age 4 nested to 224 at age 7 | F72/F73 | — | **NOT STARTED** |

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
