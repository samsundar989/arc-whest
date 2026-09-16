# Strategist packet — we can replicate the published leader, and that may be the wrong goal

**Private decision request, 2026-09-15.** Answer as if you have no repository
access and no chat history. Everything needed is below.

This is **not** a request for another cost optimisation. We have just finished
one, it worked, and finishing the remaining ones is mechanical. The question is
whether we should, and what else the remaining effort could buy.

---

## 1. The task, the score, and the hard limits

The ARC White-Box Estimation Challenge 2026, Phase 2. Given a random
He-initialised, bias-free ReLU MLP — **width 1024, depth 16** — predict the
expected post-ReLU activation of every neuron under `x ~ N(0, I)`, as a
`(16, 1024)` array. The estimator sees **the weights**, never input samples and
never truth.

    score = mean over MLPs of  MSE_final_layer  x  max(0.1, FLOPs / 2**41)

Lower is better. Both halves matter and they trade against each other.

Hard limits: Phase-2 λ is zero, so counted FLOPs are the whole cost term.
Residual (uninstrumented) wall time is **not** priced but a network exceeding
**0.4 s** of it FAILS and scores as zeros. Wall cap 120 s/network, memory
8 GiB. Stack is WhestBench 0.16.1 / flopscope 0.12.1. Grader arrays are
**immutable**; the estimator may import only flopscope, its numpy facade,
whestbench and the Python standard library — **no numpy, scipy or torch**.
Routing arithmetic through uninstrumented arrays to dodge the FLOP counter is
explicitly forbidden by the organisers and we compete clean.

**The prize is decided by a private re-run of up to two designated submissions
on fresh MLPs.** Public board rank is not the target. There is a **second,
co-equal objective**: a **$20,000 contribution prize** whose deliverable is a
write-up.

## 2. Our method, in enough detail to reason about

We propagate **cumulants** layer by layer in closed form rather than sampling.
For each layer the pre-activation distribution is summarised by its first few
cumulants; ReLU's action on them is expanded by a Wick/Hermite term table; the
result is pushed through the next weight matrix.

We carry cumulants to **third order exactly**, plus a **fourth-order channel**.
The expensive object is the third-cumulant state. Representing it densely at
width 1024 is impossible — a dense `(n,n,n)` retained state is **≥133 GiB** at
the live shape — so it is carried **factored**, as a "tower" of thin legs that
are transported by the same linear map every layer. One lineage of the tower is

    Sym( sum_j A_D[:,j] (x) M[:,j] (x) M[:,j] )     +  Sym( sum_j A_L[:,j] (x) 3 M[:,j] (x) C_L[:,j] )

with a shared `M`. A new lineage is born at every layer, so at depth 16 the
tower carries up to 16 of them; the legs are concatenated into wide `(n, k*n)`
matrices so that one matmul advances the whole tower, and the matmuls run
Strassen at depth 4.

**Where the cost goes.** Per layer the products that scale with the tower are
3 transports plus 3 contractions. Our latest change replaced the dense `A_D`
leg with an exact factored identity, removing one transport and one contraction
and leaving `O(k n² r)` thin work at rank `r = 32`. Measured, at the live shape,
in the official counter:

    control   928,411,199,060 FLOPs    cost term 0.4222
    candidate 661,806,923,368 FLOPs    cost term 0.3010     0.713x

## 3. Where we stand, and where everyone else stands

| | raw MSE | cost term | score |
|---|---|---|---|
| our graded submission (F21) | `3.780e-08` | `0.4222` | **`1.596e-08`** |
| our new candidate, projected | `3.082e-08` | `0.3010` | **`9.28e-09`** |
| a published competitor, "team A" | `2.130e-08` | `0.2526` | **`5.38e-09`** |
| accounts that submitted team A's released code | — | — | `4.2e-09` |
| **the leader** | unknown | unknown | **`2.5e-09`** |

The candidate's accuracy and cost gains are **measured** on 24 freshly minted
networks (paired, common random numbers, studentised-t, `1.3902x` over its own
control on 24/24, gate pre-registered at `1.20x` before any data existed). The
public projection is an extrapolation and is labelled as one.

**Decomposing the remaining gap to team A** — and this is the fact that
prompted this packet:

    total  1.724x   =   accuracy 1.447x   x   cost 1.191x

**We have been spending our effort on the smaller half.** After this change our
cost is within 19% of theirs; our *accuracy* is 45% worse.

The leader is a further **2.15x** beyond team A, by an unknown method, and
**3.71x** beyond our projected candidate.

## 4. The strategic problem

**Team A published their entire method under MIT on 2026-09-10 and invited
reuse.** We have their repository. Our remaining planned mechanisms are all
theirs:

- **P3** — a rank-16 feedback from the `(2,1)` slice back into the birth.
  *Accuracy.*
- **P4** — Strassen depth/pooling, block-symmetric pre-covariance, final-layer
  trim. *Cost.*
- **P5** — an age-gated shared basis, 384 dimensions at lineage age 4 nested to
  224 at age 7. *Cost.*

Our own provenance audit — run by an independent falsifier against git
timestamps rather than against our recollection — found that **we are already
less independent than we had recorded.** One identity we derived ourselves,
provably, 1h47m before we cloned their code. The second we proved 11 hours
*after* cloning, and our own design note from 36 minutes after the clone says
in writing: *"Their `_dslices` at [file:lines] is the shape to match."* It was
matched: same decomposition, same variable names, same coefficient placement.
Our rank-compression routine is their function with one line changed — and that
one changed line is a genuine correction, because their sketch is not
permutation-equivariant and ours is.

So the honest position is: **if we implement P3, P4 and P5 we will have
re-implemented team A's estimator and will land near their `5.4e-9`.** That is
a real improvement on where we are. It is also, by our own written standard,
*replication rather than contribution* — which matters because one of the two
prizes is explicitly for contribution.

## 5. What we are asking

**Q1 — the allocation question.** Given that our remaining catalogued
mechanisms are a competitor's published work, and that the residual gap to them
is **predominantly accuracy (1.447x) rather than cost (1.191x)**, how should
remaining effort be split between:

- (a) finishing the replication to reach ~`5e-9`;
- (b) searching for an accuracy mechanism that is *not* in their catalogue;
- (c) redirecting effort to the contribution write-up, whose prize does not
  depend on score at all?

We are not asking which is nicer. We are asking which has the highest expected
value given that the score prize is decided by a **private re-run of up to two
designated submissions**, where being 20th or 8th on a public board is
irrelevant and only the designated entries' fresh-network performance counts.

**Q2 — the accuracy question, which is the technical heart of it.** Our raw MSE
is `3.08e-08` against team A's `2.13e-08` at comparable truncation order. Both
of us truncate the cumulant hierarchy at third order with a fourth-order
correction. **Where does a 1.45x accuracy difference plausibly come from at
fixed truncation order?** Candidate explanations we can think of, and we would
like these ranked or replaced:

  - their `(3,1)` cumulant slice, which our frozen term table does not contain
    at all (adding it is a deliberate divergence from the upstream enumeration
    and needs its own contract);
  - a feedback path from the computed slices back into the births (their P3);
  - a fitted per-layer coefficient — they ship a table fitted on eight public
    networks, which **our validation policy forbids us from importing or
    imitating**, since we fit only on freshly minted networks;
  - accumulated float32 error in a 16-layer chain of wide matmuls, where they
    may simply be reassociating more carefully;
  - something structural about ReLU's action on the *fourth* cumulant that a
    scalar or per-neuron channel cannot represent.

**Q3 — the unknown leader.** `2.5e-9` is 2.15x beyond the best *published*
method. At this task, what method class would you expect to be capable of that,
that is **not** cumulant propagation? We would rather know we are on a
dominated approach now than after two more weeks of refining it. Be concrete
about what would be cheap to test.

**Q4 — the contribution claim.** If we do finish the replication, what is
defensibly *ours* to claim in a contribution write-up? Our current honest
inventory: an independently-derived leg identity; a label-free fit of a
per-layer coefficient that everyone else fits on public data, together with the
finding that its value is badly determined (an 8.9x leave-one-out spread) while
its *effect* is flat enough near the optimum that the uncertainty does not
propagate; a correction to their range finder's permutation-equivariance; a
measurement protocol with pre-registered gates and an independent falsifier;
and a set of negative results. Is that a contribution-prize submission, or is
it a methods note?

## 6. Constraints on any answer

- **No fitting on evaluation data.** Constants are fitted on freshly minted
  networks only, never on the public split, never on anything we score against.
  A proposal that requires fitting on public data is refused, whoever else does
  it.
- **No open-ended search.** Autonomous optimisation loops are out: the cost half
  of the objective is the FLOP counter, and a search that optimises against it
  is optimising against our own evaluator.
- **Shape-generic and deterministic.** The estimator must work at any width and
  depth, and any randomness must derive from the provided seed. We currently use
  no randomness at all.
- **Nothing is submitted without the principal's explicit, per-submission
  approval**, and no agent designates entries for the private re-run.
- We have roughly **two person-weeks** of agent time and a single workstation.
  Heavy runs are queued, not launched.

## 7. What a useful answer looks like

A ranked allocation for Q1 with the reasoning made explicit; for Q2, either a
ranked list of the accuracy candidates or a better hypothesis, each with the
**cheapest experiment that would discriminate it**; for Q3, a named method class
and a bounded test; for Q4, a plain judgement. Where you are guessing, say so —
we have been burned this month by confident numbers that were models rather than
measurements, and we would rather have a calibrated "I don't know" than a
plausible fabrication.
