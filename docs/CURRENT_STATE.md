# CURRENT_STATE.md — read this first

**Snapshot: 2026-08-24. Phase 2 is live.** This file says where execution stands;
[`GAME_PLAN.md`](GAME_PLAN.md) is the only active architecture/roadmap;
[`VALIDATION.md`](VALIDATION.md) is binding for measurement; the newest entries at
the bottom of `experiments/LOG.md` contain the evidence narrative.

## One-minute orientation

- Phase 2 closes **2026-10-17 23:59 UTC**. Team/registration freeze is
  **2026-10-02 23:59 UTC**; contribution write-up is due **2026-10-24 23:59 UTC**.
- Shape is `1024 x 16`; budget is `2**41`; multiplier floor is `0.1`.
- Official stack is WhestBench 0.16 / flopscope 0.12. Lambda is zero; residual cap
  is 0.4 seconds; total wall cap 120 seconds; solution memory 8 GiB.
- The live leader is Puffi at `1.802e-8` adjusted / `3.603e-8` raw with
  `1.099802385e12` FLOPs. There were about 30 entries at the snapshot.
- This account has **no Phase-2 submission**.
- The current shipped file routes Phase-2 width to basic `k2p1`. Expected Phase-2
  performance is around `3.96e-7` adjusted / `3.95e-6` raw: about 22x and 100x
  behind the leader respectively.
- The repo's scientific decision protocol is strong. Its deployment evaluator is
  not yet Phase-2-truthful.
- **Immediate next action:** Phase-2 evaluator repair, an IMPORTANT slice. Samuel's
  three written predictions are required before instrument edits.

## Authoritative document map

| need | source |
|---|---|
| current status and next action | this file |
| target architecture and full execution plan | `docs/GAME_PLAN.md` |
| live rules/leaderboard/research evidence | `docs/PHASE2_LIVE_AUDIT_2026-08-24.md` |
| measurement protocol | `docs/VALIDATION.md` |
| operational traps | `docs/GOTCHAS.md` |
| contracts, results, Skeptic verdicts | bottom of `experiments/LOG.md` |
| independent review brief | `docs/SKEPTIC.md` |
| Claude implementation prompt | `docs/CLAUDE_CODE_HANDOFF.md` |
| contribution manuscript | `docs/writeup-draft.md` |

The old frontier packets and handoffs are superseded. Stub files remain only where
source comments/tools link to their historical names; they redirect here and must
not be treated as plans.

## Current deployed and research state

### Shipped estimator

- `estimator.py` is still v2.6.1 and must remain byte-identical to its frozen archive
  unless a STRATEGIC contract authorizes a versioned change.
- The graded Phase-1 file remains v2.4.1.
- Phase-2 width takes `_analytic_only`; the effective method is K2p1.
- None of the Phase-1 rotation/Sobol/pruning/Strassen carrier is active at width
  1024.
- Official 0.16 validation passes.
- One official Phase-2 subprocess smoke produced:

| metric | one-row anchor |
|---|---:|
| score | `4.501854e-7` |
| final-layer MSE | `4.501854e-6` |
| multiplier | `0.1` |
| counted FLOPs | `137,600,326,656` |
| `C/B` | `0.0625734` |
| residual | `0.05368 s` |
| total wall | `0.35487 s` |
| failures | `0` |

This is an operational smoke, not a score estimate.

### Phase-2 local evidence that survives audit

P2-CP1 used 96 fresh 1024x16 networks and two independent `2**22` truth replicas.
There were no truth failures. Its narrow conclusions remain credible:

- K2p2 improved K2p1 about 1.052x for about 0.18% extra counted cost.
- Tested IID and ordinary random-orthogonal arms lost to K2.
- Higher Price/Hermite covariance-map orders plateaued.
- Those arms do not bound valid Kerdock/MUB structured quadrature.
- P01 was a DESIGN block, so it cannot support binding global lane kills.

P2-CP3R's durable mechanistic result:

- 94.61% of K2 error is accumulated mean/variance-state drift;
- marginal third/fourth corrections in the scalar ReLU mean map recover 4.47%;
- the registered O3-versus-O0 gate passed, then the recorded KILL used a different
  post-hoc O1-versus-O3 estimand;
- the oracle reset to true marginal state every layer and did not propagate ARC's
  augmented-K2 joint state.

Therefore marginal final-mean Edgeworth correction remains low value, but recurrent
joint-state augmented K2 is open.

### Seed and vault state

- P01 has consumed **144/600** seeds; 456 DESIGN seeds remain.
- P01 is not a DECIDE block.
- Existing SINGLE_USE Phase-1 blocks keep their recorded state in the ledger.
- The Phase-1 vault cannot be used as a Phase-2 vault.
- No Phase-2 vault exists. Creating or touching one requires Samuel's explicit
  authorization and a STRATEGIC preregistration.

## Live frontier and architecture decision

The strongest external clue is Puffi's cost:

```text
1,099,802,385,000 FLOPs ~= cost of 32,768 full dense directions
32,768 = 16 Kerdock/MUB bases * 1,024 axes * 2 antipodes
```

Puffi publicly used Kerdock/Walsh mutually unbiased bases, antipodes, exact radial
integration, analytic state, and checkpoint control in Phase 1. Phase-2 code is
private, so the specific carrier is **inferred**, not confirmed. The factorization
and roughly 74x improvement over matched-row IID are nevertheless strong enough to
reopen structured sampling under the repository's own rule.

The target architecture is now:

1. 16-basis width-1024 Kerdock/MUB angular carrier;
2. antipodes and exact radial integration;
3. K2p2 analytic state reference;
4. faithful recurrent augmented-K2 joint state if it passes its own decision;
5. compact mid-network sampled-versus-analytic checkpoint response;
6. optional shared-scale/equivariant state correction only if residual diagnostics
   justify it;
7. two final nominees: a conservative carrier/reference candidate and an aggressive
   supported state/response hybrid.

Coupled residual sampling is deprioritized. Its former 1.305x projection assumed a
float32 K2 cost reduction that the frozen artifact does not show: measured cost fell
only 3.1%, not 50%.

## Evaluator blockers

Phase 0 in `GAME_PLAN.md` must fix these before promotion evidence is trusted:

1. `pyproject.toml` still pins WhestBench 0.14 / flopscope 0.10.
2. `competition_score.py` defaults to the old `1e11` lambda.
3. `lifecycle_eval.py` defaults to Phase-1 wall/residual/memory/budget semantics.
4. Local lifecycle creates raw NumPy weights, so weight-method operations such as
   `astype` are not metered like official flopscope arrays.
5. Local K2 price is low by 37,779,456 FLOPs (0.0275%) versus the official anchor.
6. The old Phase-2 ladder did not enforce the 0.4-second residual failure cliff.
7. Native 8-GiB memory and worker death/restart are not faithfully emulated.
8. Manifest/checklist and preflight can be green while exercising the stale stack.
9. Shape/GPU parity does not yet prove the 1024x16 live path.

The paired arithmetic score test, common-random-number design, ground-truth noise
reporting, seed ledger, public-CONFIRM rule, and vault discipline remain sound.

## Active execution queue

| order | slice | class | status |
|---:|---|---|---|
| 0 | official-stack/evaluator repair | IMPORTANT | **NEXT; awaiting Samuel predictions** |
| 1 | Kerdock/MUB generator, radial rule, invariants | ROUTINE/IMPORTANT | pending |
| 2 | structured carrier DESIGN | STRATEGIC on seed claim | pending |
| 3 | structured carrier DECIDE | STRATEGIC | pending |
| 4 | recurrent augmented-K2 parity and DESIGN | IMPORTANT/STRATEGIC | pending |
| 5 | recurrent augmented-K2 DECIDE | STRATEGIC | pending |
| 6 | checkpoint response DESIGN/DECIDE | STRATEGIC | pending |
| 7 | integrated frontier and robustness | STRATEGIC | pending |
| 8 | public CONFIRM / authorized Phase-2 vault | STRATEGIC | pending |
| 9 | two frozen submissions and nomination | STRATEGIC | pending |
| 10 | contribution write-up | STRATEGIC claim review | pending |

## Required predictions before Phase 0

Record Samuel's answers in `experiments/LOG.md` before changing the instrument:

1. Official-stack repair changes current K2 score by `<1%`, `1–5%`, or `>5%`.
2. Kerdock16 reduces variance versus IID at 32,768 directions by `<2.7x`,
   `2.7–20x`, or `>20x`.
3. Expected raw-MSE gain from recurrent augmented K2, plus whether any official
   residual-cap failures are expected.

Then write the IMPORTANT foundation contract/knowledge target, capture baseline
evidence, implement, run full preflight/portability/official subprocess checks, and
send raw artifacts to a fresh independent Skeptic. Do not spend a seed or edit the
submission before the relevant STRATEGIC contract.

## Closed and open lanes

| lane | state | scope |
|---|---|---|
| basic K2/K2p2 | control / conservative branch | sound but not winning alone |
| higher Gaussian covariance-map orders | low value | measured plateau near 1.052x |
| marginal ReLU mean Edgeworth | closed | capped near 1.047x in CP3R |
| recurrent augmented K2 joint state | **open** | never tested by CP3R |
| 16-basis Kerdock/MUB + radial | **highest priority** | reopened by external fingerprint |
| ordinary IID/random orthogonal | control only | tested arms lose |
| broken two-cycle lattice | closed | does not bound valid QMC/MUB |
| checkpoint response closure | **open after carrier** | compositional mechanism |
| shared-scale mixture | conditional | run only on matching residual evidence |
| equivariant learned state corrector | conditional | rollout/cross-fit required |
| dense full K3 | deferred | projected multiple budgets |
| coupled residual sampler | deprioritized | float32/headroom premise unsupported |

## Authority boundaries

- Do not edit `docs/HUMAN_MODEL.md`.
- Do not edit `estimator.py`, claim a seed block, reopen a killed lane, touch a
  vault, change a gate/residual model/protocol, or make a write-up claim without the
  required STRATEGIC sequence.
- Do not submit without Samuel's explicit approval after official validate and a
  clean Phase-2 subprocess mini with zero failures.
- After every IMPORTANT or STRATEGIC slice, obtain an independent Skeptic verdict
  from raw artifacts before believing or propagating the result.
- Keep `CURRENT_STATE.md` updated when a lane, baseline, live rule, or next action
  changes. Do not stack another override on top of this file.
