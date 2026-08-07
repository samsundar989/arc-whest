---
name: review-slice
description: Close the epistemic loop on a completed IMPORTANT or STRATEGIC slice in arc-whest. Inspects raw evidence rather than the builder's narrative, runs the independent skeptic agent, writes the Knowledge Delta into durable docs, quizzes Samuel on mechanism, and ends in exactly one terminal state. Run after any slice before starting another strategic change.
---

# /review-slice — arc-whest

**The builder does not certify its own work.** This repository has twice produced a
finding with a tidy causal story that turned out to be noise, and once had four
separate claims corrected by an independent audit. This skill is that audit, made
routine.

Skip it for ROUTINE work. Run it for every IMPORTANT and STRATEGIC slice, **before**
`/next-slice` opens another one.

## Step 1 — gather the raw evidence, not the summary

Read the actual artifacts. If you only have prose, say so and go get them — that gap
is itself a finding.

| evidence | where |
|---|---|
| the Slice Contract and its pre-registration | `experiments/LOG.md`, the entry dated before the data |
| commit / diff | `git log`, `git diff` — the entry should cite a SHA |
| project checks | `.\scripts\preflight.ps1` (5/5), `local_sim/edge_cases.py` (30/30) |
| raw evaluator output | JSON under `experiments/`, `experiments/run_reports/`, `experiments/packed/` |
| seeds / block / config | which block, design or decision, n MLPs, n shifts, N, truth samples |
| GT-noise floor | quoted alongside every MSE, per `docs/VALIDATION.md` §3.5 |
| vault state | counter vs `touch_log.jsonl` line count |

**The pre-registration check comes first and is disqualifying.** Compare the gate,
the residual model, and the statistic as *written before the block was touched*
against what was reported. Any movement in either direction invalidates the decision,
regardless of the result. If a single-use block (`seed_blocks.SINGLE_USE_BLOCKS`)
appears in an earlier LOG entry, this entry is not evidence.

## Step 2 — independent falsification

Launch the **`skeptic` agent** with the contract, the diff, the evaluation protocol,
and the **raw artifact paths** — never with your own summary. It carries this repo's
failure checklist: ratio-of-means vs paired geometric mean, one-block anomalies,
counted-vs-effective units, the 7× local residual, the 18% shim gap, spent blocks,
tier violations, seed coupling, heavy-tail artefacts.

Its verdict (ACCEPT / PROVISIONAL / REJECT) goes verbatim into the LOG entry. A
PROVISIONAL verdict names the cheapest discriminating follow-up — run that before
promoting anything.

Do not soften the skeptic's finding when relaying it, and do not defend the work
against it. If the skeptic is wrong, show the evidence that makes it wrong.

## Step 3 — answer these ten, in the output

1. Was the **prediction** supported? Compare to what was written before the data.
2. Was the claimed **causal mechanism** supported — or only the sign? A mechanism
   that predicts direction but not magnitude has not been demonstrated.
3. What is the **strongest competing explanation**?
4. Were there **regressions or tradeoffs**? Preflight, invariants, cost, runtime,
   memory, subprocess behaviour.
5. What **new information** do we have?
6. Which belief **strengthens**?
7. Which belief **weakens**?
8. What remains **uncertain**?
9. Did we find a durable **invariant or failure mode**? (If it can be tested, it
   belongs in `local_sim/edge_cases.py`; if it can only be remembered, `docs/GOTCHAS.md`.)
10. What should happen **next**?

## Step 4 — write the Knowledge Delta into durable state

Chat is not the experiment database. Update, where genuinely changed:

| doc | when |
|---|---|
| `experiments/LOG.md` | always — the result, the skeptic verdict, the delta |
| `docs/CURRENT_STATE.md` | a lane opened or closed; a baseline moved; a risk resolved |
| `docs/GOTCHAS.md` | a trap was found that must not be rediscovered |
| `docs/VALIDATION.md` | the measurement protocol itself changed (rare, and STRATEGIC) |
| `docs/GAME_PLAN.md` | strategy changed — new rev, do not edit an old one |
| `docs/BENCHMARK.md` | our numbers or the field's anchors moved |
| `docs/MOONSHOT.md` | the cumulant lane advanced |

If there is no meaningful knowledge delta, say so. Do not manufacture one.

**Never edit `docs/HUMAN_MODEL.md`.** A hook blocks it. If the result contradicts
something Samuel wrote there, name the specific line, cite the evidence, and propose
the correction for him to make.

## Step 5 — teach-back, at most four questions

Test **mechanism, critical path, evidence quality, remaining uncertainty, and likely
failure behaviour**. Never implementation trivia. Good shapes for this project:

- Why did this work, or fail, at the level of mechanism?
- Which component was most causally important, and how do we know?
- What would have to be true for the strongest competing explanation to be right?
- If this number regressed tomorrow, where would you look first?
- Does this change what we expect on the private re-run, or only on this machine?

If Samuel cannot answer: teach that one concept using this repository's real files and
one real number, ask a single transfer question, then continue. This is a
comprehension gate, not an exam.

## Step 6 — end in exactly one state

```text
PROMOTE_AND_CONTINUE   gate cleared, skeptic ACCEPT, no regressions, evidence durable
FOLLOW_UP_EXPERIMENT   skeptic PROVISIONAL, or a confounder needs one discriminating run
REVERT                 the change is not supported, or a regression outweighs it
STRATEGIC_RESEARCH     repo evidence is exhausted and external prior art would decide it
STRATEGIC_DECISION     evidence exists; the direction needs deeper reasoning
FIX_EVALUATION         the instrument could not judge this, and that is now the work
```

Then stop. **Do not begin another strategic implementation from inside this skill.**
One strategic result must not flow into the next without synthesis — that rule is what
keeps this project's beliefs from drifting ahead of its evidence.

- Obvious and local next move → tell Samuel to run `/next-slice`.
- `STRATEGIC_RESEARCH` or `STRATEGIC_DECISION` → emit the **Strategist Packet** in the
  format owned by the `next-slice` skill, and hand it to the stronger reasoning model.
- `FIX_EVALUATION` → this outranks optimization; `/next-slice` will route it as
  EVALUATION_REPAIR.

## Promotion rule for this repository

Promote only when **all** hold: the pre-registered gate is cleared on a fresh
single-use block with a 95% interval excluding 1.0 · the statistic is the paired
geometric mean of per-MLP ratios · the improvement exceeds the GT-noise floor ·
`preflight` 5/5 and invariants 30/30 · no cost claim rests on a local absolute
residual · the skeptic did not find a blocking flaw. "It feels cleaner" is not a
promotion criterion, and neither is a single 100-MLP run — two realisations of the
same estimator differ ~18% in the mean.
