# review-slice — arc-whest

*Canonical workflow. Agent-independent. Adapters: `.claude/skills/review-slice/`,
`.agents/skills/review-slice/`.*

**The builder does not certify its own work.** This repository has twice produced a
finding with a tidy causal story that turned out to be noise, and once had four
separate claims corrected by an independent audit. This workflow is that audit, made
routine.

Skip it for ROUTINE work. Run it for every IMPORTANT and STRATEGIC slice, **before**
`next-slice` opens another one.

The review chain is fixed and is the reason this workflow exists:

```text
Builder  →  objective evaluator  →  independent Skeptic / falsifier  →  synthesis
(the slice)  (the repo's own       (a separate agent context that     (Knowledge
             scripts and raw        never saw the builder's            Delta into
             artifacts)             reasoning)                         durable docs)
```

## Step 1 — gather the raw evidence, not the summary

Read the actual artifacts. If you only have prose, say so and go get them — that gap
is itself a finding.

| evidence | where |
|---|---|
| the Slice Contract and its pre-registration | `experiments/LOG.md`, the entry dated before the data |
| commit / diff | `git log`, `git diff` — the entry should cite a SHA |
| project checks | `.\scripts\preflight.ps1` — quote the tier: `9/11` default, `11/11` with `-Full`. `local_sim/edge_cases.py` (30/30) |
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

Run the **Skeptic** role in a **separate agent context that has not seen the builder's
reasoning**. Its brief is `ai/workflows/skeptic.md`; every agent adapter binds it to a
native subagent so this is one invocation, not a copy-paste. Requirements that are not
negotiable regardless of which agent is driving:

- It receives the **Slice Contract, the diff, the evaluation protocol, and the raw
  artifact paths** — never your summary of them. A Skeptic handed prose should say so
  and ask for the artifacts; that request is itself a finding.
- It must be able to read the repository and run read-only commands, so it can
  recompute a statistic rather than take one on trust.
- It must not be the same context that produced the work. If no subagent capability is
  available at all, run it as an explicitly separate, fresh session and say in the LOG
  entry that it was run that way.

It carries this repo's failure checklist: ratio-of-means vs paired geometric mean,
one-block anomalies, counted-vs-effective units, the 7× local residual, the 18% shim
gap, spent blocks, tier violations, seed coupling, heavy-tail artefacts.

Its verdict (ACCEPT / PROVISIONAL / REJECT) goes verbatim into the LOG entry. A
PROVISIONAL verdict names the cheapest discriminating follow-up — run that before
promoting anything.

Do not soften the Skeptic's finding when relaying it, and do not defend the work
against it. If the Skeptic is wrong, show the evidence that makes it wrong.

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

Chat is not the experiment database, and the next session may be driven by a
different agent that cannot read this one. Update, where genuinely changed:

| doc | when |
|---|---|
| `experiments/LOG.md` | always — the result, the Skeptic verdict, the delta |
| `docs/CURRENT_STATE.md` | a lane opened or closed; a baseline moved; a risk resolved |
| `docs/GOTCHAS.md` | a trap was found that must not be rediscovered |
| `docs/VALIDATION.md` | the measurement protocol itself changed (rare, and STRATEGIC) |
| `docs/GAME_PLAN.md` | strategy changed — new rev, do not edit an old one |
| `docs/BENCHMARK.md` | our numbers or the field's anchors moved |
| `docs/MOONSHOT.md` | the cumulant lane advanced |

If there is no meaningful knowledge delta, say so. Do not manufacture one.

**Never edit `docs/HUMAN_MODEL.md`.** The protected-path guard blocks it under every
agent, and a git pre-commit hook blocks it even if no agent guard fired. If the result
contradicts something Samuel wrote there, name the specific line, cite the evidence,
and propose the correction for him to make.

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
PROMOTE_AND_CONTINUE   gate cleared, Skeptic ACCEPT, no regressions, evidence durable
FOLLOW_UP_EXPERIMENT   Skeptic PROVISIONAL, or a confounder needs one discriminating run
REVERT                 the change is not supported, or a regression outweighs it
STRATEGIC_RESEARCH     repo evidence is exhausted and external prior art would decide it
STRATEGIC_DECISION     evidence exists; the direction needs deeper reasoning
FIX_EVALUATION         the instrument could not judge this, and that is now the work
```

Then stop. **Do not begin another strategic implementation from inside this workflow.**
One strategic result must not flow into the next without synthesis — that rule is what
keeps this project's beliefs from drifting ahead of its evidence.

- Obvious and local next move → tell Samuel to run `next-slice`.
- `STRATEGIC_RESEARCH` or `STRATEGIC_DECISION` → emit the **Strategist Packet** in the
  format owned by `ai/workflows/next-slice.md`, and hand it to the strongest reasoning
  capability available.
- `FIX_EVALUATION` → this outranks optimization; `next-slice` will route it as
  EVALUATION_REPAIR.

## Promotion rule for this repository

Promote only when **all** hold: the pre-registered gate is cleared on a fresh
single-use block with a 95% interval excluding 1.0 · the statistic is the paired
geometric mean of per-MLP ratios · the improvement exceeds the GT-noise floor ·
`preflight -Full` **11/11** (a default `9/11` run is NOT sufficient for promotion)
and invariants 30/30 · no cost claim rests on a local absolute
residual · the Skeptic did not find a blocking flaw. "It feels cleaner" is not a
promotion criterion, and neither is a single 100-MLP run — two realisations of the
same estimator differ ~18% in the mean.
