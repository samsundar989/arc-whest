# cold-start — arc-whest

*Canonical workflow. Agent-independent. Adapters: `.claude/skills/cold-start/`,
`.agents/skills/cold-start/`.*

**Ask before you tell.** The value is entirely in the gap between what Samuel says
from memory and what the repository shows. Revealing the answers first destroys the
measurement.

Run this every 3–5 strategic slices, after a phase change or major architecture
change, after several days away, or whenever he says he is losing the thread.

## Step 1 — ask, and stop

Present the nine questions below and **wait**. Do not preview answers, do not hint,
do not read files aloud first. Say explicitly that "I don't know" is a useful answer
and rough answers are fine — this is a diagnostic, not an exam.

Prefer the eight project-specific questions already written in
`docs/HUMAN_MODEL.md` § *Cold-start check* — they were drawn from the real critical
path. Map them onto the standard nine:

1. **What are we optimizing?** (And what is the difference between the leaderboard
   number and the number that decides the prize?)
2. **How does the strategically relevant system work?** Roughly five boxes, from
   `mlp.weights` to a score.
3. **What is the critical execution and evaluation path?** Where does a prediction
   come from, and what judges it?
4. **What is the current baseline?** Which tier is it from?
5. **What is the current bottleneck?**
6. **What assumptions and invariants matter most?** Name three that, if false, would
   invalidate banked work.
7. **What did the last important experiments establish?** Sprint 5 / D14; Moonshot G0.
8. **What is our strongest current hypothesis?**
9. **What is the next experiment, and why that one?**

## Step 2 — compare against repository evidence

Only after his answers are in. Check against `docs/CURRENT_STATE.md`, the tail of
`experiments/LOG.md`, `docs/GOTCHAS.md`, and the code itself.

Classify each answer:

```text
CORRECT           — matches the evidence
DIRECTIONALLY OK  — right shape, imprecise; usually fine, do not nitpick
STALE             — was true, superseded (e.g. a lane that has since been killed)
WRONG             — contradicted by evidence, and it matters
MISSING           — "I don't know"
```

Note that **stale beliefs are the most dangerous class here.** This project kills
lanes fast; a belief that the packed lane is still promising, or that score
optimization is still open, would misdirect real work.

## Step 3 — teach only the gaps that matter

Prioritize ruthlessly. A gap earns teaching time only if it affects:

- **strategic direction** — what to work on next;
- **diagnosis** — reading a result correctly;
- **evaluation judgment** — knowing whether a number means anything;
- **ability to challenge an agent's proposal** — the most valuable of the five;
- **ability to form a novel hypothesis.**

Do **not** teach: flopscope shim internals, the GPU mirror, board tooling, the
packed-sparse lab, or anything in `estimator.py` below the level of "what this stage
does and why."

The gaps most likely to matter in this repository, with where to teach them from:

| gap | teach from | why it matters |
|---|---|---|
| the `V·c/B` identity; why `N` cancels | `docs/COMPETITION_BRIEF.md`, `estimator.py:439 _cost_per_sample` | without it, every cost/variance tradeoff looks arbitrary |
| why CONFIRM is read-only | `docs/VALIDATION.md` §0 | this is the rule most likely to be broken by a plausible-sounding request |
| single-use decision blocks | `local_sim/seed_blocks.py:78` | explains why we cannot "just re-run D14 with a lower gate" |
| local residual is ~7× the grader's | `docs/GOTCHAS.md` | explains why cost claims are models, and why the deciding rule takes a `max` |
| the heavy tail, max/median ≈ 12.9× | `docs/CURRENT_STATE.md` § risks | explains why a 2% mean improvement is not obviously worth anything |
| noise wearing a mechanism | LOG: antithetic 12%, D12's 1.2079× | the failure mode he most needs to be able to catch in an agent's proposals |

Teach one concept at a time, using a real file and a real number, then ask **one
transfer question** — a nearby case, not a repetition. Move on.

## Step 4 — propose, do not write

End with a specific list of what Samuel should personally add to or change in
`docs/HUMAN_MODEL.md`: which slots, and what the correction is about — not drafted
prose for him to paste. **Do not edit that file**; the protected-path guard blocks it
under every agent, and the file is only worth something if the words are his.

Also report the comprehension-debt signals plainly, since they are the reason to run
this at all:

```text
unsynthesized strategic slices
stale beliefs found
unresolved HUMAN_MODEL gaps
claimed improvements never reproduced
open doubts about the evaluator
critical assumptions with no test
repeated agent mistakes not yet promoted into the harness
```

If several trend upward while output accelerates, say so directly and recommend
pausing new strategic slices until they are worked down. That recommendation is the
whole point of the exercise.
