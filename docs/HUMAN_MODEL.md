# HUMAN_MODEL.md — Samuel's mental model

> **This file belongs to Samuel. Agents may read it, quiz against it, point out
> contradictions with the evidence, and *propose* corrections in chat. Agents must
> never write to it.** A `.claude/settings.json` hook denies Edit/Write on this path.
>
> It records what Samuel actually believes right now — including where that turns
> out to be wrong. A file that agents keep "correcting" stops measuring anything.

*Created 2026-08-07 (harness audit) as an empty template. Answer from memory,
without re-reading the docs first — that is the whole point. Rough answers and
"I don't know" are both useful; "I don't know" is a request for a 3-minute
explanation, not a failure.*

---

## Objective

*What is this project trying to optimize, in one sentence — and what is the
difference between the number on the leaderboard and the number that decides the
prize?*

>

## The system from memory

*Sketch the path from `mlp.weights` to a score. Roughly five boxes. Do not look.*

>

## Current bottleneck

*What do I believe most limits progress right now?*

>

## Why

*What evidence makes me believe that? Which measurement, on which tier of data?*

>

## Current strongest hypothesis

*What do I think would unlock the next real gain — on the score lane or the
contribution lane?*

>

## Dangerous assumptions

*Three things that, if false, would invalidate work we've already banked.*

1.
2.
3.

## Things I do not understand yet

*Name them plainly. Each one becomes a 3-minute teach-back, not a lecture.*

1.
2.
3.

## Current failure model

*What are the most likely ways this entry fails on the private re-run? Rank them.*

>

## Next experiment

*I expect ______ because ______.*

>

## Confidence

*What am I confident about vs uncertain about?*

>

---

## Cold-start check — the questions this project actually turns on

Run this after ~3–5 strategic slices, after a phase change, or at the end of a
focused day. Answer without reading generated summaries. These are drawn from the
real critical path, not from a generic template.

**Level 4 — the things that let you say "no" to a bad experiment**

1. The score is `MSE × max(0.1, C/B)`. Why does the number of samples `N` mostly
   cancel out of it, and what does that mean about de-noising a sample versus
   making a sample cheaper?
2. Why is the public mini split read-only now? What specifically goes wrong if a
   constant is fitted on it?
3. What is a single-use decision block, and what would be lost if D14 were rerun
   with a different threshold?

**Level 3 — the things that let you spot a false result**

4. Local residual wall time is ~7× the grader's. Why does that make a cost claim a
   *model* rather than a measurement, and why does the deciding rule take the `max`
   of two models rather than the better one?
5. The fresh-MLP score distribution has max/median ≈ 12.9×. What does that imply
   about a 2% improvement in the mean?
6. Two "findings" this project made turned out to be noise wearing a mechanism.
   What did they have in common, and what rule now catches that class?

**Level 2 — the shipped method**

7. Trace one `predict()` call at a high level: what does covariance propagation
   compute, what does the calibration table fix, and what is the rotation for?
8. Layer 1 is exact to 1.8e-16 while layer 30 is not. Why is layer 1 special?

**If the answer is "I don't know":** that concept gets explained using this
repository's own files and one real execution trace, then work resumes. Nothing
here is an exam.
