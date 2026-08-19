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

> We are trying to optimize the error rate of our prediction algorithm of the neural net given the weights. The leaderboard number is misleading as some times have optimized to the neural net weights iteratively, but will fail on the fresh set of nets the final submissions are graded on. 

## The system from memory

*Sketch the path from `mlp.weights` to a score. Roughly five boxes. Do not look.*

> I don't know 

## Current bottleneck

*What do I believe most limits progress right now?*

> Understanding the mathematics/best frontier architecture for approaching this problem. 

## Why

*What evidence makes me believe that? Which measurement, on which tier of data?*

> Without optimizing specific to the public neural nets, our position on leaderboard is top 100, the best model should be able to predict close/past the reasonable mathematical limit which I don't believe we've hit. 

## Current strongest hypothesis

*What do I think would unlock the next real gain — on the score lane or the
contribution lane?*

> Evaluating our current architecture and finding a way to improve our estimation without consuming more compute, so the score lane. 

## Dangerous assumptions

*Three things that, if false, would invalidate work we've already banked.*

1. We already have the best architecture to solve this problem. 
2. We fully understand the mathematics and the physical limitation of this problem. 
3. We can beat the over-trained/fitted submissions on the leaderboard with the best possible model (not over trained). 

## Things I do not understand yet

*Name them plainly. Each one becomes a 3-minute teach-back, not a lecture.*

1. How our system actually works
2. What the best teams/frontier is doing
3. The mathematics and what the physical limit of this problem is. 

## Current failure model

*What are the most likely ways this entry fails on the private re-run? Rank them.*

> By over training/over fitting to the public neural nets the leaderboard is set against. 

## Next experiment

*I expect ______ because ______.*

> I expect a higher position on the leaderboard because we improve our system to more accurately predict the behavior of the neural nets without consuming more compute. 

## Confidence

*What am I confident about vs uncertain about?*

> That we are on the right track, as we are currently in the top 100, but I'm not sure we know how to improve from here without over-fitting. 

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
