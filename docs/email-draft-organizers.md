# Draft email to the organizers — for Samuel to send from his own address

> **HISTORICAL — DO NOT SEND.** This August 6 draft predates Rules v14, the live
> Phase-2 round and the two graded submissions. Its “one designated submission”
> premise and several dates/questions are obsolete. Preserve it only as an audit
> record; any future organizer message must be drafted from current evidence.

**To:** arc-whestbench@aicrowd.com
**Subject:** Final-ranking mechanics, a scoring-channel issue, and a cost-model ruling request

> **Historical status at 2026-08-06: it was then marked final. It is not ready to
> send now, and none of it goes on the forum.**
>
> **What changed and why**: the λ / raw-array question was drafted as a request for
> a ruling, and then the crawl found the organizers had already ruled, twice —
> thread 18122 ("on the eval servers you have flopscope-client … you won't be able
> to do operations directly on the array that bypass the accounting boundary") and
> thread 18132 ("we will examine this during the private re-evaluation and prize
> due diligence … using residual pricing as an alternative arithmetic budget is
> not [legitimate]"). Sending the original text would have asked for something
> already given. Paragraph 5 is now an acknowledgement plus our own instrumented
> share (98.2%), volunteered because 18132 says that number gets examined anyway.
> A new paragraph 0 raises the Aug-7-vs-Aug-10 deadline contradiction, which is
> genuinely unanswered and genuinely time-critical.

> Notes for Samuel before sending:
> - Send from your own address; this is a participant asking about rules and
>   reporting one technical observation.
> - Tone is deliberately neutral. It names no participant, describes no one's
>   submissions, and makes no accusation — question 4 reports a property of the
>   evaluation machinery, which is a thing about *their* system, not about anyone
>   using it.
> - **This does not go on the forum**, and neither does anything behind it. If
>   they reply asking for detail, send detail; if they don't reply, we do nothing
>   further and it stays in our private appendix.
> - Everything asserted in question 4 is checkable in our own four score reports;
>   nothing in it required looking at anyone else's data.
> - Q4 deliberately stops at "here is the mechanism and here are the standard
>   fixes". It does not say anyone has used it, because we have not established
>   that and it is not our place to.
> - Q5 is back in as its own paragraph because it is the one integrity question
>   the private re-run does **not** settle. A leaderboard artifact disappears when
>   the ranking is decided on fresh MLPs; a billing discount does not, because it
>   is a property of how work is priced, not of which MLPs are used. Both threads
>   are public and were raised by others, so pointing at them adds nothing to the
>   world.

---

Hello,

I'm competing in the ARC White-Box Estimation Challenge (participant
`neopok919`). One time-sensitive discrepancy, three questions about how the final
ranking is decided, and one technical report about the scoring channel that I
think you'll want to see.

**0. One operational thing first**, because it may be time-critical: the update in
thread 18125 gives the Phase 1 deadline as 10 August 23:59 UTC, but the challenge
page's own round object still ends Phase 1 at **2026-08-07 23:59 UTC** and starts
Phase 2 at 2026-08-08 00:00 UTC. Thread 18130 raised the same discrepancy on
5 August and hasn't had a reply. Which one governs submission acceptance? We are
planning around the earlier of the two to be safe, but teams reading only the
announcement may find the round closed under them.

**1–2. The private re-run.** My understanding is that prizes are decided by
re-running each team's *one designated submission* on a fresh set of MLPs the
submission has never seen, rather than by public-leaderboard position — thread
18125 says as much. Could you confirm how the designation works (default to
best-public-score, or an explicit choice by the team), where that interface is,
and by when it must be done? Thread 18130 asked this on 5 August and is still
open. Relatedly,
the townhall said the re-run set would be "finalized closer to the time" — could
you say what is currently expected for its **size and architecture**? We are
spending effort on making our estimator robust across MLP draws rather than tuned
to any particular one, and the number of re-run MLPs and their width/depth are
exactly what set the target: robustness to a 10-MLP re-run and to a 1000-MLP
re-run are different engineering problems, and a shape change would mean refitting
before we harden anything.

**3. The contribution prize.** The guidelines describe a screening pool of roughly
the top ten entries. Is that pool drawn from the **public leaderboard** standing or
from the **private re-run**? It affects how we budget the remaining weeks between
score work and the write-up, and I'd rather ask than guess.

**4. A property of the scoring channel that I think is worth reviewing.** From our
own four score reports: the public score is the mean over a **fixed set of 50 MLPs**
(indices 0–49, identical names in identical order across all four submissions), and
every submission returns those MLPs' **individual** `final_layer_mse` — plus a
32-element `per_layer_mse` array each — as full-precision floating-point values in
`results.public_scores`. With a 50-submissions-per-day allowance, that combination
is stronger feedback than it may look. Each returned per-MLP value is
‖p − t‖²/256 for the submitted prediction p and the hidden reference t; differencing
the values from two submissions cancels the quadratic terms and leaves a *linear*
equation in t's 256 unknowns, supplied for all 50 scored MLPs at once. About **257
submissions therefore determine every hidden reference vector exactly** — roughly
five days at the current cap — with no modelling involved. This is the standard
reconstruction-from-aggregates phenomenon (Dinur & Nissim, PODS 2003), and the
adaptive-overfitting literature covers the leaderboard case specifically: Blum &
Hardt, *The Ladder: A Reliable Leaderboard for Machine Learning Competitions*
(arXiv:1502.04585), and Dwork et al., *The reusable holdout* (arXiv:1506.02629).

The usual mitigations are cheap and well understood: report a **single rounded
aggregate scalar** rather than per-item, per-layer real values; adopt
**Ladder/Thresholdout-style reporting**, where a submission's score is only updated
when it improves by more than a stated threshold; and/or **rotate the scored set**
so the public feedback is never about a fixed hidden vector. I'm raising it because
it bears on the integrity of the public board and, if the top-K contribution-prize
screening is taken from that board, on the screening too. To be clear about our own
position: we have not used this and will not — our estimator is a sampling method
scored offline against the public 1e9-sample ground truth — and I'm not asking for
any change to our scores. Happy to send the exact field names, our four reports, or
a worked example if that's useful.

**5. Residual-time billing — noted, and one volunteered number.** I had a question
drafted here about whether computation charged only through the residual term was
within the intended cost model (threads 18099 and 18108). Your replies in 18122
and 18132 answer it, so I'll just say I've read them and that we agree with the
line drawn — "optimizing ordinary control flow and data handling is legitimate,
using residual pricing as an alternative arithmetic budget is not." Since you
mentioned instrumented share will be examined during prize due diligence, here is
ours up front rather than on request: on submission 324420 the per-MLP telemetry
gives 5.31e10 instrumented FLOPs against 9.84e-3 s of residual wall time, i.e.
**98.2% of our billed compute is instrumented**, and the residual is the ordinary
cost of allocating and freeing sampling buffers. Our estimator is Sobol RQMC with
analytic dead-neuron pruning, built entirely through `flopscope.numpy`; happy to
walk through it if that is ever useful.

Thanks for your work on the challenge,

Samuel Sundararaman
