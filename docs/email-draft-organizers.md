# Draft email to the organizers — for Samuel to send from his own address

**To:** arc-whestbench@aicrowd.com
**Subject:** Final-ranking mechanics, and a scoring-channel issue worth a look

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

---

Hello,

I'm competing in the ARC White-Box Estimation Challenge (participant
`neopok919`). Three questions about how the final ranking is decided, and one
technical report about the scoring channel that I think you'll want to see.

**1–2. The private re-run.** My understanding is that prizes are decided by
re-running each team's *one designated submission* on a fresh set of MLPs the
submission has never seen, rather than by public-leaderboard position. Could you
confirm that, and confirm how the designation works in Phase 2 (default to
best-public-score, or an explicit choice by the team, and by when)? Relatedly,
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

(Separately, and lower priority since it is already public: threads 18099 and 18108
asked whether computation performed on raw NumPy-backed arrays — charged only
through the residual wall-time term — is within the intended cost model or a defect
of the same kind as the batched-RHS `linalg.solve` undercount in 18082. We build
entirely through the instrumented API either way, but an answer would tell us
whether to expect a re-evaluation before the final ranking.)

Thanks for your work on the challenge,

Samuel Sundararaman
