# Draft email to the organizers — for Samuel to send from his own address

**To:** arc-whestbench@aicrowd.com
**Subject:** Two clarification requests: residual-wall-time computation, and the `sampling_mse` field in score reports

> Notes for Samuel before sending:
> - Send from your own address; this is a participant asking rules questions.
> - Tone is deliberately neutral and technical. It names no participants and makes
>   no accusation — each question asks which of two readings is correct.
> - We have **not** posted this or anything about it on the forum, and should not.
>   That applies with extra force to question 2: it is a question about the
>   evaluation machinery, and asking it privately is the whole point.
> - The two forum threads referenced in question 1 are public and already raised by
>   others, so pointing at them adds no new information to the world. Question 2
>   references only a field that appears in **our own** score reports.
> - Question 2 deliberately stops at "what does this field mean and is the suite
>   fixed". It draws no inference about anyone's score, and it should not — the
>   arithmetic that motivates it is ours to keep.

---

Hello,

I'm competing in the ARC White-Box Estimation Challenge (participant
samuel_sundararaman). I have two clarification questions — one about the compute
model and one about a field in the score report — because both affect how we
implement and calibrate our estimator, and I'd rather build against the intended
reading than guess.

**1. Computation charged only through residual wall time.**

Threads 18099 and 18108 describe computation performed on raw NumPy-backed arrays
obtained through `flopscope.numpy`, which does real arithmetic but reports no
instrumented FLOPs, so it is charged only through the residual wall-time term
`lambda * R_m`. Since lambda = 1e11 FLOP/s is well below the effective throughput
of vectorised CPU code, that route can be substantially cheaper than the same
arithmetic performed through the instrumented API. Could you clarify whether this
is considered within the rules as intended, or whether it is a cost-model defect
of the same kind as the batched-RHS `linalg.solve` undercount reported in thread
18082?

If it is regarded as a defect, could you also say whether affected submissions
would be re-evaluated under the corrected model before the final Phase 2 ranking?
I ask because of the commitment made when the `solve` issue was fixed — that
"once the fix is in, the affected submissions will automatically be evaluated
using the patched flopscope" — and because Phase 1 submissions were re-scored on
the updated cost model after the 0.9/0.10 changes. Knowing whether the same
treatment applies here would tell us whether to plan around the current numbers or
the corrected ones.

We're building our estimator entirely through the instrumented API and intend to
keep doing so regardless of the answer; this is a question about what to expect
from the final ranking, not a request for any change to our own scores.

**2. What `sampling_mse` denotes, and whether the reference means are estimates.**

Every score report we have received carries a field `sampling_mse`, and its value
is identical to full precision across all four of our submissions, which have used
quite different estimators. That it does not vary with our predictions suggests it
describes the evaluation reference rather than our output, but the field is not
covered in the score-report documentation, so I would rather ask than infer.
Specifically:

  a. What does `sampling_mse` measure?

  b. Are the per-neuron reference means that submissions are scored against exact,
     or are they Monte Carlo estimates with their own finite-sample error? The
     dataset card documents 1e9 samples per MLP for the *public* ground truth; I
     could not find the corresponding statement for the evaluation references, and
     the two need not be the same. If they are estimates, is the reported
     `final_layer_mse` the error against those estimates, or is it corrected for
     their sampling noise?

  c. Are the evaluation MLPs and their reference means fixed across submissions
     within a round — i.e. does the same suite get reused for every submission a
     team makes — or are they redrawn per submission? From our own four reports
     the `per_mlp` MLP names are identical, in the same order, so it looks fixed,
     and the public score is the mean over the 50 MLPs (indices 0–49) whose
     individual `final_layer_mse` values come back in `public_scores`. I mention
     it because a fixed scored set combined with exact per-MLP feedback each
     submission is a stronger form of feedback than a leaderboard usually gives,
     and if that combination is not deliberate you would probably want to know.

I ask (b) and (c) purely to calibrate our own development loop. We use the public
1e9-sample ground truth offline, and our local-versus-graded agreement has been
good, so the practical question for us is how much of the remaining difference is
reference noise versus a real difference in our estimator, and how much weight to
put on a single graded run when deciding whether a change helped. If any of this
is intentionally undocumented, that's a fine answer and I won't press further or
repeat it publicly.

Happy to provide details of anything we've measured if that would be useful.

Thanks for your work on the challenge,

Samuel Sundararaman
