# Draft email to the organizers — for Samuel to send from his own address

**To:** arc-whestbench@aicrowd.com
**Subject:** Clarification request: status of residual-wall-time-only computation under the cost model

> Notes for Samuel before sending:
> - Send from your own address; this is a participant asking a rules question.
> - Tone is deliberately neutral and technical. It names no participants and makes
>   no accusation — it asks which of two readings of the cost model is correct.
> - We have **not** posted this or anything about it on the forum, and should not.
> - The two forum threads referenced are public and already raised by others, so
>   pointing at them adds no new information to the world.

---

Hello,

I'm competing in the ARC White-Box Estimation Challenge (participant
samuel_sundararaman). I'd like to ask a clarification about the compute model,
because it affects how we choose to implement our estimator and I'd rather build
against the intended reading than guess.

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
from the final ranking, not a request for any change to our own scores. Happy to
provide details of anything we've measured if that would be useful.

Thanks for your work on the challenge,

Samuel Sundararaman
