# P09 job 1099 failure forensics (read-only, 2026-09-13)

This note inspects the terminal workerq log/receipt and the source that ran. It
does **not** open P09 truth, rerun any P09 seed, or infer a P09 score. The scorer
at the recorded live read was `local_sim/f21_score.py` SHA256
`68b162ef20562cfed423f0105df98ad7a4f45f65058025834ba25bb446d5fedb`;
`local_sim/lifecycle_eval.py` is SHA256
`2742724e80af5e3b1324d083a5261aec0940feda111ebc0bee608c3e374e0a5c`.

## What the terminal record establishes

- Workerq 1099 ended **FAILED, exit 1** after 4h55m; its receipt reports a
  measured 5,851.27 MiB peak against 8,192 MiB requested. The process's own
  traceback, not a workerq resource error, ended the job. The log SHA256 is
  `95f322e9514f9687be5d6bf140c8947c702ce13842420d468ba627deec32bf14`
  for the finalized `workerq_1099_terminal.log` artifact;
  terminal receipt SHA256 is
  `78ede92ea62793d30629a426576976ff9ea0f4c22671ead502028dfe53b0370e`
  for `workerq_1099_terminal_receipt.json`. Earlier transient captures had
  different bytes; these are the durable raw artifact hashes.
- The log shows `128/128` for original L4, then `128/128` for fused L4. The
  scorer completes an arm's `le.evaluate()` and checks `any(r.failed)` before
  starting the next arm. Thus original L4 passed that check, and **at least
  one fused L4 row failed it**. L2 was not started. The only traceback is
  `ValueError: fused_L4 had a local non-residual lifecycle failure` at
  `f21_score.py:291`.
- `decide_P09.json` does not exist. The scorer builds its complete report and
  calls `competition_score.decide()` only after all three arms pass. Therefore
  job 1099 produced **no paired P09 score, interval, or promotion verdict**.
  The completed live truth-read receipt means P09 is exhausted; this exception
  does not authorize a replay.
- Neither the terminal log nor workerq receipt contains a failed row's index,
  seed, failure flag, error type, FLOPs, per-row timing, or MSE. The full L4
  and fused evidence lists existed only in process memory when the exception
  occurred. The log's normal cdf/pdf dtype warnings identify no failed row
  and are not an error diagnosis.

## Meaning and limits of `any(r.failed)`

`EvidenceRow.failed` is the disjunction of `budget_exhausted`,
`time_exhausted`, `residual_exhausted`, `combined_exhausted`, and a nonempty
`error_code`. The evaluator sets `error_code` to the exception *type* for a
generic prediction/materialization/validation exception; a nonfinite or
wrong-shaped prediction can reach that path. A setup error would instead raise
`SetupFailure` before per-MLP rows and would not produce this traceback.

This score run explicitly used `residual_wall_time_limit_s=None`; it therefore
could not set `residual_exhausted` through the evaluator's residual check.
Its message's “non-residual” wording is accurate only in that narrow sense.
It does **not** distinguish the remaining budget, 120-second local wall,
combined-budget, or exception paths, and it does not reveal how many rows
failed. With Phase-2 lambda zero, the combined check is FLOPs-only. The
separate `nonfinite` field is not itself part of `failed`; prediction
validation raises on nonfinite values before the row is built.

Prior engineering evidence makes a straightforward counted-FLOP overrun less
plausible but does not rule it out on P09: the measured F21 full-shape ledger
was about 928.46 billion FLOPs versus the `2**41` budget, the 24 exposed P01
DESIGN rows had zero lifecycle failures on all arms, and the separate Linux
memory gate passed 23/23. Conversely, the 120-second **local** per-MLP wall
limit remained enabled during this nearly five-hour job, so a local time
exhaustion is plausible. None of these artifacts proves which condition fired
on P09. The Linux cap and exposed DESIGN results are different tests; the
workerq process peak does not establish the failed row's cause.

## Smallest future diagnostic change

For a **new, separately preregistered fresh block only**, keep the same
`if any(r.failed): raise ValueError(...)` decision and, immediately before
that raise, emit one flushed stderr record per failed row. Include the arm,
`index`, `input_seed`, the five failure flags (including `error_code`),
`wall_time_s`, and `residual_wall_time_s`. Do not print the row's MSE,
truth floor, predictions, or score. This is a source-only reporting change:
the terminal workerq log would preserve the failing identities and category
without changing zeroing, score arithmetic, gate, or abort behavior. It cannot
recover job 1099's transient rows and must not be used to rerun P09.
