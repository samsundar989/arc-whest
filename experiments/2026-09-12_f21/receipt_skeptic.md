# F21 snapshot receipt repair: fresh-context Skeptic, 2026-09-12

Read-only review of `local_sim/f21_score.py` after commit `1521fd7`; no labels
or candidate results existed. Verdict copied from the independent agent's
file-grounded response.

## Skeptic verdict: PROVISIONAL

**Strongest challenge** The scorer cannot itself prove that
`--live-register-root` is persistent. A workerq snapshot is also a Git root;
if both that argument and `--out` point into the snapshot, the checks pass
while the receipt can later disappear.

**Evidence for it** `local_sim/f21_score.py:231` requires an absolute Git root
and an output inside its `experiments` directory, but does not distinguish the
live root from a snapshot. Read-only inspection confirmed job 1049's snapshot
passes `git rev-parse --show-toplevel`.

**Evidence for the original claim** With the actual live root supplied,
`local_sim/f21_score.py:68` derives a canonical repo-relative truth path and
passes the live receipt and ledger paths to `load_truth`.
`local_sim/truth_reads.py:276` appends `reserved` under the ledger lock before
`np.load` exposes labels; the scorer then verifies a matching `completed`
record, reservation ID, canonical path, and SHA. Both P01 halves are configured
as workerq passthroughs. P09 needs an explicit per-job passthrough for its
exact NPZ. The diff has no whitespace errors.

**Unresolved ambiguity** No exact scoring command or live receipt exists yet.
The receipt precedes the label load, although SHA checks read file bytes
beforehand. Job 1049 established snapshot source hashes and passthrough
presence only; it did not test the receipt route.

**Cheapest discriminating follow-up** Before the first protected read, inspect
the exact workerq command and manifest: snapshot-relative script and truth
paths; `--live-register-root C:\Users\samsu\Documents\arc-whest`; absolute
`--out` under that live tree; P01 passthroughs or the exact P09
`--passthrough`. After that authorized read, check the live
`experiments/truth_reads.jsonl` for matching `reserved` and `completed`
records, canonical artifact path, SHA, and output path. No new block is needed
for the path check; exposed P01 DESIGN suffices for first verification.

**Checklist items triggered** 10, 15, 17. No labels, estimator results, or
files were changed in this review.
