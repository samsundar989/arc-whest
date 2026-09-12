# F21 pre-score independent Skeptic, 2026-09-12

Fresh-context read-only review of the committed contract, parent/candidate diff,
tiny oracle, trajectory probe and score runner. No candidate result or truth read
existed at review time. This is a protocol review, not promotion acceptance.

## Skeptic verdict: PROVISIONAL

**Strongest challenge** A workerq snapshot can discard the single-use P09
truth-read receipt if `--live-register-root` points to the snapshot. The scorer
checks that the path is absolute and contains registries, but cannot prove it
is the persistent workspace.

**Evidence for it** `local_sim/f21_score.py:201` accepts that root;
`local_sim/truth_reads.py:276` writes the reservation there before exposing
labels. Its relative artifact path is computed from the executing module's
root (`truth_reads.py:178`), which also makes a snapshot-produced receipt path
misleading even when the registry itself is live.

**Evidence for the original claim** The L4 archive member and staged parent
are byte-identical, SHA256 `052dd768…`; the candidate is SHA256 `d77dab0e…`;
`parent_to_f21.diff` exactly matches their source diff. Midpoint truth floors
and official-precision scaling are consistent (`f21_score.py:106,134,155`).
The binding comparison uses paired arithmetic forecast with the common server
offset and preregistered `decide(gate=1.05, interval="t")`
(`f21_score.py:164-167,276-289`). `lifecycle_eval.py:561-693` supplies
canonical mints and grader-style `f64_readonly` weights.

**Unresolved ambiguity** No strategic result exists. The trajectory probe
uses float32 input weights (`test_p2_d21_fusion_trajectory.py:149`), so its
pass alone cannot establish the grader's float64-input cast or full metered
lifecycle; the scorer and separate deployment gates must do that.

**Cheapest discriminating follow-up** Before P09 is read, check the exact
workerq invocation and a no-label workerq path/hash probe. Run the single-use
scorer with `--no-snapshot`, or establish that both its registry and receipt
paths resolve to the persistent workspace. No seed block is needed.

**Checklist items triggered** 1, 4, 10, 13, 14, 15, 16, 17.

**Builder follow-up status (separate from Skeptic verdict):** no-label workerq
job 1045 failed because PowerShell removed an inner quote from its `-c`
argument. The tracked no-label script `local_sim/f21_snapshot_hash.py` then
ran as workerq job 1049 on a Git snapshot and passed all three candidate,
parent and L2 SHA256 checks. The receipt-path concern remains until the
scorer's persistent-register route is repaired and verified.
