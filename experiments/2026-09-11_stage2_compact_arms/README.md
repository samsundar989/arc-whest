# Stage-2 compact-arm gate evidence

Jobs 878--883 were the first real-shape attempt from commit `bb83897`. They are
preserved as diagnostic evidence only. Job 878 completed the exact row with
1,126.7 MiB measured peak RAM, but a subsequent independent audit found that the
gate did not hold and recheck source bytes through publication. Jobs 879 and 880
completed computation but could not publish after the untracked destination
directory was temporarily absent. Jobs 881--883 were cancelled after the audit;
job 882 had already written a reference ledger, which is explicitly named as a
cancelled-after-write artifact. None of these jobs satisfies the compact-arm
acceptance gate.

`gate_conformance.json` is the tiny truth-free positive/refusal suite for the
repaired gate. Jobs 884--888 are the repaired real-shape arm/reference runs.
Job 889 is preserved negative evidence: its git snapshot normalized tracked line
endings, so the source/artifact hashes correctly failed. Job 890 reran the default
preflight against the actual live working bytes and passed 12/16 with the four
Full-only deployment steps skipped. `acceptance_summary.json` binds the results,
source hashes and the explicit worker telemetry limitation.
