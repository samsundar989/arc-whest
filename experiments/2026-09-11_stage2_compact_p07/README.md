# Compact Stage-2 P07 coefficient producer

This directory is reserved for the registered label-free four-value `lam` producer
that follows the accepted compact-arm gate at commit `f3e453e`. The producer uses
the committed P07 identity list and writes four immutable coherent-arm prediction
sets without reading labels.

The workerq job uses the live working bytes (`--no-snapshot`) because the evidence
contract binds exact repository bytes and the queue's git snapshot can normalize
line endings. Its output directory is `observations/` and did not exist before
the job started. The accepted computation is 1,536 lifecycle rows; one compact
prediction had measured about 1.0--1.2 GiB process RAM and 345,587,712 bytes of
accounted peak array payload. Whole-batch process memory is a separate quantity
and must use workerq's final measured peak.

On success, `predictions.npz` and `producer_receipt.json` must pass the standalone
truth-free verifier before the next registered protected P07 access. A queued or
partially written job is not accepted evidence.

Workerq job **891** was submitted at high priority with the declared envelope
above, then cancelled while still queued after an independent audit found an
unbound imported selector dependency. It never started and created no output
directory. Repair and fresh acceptance of that provenance gap precede the next
submission; job 891 is not producer evidence.

The repaired production path passed 97/97 scoped checks, current-byte default
preflight in workerq 893 and a new fresh-context Skeptic. Jobs 894 and 900 were
cancelled while queued and are not evidence. Workerq job **903** is the actual
accepted-byte run at high priority with 2 CPUs, no GPU and no source snapshot.
It was submitted with 5 GiB RAM after the old 32 GiB/8 GiB reservations were
reduced, but its live measured peak exceeded that declaration while ample system
headroom remained. The final safe declaration is therefore pending the terminal
peak; 6 GiB is only a lower bound. Job 903 has no result until it terminates and
its receipt passes standalone verification. Its partial `observations/` directory
must not be inspected or committed.
