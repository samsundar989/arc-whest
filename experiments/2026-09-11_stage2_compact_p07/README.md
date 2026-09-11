# Compact Stage-2 P07 coefficient producer

This directory is reserved for the registered label-free four-value `lam` producer
that follows the accepted compact-arm gate at commit `f3e453e`. The producer uses
the committed P07 identity list and writes four immutable coherent-arm prediction
sets without reading labels.

The workerq job uses the live working bytes (`--no-snapshot`) because the evidence
contract binds exact repository bytes and the queue's git snapshot can normalize
line endings. Its output directory is `observations/` and must not exist before
the job starts. The declared envelope is 8 GiB RAM, 2 CPUs and no GPU. This is a
batch-process allowance for 1,536 lifecycle rows; one compact prediction has
measured about 1.0--1.2 GiB process RAM and 345,587,712 bytes of accounted peak
array payload.

On success, `predictions.npz` and `producer_receipt.json` must pass the standalone
truth-free verifier before the next registered protected P07 access. A queued or
partially written job is not accepted evidence.

Workerq job **891** was submitted at high priority with the declared envelope
above. It uses no source snapshot and was queue position zero at submission,
waiting because two running jobs already reserved 46 GiB of the queue's 53.6 GiB
usable RAM. This is an admission wait, not producer progress or completion.
