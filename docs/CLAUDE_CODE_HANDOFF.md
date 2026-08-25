# Claude Code handoff — close P1-PRE, calibrate Phase 2, start Phase 1A

Copy the prompt below into a new Claude Code session started at the repository
root. It assumes no chat history. Repository state is authoritative if a SHA or
fact has changed since this handoff was written.

---

## Prompt

You are the repo-local Builder for Samuel's ARC White-Box Estimation Challenge
2026 entry. Work from:

```text
C:\Users\samsu\Documents\arc-whest
```

Continue autonomously through safe work, but stop at every Samuel-only prediction,
vault, submission or material scope-expansion gate. Never submit or touch a vault
without Samuel's explicit approval.

First read `AGENTS.md`, then `docs/CURRENT_STATE.md`, the bottom of
`experiments/LOG.md`, `docs/GAME_PLAN.md`, `docs/VALIDATION.md`,
`docs/GOTCHAS.md`, and `docs/SKEPTIC.md`. Inspect the worktree and current HEAD
before editing. At the handoff decision, HEAD was
`8e58db09a12e054153d0fb9ea6cadf1ef2aafa59`, the tree was clean, and
`estimator.py`, the seed ledger and the vault were untouched throughout P1-PRE.

## What is true now

- Phase 2 is live at width 1024, depth 16, `B = 2**41`; WhestBench 0.16 /
  flopscope 0.12; lambda 0; residual cap 0.4 s; wall cap 120 s; memory class
  default 8192 MB.
- Phase 0 is CLOSED on a literal independent ACCEPT. The local counted-FLOP
  meter matches the official runner op by op.
- The signed-in team/Phase-2 view showed **zero submissions** on 2026-08-25.
- The shipped width-1024 path is basic K2p1. Its expected score is
  `~3.93–3.96e-7`, around rank 33 of 44 on the 2026-08-25 board. It is not a
  materially improved Phase-2 system.
- The only measured Phase-2 method gain is K2p2 at about 1.052x for about 0.18%
  extra counted cost; it is not integrated into the submission.
- P1-PRE's terminal implementation genuinely closes the six named guarantees.
  The sixth independent review returned PROVISIONAL on one behavioural defect
  and three false/stale durable claims. Phase 1A has not begun.

Do not edit `docs/HUMAN_MODEL.md`, a frozen estimator archive, the seed ledger,
vault state or `estimator.py` in P1-PRE or Phase 1A.

## Slice 1 — P1-PRE corrections only

This is IMPORTANT instrument work. Claim no seed and measure no truth data. Do
not add a new assurance framework and do not pursue the review's five explicitly
non-blocking design observations.

Fix exactly four items:

1. **Make T6 true.** Default `local_sim/p1pre_fault_inject.py --out` to a path
   outside the repository. Remove the tracked-transcript carve-out and reject
   every output path resolving inside the repo. Include the old tracked
   `fault_injection.txt` transcript in the protected hash/state evidence. Take
   the untouched-repository measurement after the output has been written, not
   before. Correct the module docstring.
2. **Correct the record append-only.** Do not alter the preregistration or prior
   result text. Append an explicit correction to `experiments/LOG.md`: T7's
   substantive property held, but the actual arithmetic was green `42 -> 43`
   and declined `0 -> 1` for the committed artifact pair. Replace the claimed
   score in the correction with **nine as written, one corrected** and retract
   “Ten of ten as written. No correction is owed.”
3. **Restore injection J.** Its deleted-blind-spot mutation still fires at HEAD,
   as the reviewer independently demonstrated. Restore it, require its named
   refusal, and record truthfully that the set went from 11 to 14 by dropping J
   and adding four, then to 15 when J was restored.
4. **Reconcile durable docs.** Ensure `CURRENT_STATE.md`, `GAME_PLAN.md`,
   `GOTCHAS.md` and this handoff describe current behaviour and do not retain the
   old HEAD-0810537 destructive-cleanup warning as a present-tense instruction.

Verify the corrected tree with an external default-output run and an explicit
in-repository-output refusal; prove after-write status and hashes unchanged;
require 15/15 named mutations; re-run D1-D3 focused evidence; run full quiet-host
preflight and portability. Preserve raw transcripts and append the result and
knowledge delta to the LOG.

Then launch one fresh independent Skeptic that has not seen Builder reasoning.
Give it the four-item correction scope, exact diff, raw artifacts and
`docs/SKEPTIC.md`. Ask whether those four items are closed and the durable record
is truthful. The five disclosed meta-design observations are limitations, not
this gate. Record the literal verdict.

- If the verdict is ACCEPT, synthesize the docs and continue to both post-accept
  tracks below.
- If one of the same four items is still false, stop and report the exact evidence
  to Samuel. Do not begin another open-ended evaluator-hardening cycle.

## Post-accept track A — prepare one Phase-2 calibration submission

This is STRATEGIC because submission changes external state. It may run alongside
Phase 1A, but it must not delay Phase 1A.

Prepare, but do not submit autonomously:

1. freeze and record the current `estimator.py` hash and its K2p1 identity;
2. record the expected baseline result (`~3.93–3.96e-7`, roughly rank 33), making
   explicit that it is calibration and not a claimed improvement;
3. run `whest validate` and a clean official `v2-phase2` subprocess mini with
   zero failed MLPs;
4. verify official config, ship-file identity, clean worktree and current daily
   allowance;
5. present the readiness packet to Samuel and ask for explicit approval.

Only after explicit approval may you append the submission contract, submit once,
and record the official result. Fit no constant to the public result and do not
nominate this baseline as a final candidate merely because it exists.

## Post-accept track B — Phase 1A construction, no truth data

Phase 1A is IMPORTANT because it changes probe mathematics. Record its written
prediction/knowledge target before implementation. Claim no seed, use no truth
MSE and do not touch `estimator.py`.

Build the width-1024 Kerdock/MUB mathematical specification, direct generator,
independent verifier, exact-radial rule, dense propagation reference, and
counted-cost/memory model. The existing length-256 construction is GR(4,7).
Width 1024 requires the known GR(4,9)/real-MUB construction and a direct way to
emit 1/2/4/8/16 bases with antipodes without materialising the full 1,048,576-word
code. Use Calderbank, Cameron, Kantor and Seidel, “Z4-Kerdock Codes, Orthogonal
Spreads, and Extremal Euclidean Line-Sets” as the primary mathematical anchor.

Required evidence:

1. write the exact algebraic specification, phase/sign convention, basis indexing
   and deterministic ordering before coding;
2. implement the efficient generator and a verifier/reference using genuinely
   different logic, with reduced-dimension exhaustive cases;
3. prove norms, within-basis orthogonality, cross-basis coherence `1/32`,
   antipodal closure, uniqueness and determinism for 1/2/4/8/16 bases;
4. validate positive homogeneity and the Gaussian radial factor separately;
5. measure degree-4 angular error, including aligned/adversarial probes; do not
   call the 16-basis subset an exact degree-4 design without proof;
6. build the dense 32,768 x 1,024 float32 reference first, reproduce the leading
   32,768-forward FLOP fingerprint with explained overhead, then compare any
   chunked spelling against it;
7. account for counted FLOPs and point/weight/temporary memory honestly.

Exit only on zero algebraic/reference/determinism failures, explained row and
cost accounting, full preflight, portability, raw evidence, LOG knowledge delta,
and a fresh independent Skeptic verdict. No score or variance claim is permitted
in Phase 1A.

After Phase 1A ACCEPT, update `CURRENT_STATE.md` and `GAME_PLAN.md`, then stop for
Samuel's review of the Phase 1B STRATEGIC contract and his restated S2 prediction
before claiming a new P02 DESIGN block. Later branches remain exactly those in
`GAME_PLAN.md`: carrier DECIDE, faithful recurrent augmented K2, checkpoint
response closure, integration/robustness, public CONFIRM, authorized Phase-2
vault, and up to two frozen final nominees.

At every stop, report exact evidence, current SHA, decision needed, and safe work
that can continue without crossing the gate.
