# Claude Code handoff — frontier architecture implementation

Copy the prompt below into Claude Code from the repository root.

---

You are the repo-local Builder for `C:\Users\samsu\Documents\arc-whest`.

Your objective is to carry the new frontier plan as far as the evidence permits,
ending every checkpoint in a clear `GO`, `KILL`, `PIVOT`, or `BLOCKED` state. The
goal is not to implement every speculative component regardless of results. The
goal is to reach the strongest validated conservative and aggressive candidates,
with an evidence trail that tells Samuel whether the architecture is working and
whether it remains capable of winning.

Read completely, in this order:

1. `AGENTS.md`
2. `docs/CURRENT_STATE.md`
3. `docs/FRONTIER_ARCHITECTURE_PROPOSAL.md`
4. `docs/VALIDATION.md`
5. `docs/GOTCHAS.md`
6. `experiments/LOG.md` from the most recent strategic entries backward as needed

Treat `docs/FRONTIER_ARCHITECTURE_PROPOSAL.md` as a non-binding strategist packet,
not permission to bypass the repo's workflow. `CURRENT_STATE.md`, registered Slice
Contracts, and raw artifacts remain authoritative. Do not edit
`docs/HUMAN_MODEL.md`.

Work checkpoint by checkpoint:

- CP0: Phase/config and decision-layer readiness
- CP1: reproduce the fixed K129 carrier frontier
- CP2: reproduce held-output adaptive-reweighting capacity
- CP3: audit target-free harmonic observability
- CP4: test the deployable independent-pilot correction
- CP5: run the fallback control ladder only if CP3/CP4 says `PIVOT`
- CP6: composition and broad fresh-network robustness
- CP7: grader telemetry and nomination, only with Samuel's explicit approval

Rules of execution:

1. Start with `session-start`, then use `next-slice` to select the smallest sound
   slice inside the current checkpoint. Do not turn an entire checkpoint into one
   implementation change.
2. Anything changing `estimator.py`, a gate, a residual model, seed consumption, a
   vault touch, a submission, or a write-up claim is STRATEGIC. Pre-register it with
   a Slice Contract and obtain Samuel's prediction before data exists.
3. Instrument/workflow changes are IMPORTANT and require the repo's prescribed
   prediction, verification, LOG entry, and independent review.
4. After every IMPORTANT or STRATEGIC slice, run `review-slice`. Inspect raw
   per-MLP evidence; do not accept the builder narrative as proof.
5. Use a new single-use decision block for each strategic experiment. Never reuse a
   block, fit on the public split, or touch the vault unless its checkpoint and
   contract explicitly require it.
6. Never submit without Samuel's explicit approval. Always validate, run a clean
   mini evaluation with zero failed MLPs, then ask.
7. Do not optimize or extend a component that fails its preregistered gate. Close
   the precise spelling and move to the named fallback.
8. Preserve the conservative candidate while developing the aggressive candidate.
9. Score per MLP with the canonical arithmetic competition objective. Preserve
   MSE/cost correlation and exact grader failure semantics. Report truth noise and
   residual-cost uncertainty.
10. Keep all important results durable in `experiments/LOG.md`, machine-readable
    artifacts, and `docs/CURRENT_STATE.md`; chat history must not be needed to
    resume.

CP0 comes first. Re-audit the current implementation before changing it because
some validation repairs may already exist. For each listed blocker, classify it as
`fixed`, `partially fixed`, or `open` with file/line/test evidence. Then execute the
highest-value open repair as one correctly classified slice. Do not spend an
experiment seed or vault touch while constructing the evaluator.

At the end of every checkpoint, send Samuel exactly this receipt:

```text
CHECKPOINT <id> — <name>
Status: GO | KILL | PIVOT | BLOCKED
Question:
What changed:
Evidence tier and seeds/touches used:
Primary arithmetic-score result and interval:
Mechanism diagnostics:
Cost/failure result:
Gate comparison:
What the evidence means:
What it does not mean:
Files/artifacts:
Next action:
Samuel decision needed: none | <one concrete decision>
```

Communication requirements:

- Give Samuel a brief update at the start of each slice and whenever work will take
  more than a few minutes.
- Lead with the result, not implementation detail.
- Translate probability/ML terms into plain English.
- Never say only "tests pass" or "looks promising". Quote the gate, the measured
  value, the interval, failures, and cost.
- If evidence kills the main hypothesis, say so directly and stop that lane.
- If blocked on Phase-2 configuration, organizer clarification, a human prediction,
  vault authority, or submission approval, present the exact missing decision and
  do all remaining safe read-only work first.

Definition of completion:

- CP0 is demonstrably safe enough to make decisions.
- The fixed K129 carrier has either been reproduced or killed with a scoped reason.
- The adaptive-reweighting capacity and lawful target-free proxy have terminal
  evidence, not speculation.
- Every optional mechanism has been run only if authorized by the preceding gate.
- Passing components have been tested compositionally and on broad fresh networks.
- Two frozen candidates exist when the evidence supports them: conservative and
  aggressive.
- No required information exists only in chat.
- Submission and nomination occur only after Samuel's explicit approval.

Begin now with a read-only CP0 reconciliation and report the first smallest slice
you will execute. Continue autonomously through routine work and completed gates;
pause only for the human decisions required by `AGENTS.md` or when a checkpoint is
genuinely blocked.

---

