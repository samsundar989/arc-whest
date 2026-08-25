# Claude Code handoff — close Track 0S and Phase 1A, then stop

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

Continue autonomously through the bounded safe work below. Stop at every
Samuel-only prediction, vault, submission, seed-claim, estimator edit, or material
scope-expansion gate. Never submit or touch a vault without Samuel's explicit
approval.

First read `AGENTS.md`, then `docs/CURRENT_STATE.md`, the bottom of
`experiments/LOG.md`, `docs/GAME_PLAN.md`, `docs/VALIDATION.md`,
`docs/GOTCHAS.md`, and `docs/SKEPTIC.md`. Inspect the worktree and current HEAD
before editing. At this handoff, the reviewed worker HEAD was `7e06cd3`; the tree
was clean before the plan-review documentation update. `estimator.py`, the seed
ledger and the vault were untouched.

## What is true now

- Phase 0 and P1-PRE are CLOSED on literal independent ACCEPTs. Do not reopen
  them.
- Track 0S's frozen K2p1 candidate is technically submission-safe: SHA-256
  `00240eb1e60810f5e9a6f743b8ad89a1533cfd8671c08e07525f1ed1de54d9ec`,
  byte-identical to `estimator_v2_6_1.py`; official 100-MLP subprocess mini
  `4.050566369642183e-7` adjusted / `4.050566369642183e-6` raw, multiplier
  0.1, fixed `137,600,326,656` FLOPs/MLP, zero failures, worst residual
  `0.0932798 s` = 4.288x headroom. This is calibration, not improvement.
- The mini's required mean ground-truth noise floor is
  `7.47823932957716e-11` (0.001846% of raw MSE); the packet omitted it.
- The signed-in Phase-2 team view was rechecked on 2026-08-25 and showed zero
  submissions. The official overview says 10 submissions/team/UTC day. Preserve
  current raw evidence rather than relying only on prose.
- The live board recheck had 46 entrants. Puffi led at `1.09e-8` adjusted /
  `3.55e-8` raw; the `3.93e-7` covariance-propagation tie occupied about ranks
  35--39. Our measured baseline projects to about rank 40, 37.2x behind adjusted.
- Phase 1A's corrected generator passes 23/23 and genuinely emits a 16-basis
  real-MUB subset: exact 1/32 coherence, degree-2 phase functions and rank-10
  differences for all 120 basis pairs. The earlier coordinate mapping genuinely
  failed. But Phase 1A is PROVISIONAL: K6 overclaims an independent reference,
  and K7--K10 remain incomplete.

Do not edit `docs/HUMAN_MODEL.md`, a frozen estimator archive, the seed ledger,
vault state, or `estimator.py` in either track.

## Track 0S — cheap procedural closure, then Samuel's gate

This remains STRATEGIC because submission changes external state. Do not rerun the
100-MLP mini unless the candidate hash, dataset revision, or official config has
changed.

1. Append (never rewrite) a correction to the readiness packet recording the
   exact GT-noise floor above and correcting the stale rank/board language.
2. Preserve a current signed-in artifact showing this team has zero Phase-2
   submissions and a current official-rule artifact showing the allowance. Record
   that the visible live score is the 50-network feedback half; the other 50 are
   withheld for the final result. Do not describe a live score as public+private.
3. Append and commit the complete submission contract before submission. Include
   candidate SHA, expected official score band, zero-failure prediction, fixed
   FLOPs/floor multiplier, operational gate and falsifier, and a promise to fit
   nothing to the result. Take and record Samuel's prediction before submission.
4. Give the corrected packet and raw artifacts to a fresh independent Skeptic.
   Require literal ACCEPT on this bounded correction.
5. Stop and ask Samuel to authorize exactly one Phase-2 calibration submission of
   the exact SHA. Only after his explicit approval may you submit once. Preserve
   the returned ID/report verbatim. It is not a final nominee merely because it
   exists.

## Phase 1A — bounded mathematical/instrument closure, no truth data

Continue Phase 1A; do not restart the valid generator and do not claim a seed.
Before Phase 1B, close every item below:

1. Correct the stale `sign_vectors` docstring that still describes the rejected
   multiplicative-rank coordinate mapping. Preserve that bad mapping as an
   executable negative control with a raw artifact; correct the historical
   spectrum wording to what the executable test actually reports.
2. Fulfil K6 with an independently generated finite-field Kerdock-matrix
   reference following CCKS Example 9.2, rather than a property checker over
   generator output. Use the finite-field construction on
   `V=F_(2^m) x F2` and compare reduced dimensions exhaustively and the live
   chosen phases up to a declared fixed linear/signed equivalence. If that exact
   reference cannot be completed, append a formal K6 correction and narrow the
   gate; never retain the false independent-reference claim.
3. Test nested live `k=1/2/4/8/16` outputs, configurable basis indices,
   deterministic ordering, float32 dtype and unit norms. Either include the
   standard basis deliberately or explicitly defer basis-set choice to Phase 1B.
4. Validate the chi-radial identity separately: stable exact constant, layer-1
   closed form, every network prefix, positive homogeneity, and a guard proving
   the factor is applied exactly once.
5. Prove the exact second moment and report an exact fourth-moment tensor or frame
   diagnostic. The `N=1024,k=16` carrier-aligned fourth moment is
   `21.6881103515625x` the spherical target; random probes alone cannot validate
   K8. This is quadrature characterization, not a performance claim.
6. Build the dense 32,768 x 1,024 float32 oracle, then implement a submission-legal
   `fnp` construction/materialisation path. Measure the actual official-flopscope
   operation ledger and explain this implementation's overhead. Do not force a
   match to a leader's unknown implementation merely because its total was
   published.
7. Measure fresh-subprocess setup time and peak/transient memory including
   construction, simultaneous point buffers, weights, temporaries and operation
   history. Windows memory is an accounting upper bound, not proof that the 8-GiB
   grader limit was enforced.
8. Run full preflight and portability, preserve raw evidence, append the knowledge
   delta, and launch a fresh independent Skeptic. Phase 1A closes only on literal
   ACCEPT.

Stop after Phase 1A ACCEPT. Phase 1B is STRATEGIC: Samuel's restated S2 prediction,
a reviewed contract and a fresh DESIGN block must all precede any truth MSE,
`estimator.py` edit, or performance claim. At each stop report exact evidence,
current SHA, the decision needed, and safe work that can continue without crossing
the gate.
