# Claude Code handoff — close Phase 1A/Track 0S, then run a cost-aware Phase 1B

Copy the prompt below into a new Claude Code session started at the repository
root. It assumes no chat history. Repository evidence is authoritative.

---

## Prompt

You are the repo-local Builder for Samuel's ARC White-Box Estimation Challenge
2026 entry. Work from:

```text
C:\Users\samsu\Documents\arc-whest
```

Continue autonomously through the bounded work below. Stop at every Samuel-only
prediction, submission, vault, seed-claim, `estimator.py` edit, or material scope
expansion. Never submit or touch a vault without Samuel's explicit approval.

First read `AGENTS.md`, then `docs/CURRENT_STATE.md`, the bottom of
`experiments/LOG.md`, `docs/GAME_PLAN.md`, `docs/VALIDATION.md`,
`docs/GOTCHAS.md`, `docs/KERDOCK_MUB_SPEC.md`, and `docs/SKEPTIC.md`. Inspect HEAD
and the worktree before editing. The pre-review worker HEAD was `f8a2df5`; a later
documentation-only review commit may now be HEAD. `estimator.py`, the seed ledger
and the vault were untouched by the review.

## What is true now

- Phase 0 and P1-PRE are CLOSED on literal independent ACCEPTs. Do not reopen
  them.
- No Phase-2 submission has been made. The signed-in team/Phase-2 filter showed
  zero submissions on the 2026-08-25 Chicago / 2026-08-26 UTC recheck.
- The frozen K2p1 submission file is operationally safe but not improved:
  `4.050566369642183e-7` adjusted / `4.050566369642183e-6` raw on the official
  100-MLP subprocess mini, multiplier `0.1`, `137,600,326,656` FLOPs/MLP, zero
  failures. Its 4.288x residual headroom is **LOCAL and one-session only**.
- The live board recheck had 48 entrants. Puffi led at `9.50e-9` adjusted /
  `3.10e-8` raw. Our baseline projects to about rank 42 and is 42.6x worse in
  adjusted score. The live leader row implies multiplier about `0.306`.
- Phase 1A really does emit a valid 16-basis real-MUB carrier. Independent reruns
  reproduced verifier 34/34 and probe 16/16, exact `1/32` coherence, phase-degree
  distribution `{0: 1, 2: 511}`, all 130,816 differences rank 10, and the rejected
  mapping's failure. K8 is exact: the aligned fourth-moment ratio is
  `21.6881103515625`; 64 random probes merely missed that defect.
- Dense 16 costs exactly `2**40`, so `C/B=0.5` and its multiplier is `0.5` before
  analytic/control overhead. It is an experimental endpoint, not the presumed
  winning architecture. The current leader cost points instead toward roughly an
  8--10-basis carrier after overhead.
- Both terminal reviews are PROVISIONAL. Phase 1A lacks a grader-legal carrier
  construction path and truthful setup/memory clearance. Track 0S's full-preflight
  witness names `3adf81f`, before later `local_sim/` corrections, so its own gate
  fails at current HEAD.

Do not edit `docs/HUMAN_MODEL.md`, frozen estimator archives, the seed ledger,
vault state, or `estimator.py` in the bounded closure below.

## A. Close Phase 1A — IMPORTANT, no truth or seed

Do exactly this bounded repair; do not reopen the algebra or start another general
hardening cycle.

1. Reconcile every durable claim with the formal K6 withdrawal. The claim is only
   that the emitted object satisfies the defining real-MUB Gram properties; the
   property checker is not an independent construction. Remove stale 23/23,
   13/13, “all 512 degree 2,” independent-reference, `104`, global-extremality,
   and explained-leader-overhead wording. Preserve history by appending a LOG
   correction rather than rewriting prior LOG entries.
2. Add a genuinely grader-legal carrier construction/materialisation path using
   only Python stdlib and `flopscope.numpy as fnp`. The current probe builds with
   NumPy and wraps the finished array, which does not qualify. Prove the legal
   path byte-identical to the NumPy reference for `k=1/2/4/8/16`, arbitrary
   supported basis indices, and deterministic fresh processes.
3. Replace the memory false-green. Observed Windows peak commit around 1.147 GiB
   gives at most 7.14x headroom; it cannot prove an 8-GiB `RLIMIT_AS` pass. A check
   must not pass merely because commit is below 8 GiB. Record largest arrays,
   commit/RSS/virtual-space limitations, and leave enforcement unproved unless a
   Linux subprocess actually enforces it.
4. Do not call an after-import construction timer the official setup boundary.
   Measure repeated fresh-worker spawn + import + setup if the evaluator exposes
   that boundary; otherwise explicitly defer setup clearance. Preserve the dense
   ledger `2**40`, `C/B=0.5`, and say the private leader's remainder is unknown.
5. Rerun the 34/34 verifier, 16/16 probe, portability, and full preflight. Capture
   full preflight first to a system-temp file so the transcript header witnesses a
   clean tree. Commit the transcript/regenerated evidence as the single evidence
   commit permitted by the gate; prove no `local_sim/`, `scripts/`, or
   `estimator.py` difference from the witnessed HEAD.
6. Append the knowledge delta and give the contract, diff, raw artifacts, and
   current-HEAD transcript to a fresh independent Skeptic. Phase 1A closes only on
   literal ACCEPT. If the reviewer finds another blocking false durable claim,
   stop and return it; do not start Phase 1B.

## B. Close Track 0S — still no submission without Samuel

This is a calibration baseline, not algorithmic progress.

1. Correct the GT-floor artifact label to `mean(avg_variance) / 1e9`; qualify the
   4.288x residual number as LOCAL, one-session; anchor the duplicate-transcript
   disclosure to historical commit `6d3423c` rather than claiming the current
   files are identical.
2. Reuse the clean final-HEAD preflight witness only if the Track-0S contract's
   evidence-only-commit rule is literally met. Otherwise run the gate again.
3. Live-recheck the signed-in Phase-2 submission count and remaining daily
   allowance. Preserve capture where possible and label transcription as such.
4. Obtain a focused independent ACCEPT on the exact candidate SHA and packet.
5. Stop and ask Samuel to authorize one calibration submission of that exact SHA.
   Never infer approval. If approved later, submit once and preserve the returned
   ID/report verbatim; fit nothing to the score and do not call it a nominee.

Track 0S does not block Phase 1B after Phase 1A is accepted.

## C. Prepare exact fast paths before spending Phase 1B truth

After Phase 1A ACCEPT, but before claiming a DESIGN block, implement three
algebraically equivalent local paths:

1. dense carrier reference;
2. FWHT evaluation of layer 0;
3. FWHT layer 0 plus exact layer-1 antipodal reuse:

```text
[z]+ W1 + [-z]+ W1 = |z| W1
[z]+ W1 - [-z]+ W1 = z W1 = x (W0 W1)
```

Verify small and live shapes against dense, distinguish exact algebra from
float32 rounding, and record official flopscope 0.12 operation ledgers. The
leading fast/dense ratio is approximately `29/32 + 1/(32k)` before lower-order
transforms and assembly. Dense remains the numerical oracle, not the production
candidate. Do not use truth MSE to choose the spelling.

## D. Phase 1B — STRATEGIC adjusted-score DESIGN frontier

Before any truth exists:

1. Append the full Rule-1 contract to `experiments/LOG.md` and commit it.
2. Restate Samuel's S2 prediction and take his predictions for the new primary
   questions: winning basis-count region, raw-MSE gain versus K2p1, and residual
   safety. Stop for his answers before claiming the new DESIGN block.
3. Claim a fresh DESIGN block through the ledger. Do not reuse P01 as decision
   evidence and do not touch a SINGLE_USE DECIDE block.

The primary estimand is paired arithmetic **adjusted score**, using common random
numbers and `competition_score.decide()` with a preregistered gate. Report every
MSE with the GT-noise floor. Use studentised-t or basic bootstrap, never percentile
or BCa. Total counted FLOPs include carrier propagation, casts, means, output
assembly, and every analytic/control branch.

Dense carrier-only break-even before overhead is:

| bases `k` | `F/B` | multiplier | raw gain versus K2p1 needed to beat K2p1 |
|---:|---:|---:|---:|
| 1 | 0.03125 | 0.100 | any strict gain |
| 2 | 0.06250 | 0.100 | any strict gain |
| 4 | 0.12500 | 0.125 | >1.25x |
| 8 | 0.25000 | 0.250 | >2.50x |
| 16 | 0.50000 | 0.500 | >5.00x |

Evaluate cumulative `k=1..16` prefixes from one full trajectory. Predeclare only
these mechanism-based nested policies: validated algebraic prefix; MLP-seeded
uniform nested permutation; identity basis replacing one Kerdock basis. A seeded
global signed-Walsh input rotation may be one bounded ablation. Do not tune literal
basis indices. Include matched-row IID and ordinary-orthogonal controls as
diagnostics, but do not gate on the old 2.7x-versus-IID prediction.

Use two independent truth replicas and the cross-replica/noise-corrected identity.
If the selected arm's truth-noise floor exceeds 10% of its raw MSE, increase truth
replicas uniformly before selection. Record layerwise errors, local `k -> 2k` MSE
elasticity, carrier-versus-K2 checkpoint discrepancies, exact FLOPs, residual,
memory, setup and failures.

Freeze basis policy, `k`, exact execution spelling and chunking from DESIGN-A;
replicate them unchanged on locked DESIGN-B. Advance to a later DECIDE block only
if the replication has at least 1.20x paired adjusted-score gain over K2p1, the
allowed interval excludes zero, zero failures, and all operational falsifiers
pass. This is a DESIGN screen, not promotion.

Only after freezing the carrier, test shallow analytic checkpoints (for example
layers 4/8/12) as a separate marginal estimand against that carrier. Measure actual
prefix overhead. Prefer a shallow prefix; a full K2p2 branch adds about `1.378e11`
FLOPs and raises the approximate multipliers at `k=2/4/8/16` to
`0.1252/0.1877/0.3127/0.5627`. Gate “hybrid minus frozen carrier,” never hybrid
versus K2 again.

## Binding operational falsifiers

- Three independent 100-MLP official-subprocess soaks. Report first-MLP warm-up
  separately but never discard it. Zero rows may reach 0.4 seconds, and require
  `2.35 * quiet-session p99 < 0.4 s` to cover the measured session spread.
- No setup timeout across repeated fresh workers. The whole subprocess boundary is
  authoritative, not an after-import timer.
- Linux subprocess under an actual 6-GiB sacrificial `RLIMIT_AS`: zero
  `MemoryError`, worker death or restart; record `VmPeak`, RSS, largest array and
  workspace accounting. Without Linux enforcement, memory remains unproved and
  the candidate is not deployable.
- Total integrated counted FLOPs below `2**41`; no failed MLPs; no grader-policy
  violation.

At each gate, report exact evidence, current SHA, the decision needed, and safe work
that can continue. Do not edit `estimator.py` or claim a DECIDE seed without a new
STRATEGIC contract and Samuel's required prediction/approval.
