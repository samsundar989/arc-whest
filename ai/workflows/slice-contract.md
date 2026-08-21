# slice-contract — arc-whest

*Canonical workflow. Agent-independent. Adapters: `.claude/skills/slice-contract/`,
`.agents/skills/slice-contract/`.*

> **Position in the workflow.** This is an internal primitive beneath the standard
> control surface. `next-slice` decides *that* a strategic slice is the right move
> and enters this; this file governs *how* the slice runs and ends at the Evidence
> Packet; `review-slice` then closes the epistemic loop. Entering it directly is
> fine when resuming a slice that is already under contract.

This project's decisive failure mode is not bad code. It is **a number that was
chosen after the answer was visible.** Sprint 4 produced a 1.41× headline that
became 1.20× under a correctly paired statistic and then died at 1.1216×. Two
separate "findings" turned out to be noise with a tidy causal story attached on
demand. Both were caught, but only by accident and by an independent audit.

This workflow makes the ordering enforceable: **the contract exists in git before the
data does.**

## Step 0 — is this actually strategic?

Skip this for ROUTINE work (docs, figures, board snapshots, lab scripts that
touch no seed block and no shipped file). For IMPORTANT work (changing the
*instrument* — `harness.py`, `mint_provenance.py`, `edge_cases.py`, `seed_blocks.py`,
`vault.py`, a probe's math, or the workflow layer under `ai/workflows/`) use a
shortened form: prediction + preflight + LOG entry with a knowledge delta, no decision
block, no Skeptic.

Use the full form when the work touches `estimator.py`, spends a decision block,
touches the vault, changes a gate or a residual model or the protocol, makes a
write-up claim, or reopens a lane listed as CLOSED in `docs/CURRENT_STATE.md`.

## Step 1 — orient, and read before writing

Read, in this order: `docs/CURRENT_STATE.md` (is this lane already closed, and on
what evidence?), `docs/VALIDATION.md` (the protocol is binding), `docs/GOTCHAS.md`
(has this exact trap been recorded?). Then find the closest prior LOG entry and read
what it concluded.

If the lane is CLOSED, stop and say so. Reopening requires **new external evidence**
— a competitor's published telemetry, an organizer ruling, a paper — not a new idea
about the same evidence. State the reopening condition explicitly in the contract.

## Step 2 — allocate the seed block BEFORE measuring

```powershell
uv run python local_sim/seed_blocks.py
```

- A **design** block permits tuning. A **decision** block is single-use and is
  consumed by exactly one recorded experiment. `seed_blocks.SINGLE_USE_BLOCKS` makes
  this machine-readable; `check_disjoint()` is asserted by the invariant suite.
- A block that appears in two LOG entries means the second one is not evidence.
- Legacy blocks L1–L5 predate the mint fix and are comparable to nothing after it.
- Never fit anything on the public split. It is CONFIRM tier: read-only, forever.

## Step 3 — write the contract into `experiments/LOG.md`, then stop

Append this, with a heading dated and naming the block, **before running anything
that touches the block.** Keep it short enough that it actually gets used.

```markdown
## <date> — SLICE CONTRACT: <lane> on block <Dxx>  [PRE-REGISTERED, no data yet]

**Strategic question**  What are we trying to learn?
**Current bottleneck**  What limits us, and per which measurement?
**Evidence**            What we already know. Mark CONFIRMED / INFERRED / SPECULATIVE.
**Hypothesis**          What we believe is true.
**Mechanism**           Why this change should produce that result. If there is no
                        mechanism, there is no experiment — only a search.
**Intervention**        The smallest change that discriminates. One hypothesis only.
**Prediction**          A number with a direction and a rough magnitude.
**Gate**                The threshold, chosen NOW, and passed to `decide(gate=...)`
                        which has no default: e.g. paired arithmetic V·C >= 1.15x
                        with the 95% t interval excluding zero.
**Falsifier**           What result rejects the hypothesis.
**Baseline**            The exact control, and the tier it was measured on.
**Residual model**      REQUIRED for any cost claim. Report three (VALIDATION 2b);
                        FREEZE one here by name. The deciding rule is
                        max(local-ratio-on-graded-anchor, competitor-anchored).
**Statistic**           Paired, common random numbers, **arithmetic** score
                        difference with the studentised-t or basic-bootstrap
                        interval, through `competition_score.decide()`. Geometric
                        mean reported as a diagnostic. Say how many shifts. Never
                        a bare ratio of means, never percentile or BCa.
**Confounders**         What else could produce a positive result?
**Scope**               Files expected to change.
**Do not change**       Variables that must stay fixed.
**Evidence required**   Commands, artifacts, raw JSON paths, seeds, runtime.
**Driver**              Which agent is building this slice. Recorded so a later
                        reader knows what was in play — never so the entry depends
                        on that agent's chat history to be understood.
```

## Step 4 — Samuel's prediction (do not skip, do not lecture)

Ask at most three short questions and wait:

1. What part of the system do you think this changes?
2. What do you expect to happen — and roughly how much?
3. What result would convince you the hypothesis is wrong?

"I don't know" is a valid answer and means: explain **only** the missing concept,
using this repository's real files and one actual trace, in about three minutes,
then continue. Target total human time ~2–5 minutes. His prediction goes in the LOG
entry *before* the result — it is the only record of whether the project's model of
itself is any good.

## Step 5 — build, minimally

Change the fewest variables that answer the question. Verify assumptions against the
code rather than against the docs. If implementation reveals the premise is wrong,
**return that discovery** — do not force the original plan through. A failed
hypothesis with clean evidence is a result; this project's best material is negative.

## Step 6 — measure once

```powershell
.\scripts\preflight.ps1     # must be GREEN before a block is spent
```

Then spend the block, once, exactly as pre-registered. Do not adjust the gate. Do
not re-roll shifts. Do not add MLPs because the interval was wide.

## Step 7 — Evidence Packet, into the LOG

Prose is not evidence. Return:

```markdown
### RESULT — <lane>, block <Dxx>  [contract above, unmodified]

commit / diff       <sha> (commit BEFORE reporting, so the entry can cite it)
commands            verbatim, copy-pasteable
config / seeds      block, n_mlps, n_shifts, N, truth samples
baseline            value + tier
result              value + 95% CI
delta vs gate       PASS / FAIL, stated as such
GT-noise floor      and the result as a multiple of it
regressions         preflight 11/11 with -Full? invariants 30/30? anything that moved?
runtime / cost      wall time, vault touches spent (should usually be 0)
raw artifacts       paths under experiments/
deviations          anything that differed from the contract, and why
unexpected          what surprised us
remaining doubt     what this does NOT settle
```

Then a **Knowledge Delta**: what was learned · which prior belief changed · any new
invariant · any new failure signature · remaining uncertainty · which durable doc
must change · what this implies for the next experiment.

Update `docs/CURRENT_STATE.md` if a lane opened or closed. Add to `docs/GOTCHAS.md`
if a trap was found. Both, if both.

## Step 8 — hand off to `review-slice`, then stop

The slice ends at the Evidence Packet. **Do not review your own work here, and do not
open the next slice from inside this one.**

`review-slice` owns what comes next: independent falsification by the Skeptic role in
a separate agent context, the Knowledge Delta into durable docs, Samuel's teach-back,
and the single terminal state (PROMOTE_AND_CONTINUE / FOLLOW_UP_EXPERIMENT / REVERT /
STRATEGIC_RESEARCH / STRATEGIC_DECISION / FIX_EVALUATION).

Tell Samuel the packet is ready and that `review-slice` is the next command.
Synthesis before the next strategic change is the rule that keeps this project's
beliefs from drifting ahead of its evidence.

## Standing prohibitions

- Never submit without Samuel's explicit approval.
- Never touch the vault to browse. Three permitted touches, ever, all logged.
- Never quote a local absolute residual; transfer ratios only (~7× off).
- Never compare a counted FLOP figure to an effective one.
- Never trust `local_sim/eval_estimator.py`'s absolute level for a rotation-using
  estimator (~18% low). Paired comparisons within one path are fine.
- Never decide a <15% question on a single 100-MLP run. Two realisations of the same
  estimator differ ~18% in the mean; use multi-shift paired averaging.
