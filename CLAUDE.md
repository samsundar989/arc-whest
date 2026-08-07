# CLAUDE.md — arc-whest

*A map and a contract, not an encyclopedia. Deeper knowledge lives in `docs/`.
Harness audit: 2026-08-07. Re-audit triggers at the bottom.*

## Objective

Samuel's entry for the **ARC White-Box Estimation Challenge 2026** (AIcrowd).
Given a random He-init ReLU MLP (Phase 1: 256 wide × 32 deep, no biases), predict
every neuron's expected post-ReLU activation under x ~ N(0, I) as a `(depth, width)`
array. Score = mean over MLPs of `final_layer_MSE × max(0.1, C/B)`, lower is better;
`C` = flopscope-counted FLOPs + 1e11 × uninstrumented wall-seconds, `B` = 2.72e11.

**The prize is decided by a private re-run of ONE designated submission on fresh
MLPs.** Public board rank is not the target. There is a second, now co-equal
objective: the **$20k contribution prize**, whose deliverable is the write-up
(`docs/writeup-draft.md`).

Where we stand, what is open, and what to do next: **`docs/CURRENT_STATE.md`**.

## Layout

| path | what |
|---|---|
| `estimator.py` | **THE submission.** Self-contained; `whest submit` ships only this file. Currently v2.4.1, byte-identical to `estimator_v2_4_1.py` modulo line endings. |
| `estimator_v*.py` | frozen version archive — never edited, only read |
| `local_sim/` | the offline lab: numpy shims for `flopscope`/`whestbench`, grader-identical mint (`mint_provenance.py`), invariants (`edge_cases.py`), seed registry (`seed_blocks.py`), sealed holdout (`vault.py`), probes |
| `experiments/` | `LOG.md` (append every experiment), run reports, vault seal, JSON artifacts |
| `docs/` | see pointers below |
| `scripts/` | `preflight.ps1` — the gate that runs on every estimator change |

## Hard rules — grader contract and research integrity

1. `estimator.py` may import ONLY `flopscope`, `flopscope.numpy as fnp`, `whestbench`,
   and Python stdlib. No numpy/scipy/torch — the grader sandbox lacks them. Grader
   arrays are **immutable** (no `a[i]=`, no `+=` on arrays); pass `str` not `Path`;
   single array ≤ 4 GiB.
2. Determinism: randomness derives from `mlp.seed` (predict) / `context.seed` (setup)
   via `fnp.random.default_rng(...)`. Never `fnp.random.seed(...)`, never time-based.
3. **Never exploit the flopscope wall-time loophole** (routing math through raw
   uninstrumented arrays). Organizers call it out; we compete clean.
4. Never `flops.budget_summary_dict()` / `budget_summary()` inside `predict` — O(op
   history), 157× residual blowup.
5. Secrets: the AIcrowd key lives only in `whest login`'s credential store. Never in
   tracked files, logs, or LOG.md.
6. **Never submit without Samuel's explicit approval.** A failed MLP scores as zeros.
   Order is always: `whest validate` → clean mini run (`n_failed_mlps: 0`) → ask.
7. Keep `estimator.py` shape-generic (`mlp.width`, `len(mlp.weights)`).
   `ARC_WHEST_DISABLE_CAL=1` disables calibration (used by fitting).

## Measurement law — `docs/VALIDATION.md` is binding

- **Three tiers.** DECIDE on freshly minted networks with a **new seed block per
  experiment, never reused** (`local_sim/seed_blocks.py`) → CONFIRM on the public
  split (**read-only; no constant may ever be fitted there**) → VAULT for milestones
  (touch-counted; three permitted touches, ever).
- **Pre-register before you measure.** A decision block is single-use. Its design,
  gate threshold, and residual model go into `experiments/LOG.md` *before* the block
  is touched. A model chosen after seeing the answer is not evidence.
- **Counted compares to counted; effective compares to effective.** The paired
  statistic is the geometric mean of per-MLP ratios, not the ratio of means.
- **Every MSE is quoted with its ground-truth noise floor.**
- **Local residual wall time is ~7× the grader's** and the shim path reads ~18% low
  for rotation-using estimators. Never quote a local absolute; transfer ratios only.

## Commands

```powershell
uv sync                                                   # environment
.\scripts\preflight.ps1                                   # THE gate: provenance + 30 invariants + contract + ship-file identity
uv run whest validate --estimator estimator.py            # contract only (~46 ms)
uv run whest run --estimator estimator.py --dataset hf://aicrowd/arc-whestbench-public-2026@v1-phase1 --split mini --runner subprocess
uv run python local_sim/phase2_checklist.py               # rounds / dataset tags / boards
uv run python local_sim/seed_blocks.py                    # allocate a block BEFORE any experiment
.venv-gpu\Scripts\python.exe local_sim/vault.py verify    # seal integrity + touch log
```

Full reference (GPU venv, probes, dead-lane commands): `docs/SETUP_WINDOWS.md` §7.
**Run `whest` from PowerShell**, or set `PYTHONIOENCODING=utf-8` in Bash — the CLI's
box-drawing output kills Git Bash's cp1252 console inside its own error printer.

## Human workflow

```text
Start / re-enter a session      /session-start     (blank HUMAN_MODEL.md → first-orientation)
Choose the next high-value move /next-slice        (may be a diagnostic, not code)
After an IMPORTANT/STRATEGIC slice  /review-slice
Periodically check the mental model /cold-start
Mechanical work                 /routine <task>
End the work session            /session-close
```

Daily loop: `/session-start` → `/next-slice` → work → `/review-slice` → repeat →
`/session-close`. Run `/cold-start` every 3–5 strategic slices.

Project-specific machinery runs *beneath* these: the `slice-contract` skill
(pre-registration), the `skeptic` agent (independent falsification),
`scripts/preflight.ps1`, `local_sim/seed_blocks.py`, `local_sim/vault.py`, and the
probes. Do not invoke them as a substitute for the workflow above.

When `/next-slice` or `/review-slice` emits **STRATEGIC_RESEARCH** or
**STRATEGIC_DECISION**, it produces a Strategist Packet — send that to the strongest
reasoning model and bring the returned Slice Contract back through `/next-slice`.

**`docs/HUMAN_MODEL.md` is human-owned and must not be silently edited by agents**
(enforced by a hook). Agents read it, quiz against it, and propose corrections.

## Work classification

| class | what | required |
|---|---|---|
| **ROUTINE** | doc edits, figures, board snapshots, report fetching, new tests, lab scripts that touch no seed block and no shipped file | just run it; verification only |
| **IMPORTANT** | anything changing *the instrument* — `harness.py`, `mint_provenance.py`, `edge_cases.py`, `seed_blocks.py`, `vault.py`, a probe's math, grader-report parsing | written prediction first, `scripts/preflight.ps1` green, LOG entry with a knowledge delta |
| **STRATEGIC** | `estimator.py`; spending a decision block; a vault touch; a submission; changing a gate, a residual model, or the three-tier protocol; a write-up claim; reopening a closed lane | **Slice Contract** pre-registered in LOG before data exists → Samuel's prediction → build → evidence packet → independent Skeptic → teach-back. Use the `slice-contract` skill. |

**Never let one strategic result flow into the next implementation without
synthesis.** Closed lanes stay closed unless new *external* evidence reopens them —
the reopening condition goes in the contract.

## Tools — enabled and deliberately not

**Enabled**: the project's own scripts (they *are* the evaluator), `git`/`gh`,
`scripts/preflight.ps1`, skill `slice-contract`, agent `skeptic`.

**Deliberately disabled** — do not add without a named bottleneck: Ralph/autonomous
optimization loops (**the cost half of the evaluator is E1 — forbidden**), MLflow /
Phoenix / Langfuse, Context7, Serena, Repomix, Promptfoo, Playwright / Chrome
DevTools, Exa / Firecrawl / last30days, GitHub MCP, Symphony / Agent Deck, workflow
orchestration. Rationale: `docs/CURRENT_STATE.md` § *Harness*.

## Pointers

| doc | when to read it |
|---|---|
| `docs/CURRENT_STATE.md` | **first, always.** Baseline, open questions, closed lanes, next action |
| `docs/VALIDATION.md` | before measuring anything. The protocol and its proofs |
| `docs/GOTCHAS.md` | before quoting a number, comparing a cost, or trusting an offline path |
| `docs/GAME_PLAN.md` | strategy. **Rev 12 is authoritative**; earlier revs are history |
| `docs/BENCHMARK.md` | our numbers vs the field. **PRIVATE — never the forum** |
| `docs/COMPETITION_BRIEF.md` | plain-English explainer of the task and the math |
| `docs/RESEARCH_NOTES.md` | rules, forum intel, pricing table, links. §2g is **PRIVATE** |
| `docs/MOONSHOT.md` | the cumulant-flow lane (Phase D, the contribution prize) |
| `docs/HUMAN_MODEL.md` | **Samuel's file.** Agents read and critique it; agents never write it |
| `experiments/LOG.md` | the full narrative record, newest at the bottom |

## Working style

Samuel has a CS degree but is new to ML/probability — explain what you're doing in
plain English in the final summary, not in jargon. Concentrate his understanding on
the *decision-critical* path (the score identity, the data protocol, the residual
model); do not make him learn commodity code.

Claude (cloud session) drives strategy and heavy experiments; Claude Code handles
local pipeline, submissions, and grader feedback. Handoffs between them go through
**Slice Contracts and Evidence Packets in `experiments/LOG.md`**, never chat history.

## Re-audit this harness when

The competition phase changes · the evaluator's cost half becomes trustworthy · a
lane reopens · the write-up ships · agent/tool overhead becomes noticeable.
