# AGENTS.md — arc-whest

**The cross-agent repository operating contract.** Every coding agent that drives this
repository obeys this file, whatever product it is. Agent-specific files
(`CLAUDE.md`, `.claude/`, `.agents/`, `.codex/`) are thin adapters around it and must
never carry project behaviour of their own.

*A map and a contract, not an encyclopedia. Deeper knowledge lives in `docs/`.
Harness audit: 2026-08-07. Portability audit: 2026-08-07. Re-audit triggers at the bottom.*

## Objective

Samuel's entry for the **ARC White-Box Estimation Challenge 2026** (AIcrowd).
Given a random He-init ReLU MLP (Phase 1: 256 wide × 32 deep, no biases), predict
every neuron's expected post-ReLU activation under x ~ N(0, I) as a `(depth, width)`
array. Score = mean over MLPs of `final_layer_MSE × max(0.1, C/B)`, lower is better;
`C` = flopscope-counted FLOPs + 1e11 × uninstrumented wall-seconds, `B` = 2.72e11.

**The prize is decided by a private re-run of UP TO TWO designated submissions per
team on fresh MLPs** (organiser, forum 18143, 2026-08-10; default if you select
nothing is your two highest-ranked public entries). The challenge Rules tab still
says "a single submission"; the contradiction was raised on the forum and is
unresolved, so plan for two and confirm before any selection deadline.** Public board rank is not the target. There is a second, now co-equal
objective: the **$20k contribution prize**, whose deliverable is the write-up
(`docs/writeup-draft.md`).

Where we stand, what is open, and what to do next: **`docs/CURRENT_STATE.md`**.

## Layout

| path | what |
|---|---|
| `estimator.py` | **THE submission.** Self-contained; `whest submit` ships only this file. Currently **v2.6.1**, byte-identical to `estimator_v2_6_1.py` modulo line endings (`preflight` step 4 asserts it). The **graded** file is still v2.4.1 — v2.6.x is built and deliberately un-gated. |
| `estimator_v*.py` | frozen version archive — never edited, only read |
| `local_sim/` | the offline lab: numpy shims for `flopscope`/`whestbench`, grader-identical mint (`mint_provenance.py`), invariants (`edge_cases.py`), seed registry (`seed_blocks.py`), sealed holdout (`vault.py`), probes |
| `experiments/` | `LOG.md` (append every experiment), run reports, vault seal, JSON artifacts |
| `docs/` | durable knowledge — see pointers below |
| `ai/workflows/` | **the canonical workflow layer.** The six human commands, plus `slice-contract` and the `skeptic` brief. Agent-independent |
| `scripts/` | shared executable enforcement: `preflight.ps1` (the change gate), `guard_paths.py`, `human_model_state.py`, `git-hooks/`, `verify_portability.py` |

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
.\scripts\preflight.ps1                                   # THE gate: provenance + 30 invariants + contract + ship-file identity + vault seal
uv run whest validate --estimator estimator.py            # contract only (~46 ms)
uv run whest run --estimator estimator.py --dataset hf://aicrowd/arc-whestbench-public-2026@v1-phase1 --split mini --runner subprocess
uv run python local_sim/phase2_checklist.py               # rounds / dataset tags / boards
uv run python local_sim/seed_blocks.py                    # allocate a block BEFORE any experiment
.venv-gpu\Scripts\python.exe local_sim/vault.py verify    # seal integrity + touch log
uv run python scripts/human_model_state.py                # BLANK/PARTIAL/STALE/FILLED — drives session-start
uv run python scripts/verify_portability.py               # the harness is still swappable between agents
python scripts/install_git_hooks.py                       # install the agent-independent pre-commit backstop
```

Full reference (GPU venv, probes, dead-lane commands): `docs/SETUP_WINDOWS.md` §7.
**Run `whest` from PowerShell**, or set `PYTHONIOENCODING=utf-8` in a POSIX shell — the
CLI's box-drawing output kills a cp1252 console inside its own error printer.

## Human workflow — identical under every agent

```text
Start / re-enter a session          session-start     (blank HUMAN_MODEL.md → first-orientation)
Choose the next high-value move     next-slice        (may be a diagnostic, not code)
After an IMPORTANT/STRATEGIC slice  review-slice
Periodically check the mental model cold-start
Mechanical work                     routine <task>
End the work session                session-close
```

Daily loop: `session-start` → `next-slice` → work → `review-slice` → repeat →
`session-close`. Run `cold-start` every 3–5 strategic slices.

**The authoritative definitions are in [`ai/workflows/`](ai/workflows/README.md)** —
one file per command. Every agent exposes them under its own invocation syntax
(a leading `/` in most), but the semantics come from those files and nowhere else.

Project-specific machinery runs *beneath* these: `ai/workflows/slice-contract.md`
(pre-registration), `ai/workflows/skeptic.md` (independent falsification),
`scripts/preflight.ps1`, `local_sim/seed_blocks.py`, `local_sim/vault.py`, and the
probes. Do not invoke them as a substitute for the workflow above.

When `next-slice` or `review-slice` emits **STRATEGIC_RESEARCH** or
**STRATEGIC_DECISION**, it produces a Strategist Packet — send that to the strongest
reasoning capability available and bring the returned Slice Contract back through
`next-slice`. The packet is self-contained by construction: the recipient may be any
model, any vendor, or a human, and has neither repo access nor shared chat history.

**`docs/HUMAN_MODEL.md` is human-owned and must not be edited by agents.** Enforced
mechanically (see § *Enforcement*), not by this sentence.

## Work classification

| class | what | required |
|---|---|---|
| **ROUTINE** | doc edits, figures, board snapshots, report fetching, new tests, lab scripts that touch no seed block and no shipped file | just run it; verification only |
| **IMPORTANT** | anything changing *the instrument* — `harness.py`, `mint_provenance.py`, `edge_cases.py`, `seed_blocks.py`, `vault.py`, a probe's math, grader-report parsing, **or the workflow/enforcement layer (`ai/workflows/`, `scripts/`)** | written prediction first, `scripts/preflight.ps1` green, LOG entry with a knowledge delta |
| **STRATEGIC** | `estimator.py`; spending a decision block; a vault touch; a submission; changing a gate, a residual model, or the three-tier protocol; a write-up claim; reopening a closed lane | **Slice Contract** pre-registered in LOG before data exists → Samuel's prediction → build → evidence packet → independent Skeptic → teach-back. Use `ai/workflows/slice-contract.md`. |

**Never let one strategic result flow into the next implementation without
synthesis.** Closed lanes stay closed unless new *external* evidence reopens them —
the reopening condition goes in the contract.

## Continuity — no agent may depend on another agent's chat history

The driving agent changes. Chat history does not transfer, and must never need to.
Everything required to resume lives in the repository:

| what | where |
|---|---|
| where we stand, closed lanes, next actions | `docs/CURRENT_STATE.md` |
| every experiment, contract, result, Skeptic verdict | `experiments/LOG.md` |
| the binding measurement protocol | `docs/VALIDATION.md` |
| traps not to rediscover | `docs/GOTCHAS.md` |
| strategy (rev 12 authoritative) | `docs/GAME_PLAN.md` |
| the human's own beliefs | `docs/HUMAN_MODEL.md` |
| raw evidence | JSON under `experiments/` |
| evaluation state | vault seal, seed-block registry, run reports |

**Switching drivers is exactly:** start the new agent in the repository root, invoke
`session-start`, continue with `next-slice`. If that is not sufficient, the previous
session failed to close properly — that is a `session-close` defect, not a reason to
consult a transcript.

## Enforcement — mechanical, not prose

Rules this repository considers load-bearing are executable, and the implementation is
shared so both agents enforce the same thing:

| invariant | implementation | wired into |
|---|---|---|
| `docs/HUMAN_MODEL.md` is never written by an agent | `scripts/guard_paths.py` | `.claude/hooks/`, `.codex/hooks/`, `scripts/git-hooks/pre-commit` |
| the vault seal is moved only by `vault.py` | `scripts/guard_paths.py` | same three |
| `whest submit` requires explicit confirmation | `scripts/guard_paths.py` | both agent hooks |
| `vault.py evaluate` requires explicit confirmation | `scripts/guard_paths.py` | both agent hooks |
| vault counter == touch-log line count | `scripts/preflight.ps1` #5, `scripts/git_precommit_guard.py` | preflight, pre-commit |
| the shipped file matches its version archive | `scripts/preflight.ps1` #4 | preflight |
| the harness stays agent-swappable | `scripts/verify_portability.py` | run after any harness edit |

The git pre-commit hook is the **agent-independent backstop**: it holds even for an
agent whose hook system nobody has wired yet. Install it once with
`python scripts/install_git_hooks.py`. It refuses to commit a protected path *quietly*;
`ARC_WHEST_PROTECTED_OK=1` is the conscious human override, and Samuel needs it when
committing his own `HUMAN_MODEL.md`.

## Capabilities this project actually needs

Stated as capabilities, not product names, so any agent can satisfy them. An agent
missing one of the first four cannot drive a strategic slice.

| capability | how it is satisfied here | required for |
|---|---|---|
| repository read + content search | any agent's native file/grep tools | everything |
| shell execution (PowerShell **and** a POSIX shell) | native | `whest`, `preflight.ps1`, `uv`, hooks |
| **the evaluator** | this repo's own scripts — `preflight.ps1`, `edge_cases.py`, `mint_provenance.py`, `vault.py`, `whest run` | every gate |
| an **independent subagent context** for the Skeptic | `.claude/agents/skeptic.md`, `.codex/agents/skeptic.md` | `review-slice` step 2 |
| version control | `git`, `gh` | provenance for every LOG entry |
| escalation to a stronger reasoning capability | the Strategist Packet, delivered by hand | STRATEGIC_RESEARCH / DECISION |

**Deliberately NOT required, and not to be added without a named bottleneck**:
autonomous optimization loops (**forbidden — the cost half of the evaluator is E1**),
experiment trackers (MLflow / Phoenix / Langfuse), docs-retrieval MCPs, semantic-code
MCPs, repo-packing tools, prompt-eval harnesses, browser automation, web-research
MCPs, GitHub MCP, multi-agent orchestration. Each was considered and rejected for a
stated reason: `docs/CURRENT_STATE.md` § *Harness*. An agent that lacks these is not
disadvantaged here; an agent that uses them is deviating from the contract.

## Agent adapters

| agent | instruction discovery | skills | subagents | hooks |
|---|---|---|---|---|
| any | `AGENTS.md` (this file) | `ai/workflows/` | `ai/workflows/skeptic.md` | `scripts/guard_paths.py` + `scripts/git-hooks/` |
| Claude Code | `CLAUDE.md` (imports this file) | `.claude/skills/<name>/SKILL.md` | `.claude/agents/skeptic.md` | `.claude/settings.json` → `.claude/hooks/` |
| Codex | `AGENTS.md` directly | `.agents/skills/<name>/SKILL.md` | `.codex/agents/skeptic.md` | `.codex/hooks/` (see `.codex/hooks/README.md`) |

**Working-directory assumption: every agent is started from the repository root**
(`c:\Users\samsu\Documents\arc-whest`). All paths in this file, in `ai/workflows/`,
and in every adapter are repo-relative and assume that root. There are no nested
`AGENTS.md` files and no directory-scoped skill overrides; if one is ever added, the
more deeply nested file wins for work inside its directory, and that must be recorded
here. Starting an agent from a subdirectory may prevent skill discovery entirely.

Adapters are thin by rule. If an adapter contains project behaviour that is not in
`ai/workflows/` or `docs/`, that is a portability bug — `scripts/verify_portability.py`
checks for it.

## Working style

Samuel has a CS degree but is new to ML/probability — explain what you're doing in
plain English in the final summary, not in jargon. Concentrate his understanding on
the *decision-critical* path (the score identity, the data protocol, the residual
model); do not make him learn commodity code.

Two roles exist regardless of which products fill them: a **repo-local Builder** (the
driving agent, doing the work in this tree) and an **independent Skeptic** (a separate
context that never saw the Builder's reasoning). A third, the **strategist**, is
consulted by packet when repository evidence is exhausted. Handoffs between all of
them go through **Slice Contracts and Evidence Packets in `experiments/LOG.md`**,
never chat history.

## Pointers

| doc | when to read it |
|---|---|
| `docs/CURRENT_STATE.md` | **first, always.** Baseline, open questions, closed lanes, next action |
| `ai/workflows/README.md` | the six commands and what they mean |
| `docs/VALIDATION.md` | before measuring anything. The protocol and its proofs |
| `docs/GOTCHAS.md` | before quoting a number, comparing a cost, or trusting an offline path |
| `docs/GAME_PLAN.md` | strategy. **Rev 12 is authoritative**; earlier revs are history |
| `docs/BENCHMARK.md` | our numbers vs the field. **PRIVATE — never the forum** |
| `docs/COMPETITION_BRIEF.md` | plain-English explainer of the task and the math |
| `docs/RESEARCH_NOTES.md` | rules, forum intel, pricing table, links. §2g is **PRIVATE** |
| `docs/MOONSHOT.md` | the cumulant-flow lane (Phase D, the contribution prize) |
| `docs/HUMAN_MODEL.md` | **Samuel's file.** Agents read and critique it; agents never write it |
| `experiments/LOG.md` | the full narrative record, newest at the bottom |

## Re-audit this harness when

The competition phase changes · the evaluator's cost half becomes trustworthy · a
lane reopens · the write-up ships · agent/tool overhead becomes noticeable · a third
agent is added as a driver.
