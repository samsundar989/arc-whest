# AGENTS.md — arc-whest

**The cross-agent repository operating contract.** Every coding agent that drives this
repository obeys this file, whatever product it is. Agent-specific files
(`CLAUDE.md`, `.claude/`, `.agents/`, `.codex/`) are thin adapters around it and must
never carry project behaviour of their own.

*A map and a contract, not an encyclopedia. Deeper knowledge lives in `docs/`.
Harness audit: 2026-08-07. Portability audit: 2026-08-07. Re-audit triggers at the bottom.*

## Objective

Samuel's entry for the **ARC White-Box Estimation Challenge 2026** (AIcrowd).
Given a random He-init ReLU MLP (live Phase 2: 1024 wide × 16 deep, no biases),
predict every neuron's expected post-ReLU activation under x ~ N(0, I) as a
`(depth, width)` array. Score = mean over MLPs of
`final_layer_MSE × max(0.1, FLOPs/2**41)`, lower is better. The official Phase-2
stack is WhestBench 0.16 / flopscope 0.12; lambda is zero; residual cap is 0.4 s,
total wall 120 s, solution memory 8 GiB.

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
| `estimator.py` | **THE submission.** Self-contained; `whest submit` ships only this file. Currently **v2.6.1**, byte-identical to `estimator_v2_6_1.py` modulo line endings (`preflight` step 5 asserts it). The **graded Phase-1** file is still v2.4.1. There is no Phase-2 submission; width 1024 currently takes the basic K2p1 analytic fallback. |
| `estimator_v*.py` | frozen version archive — never edited, only read |
| `local_sim/` | the offline lab: numpy shims for `flopscope`/`whestbench`, grader-identical mint (`mint_provenance.py`), invariants (`edge_cases.py`), seed registry (`seed_blocks.py`), sealed holdout (`vault.py`), probes |
| `experiments/` | `LOG.md` (append every experiment), run reports, vault seal, JSON artifacts |
| `docs/` | durable knowledge — see pointers below. `SKEPTIC.md` is the independent-falsifier brief both agents load |
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
  gate threshold, and cost model go into `experiments/LOG.md` *before* the block
  is touched. A model chosen after seeing the answer is not evidence.
- **Counted compares to counted; effective compares to effective.** Always pair on
  common random numbers; never quote a bare ratio of means.
- **The binding promotion statistic is the paired ARITHMETIC score difference**
  with the studentised-t or basic-bootstrap interval — the competition's own
  identity (adopted 2026-08-20, `docs/VALIDATION.md` §2b-i). The geometric mean of
  per-MLP ratios is a **stability diagnostic** with no authority over an outcome.
  The percentile bootstrap and BCa are **refused**: measured false-promotion rates
  6.8–8.2% and 9.1–11.5% against a nominal 5%. The gate is executable —
  `local_sim/competition_score.py::decide()` is the only function here that returns
  a promotion boolean.
- **Every MSE is quoted with its ground-truth noise floor.**
- **Phase 2 sets λ = 0, so `C = F` and residual is not a cost** — it is a 0.4 s
  per-MLP cliff that FAILS the MLP. Freeze a counted-FLOP model; report *headroom*,
  `0.4 / observed`. `VALIDATION.md` §2b is Phase-1 law, §2b-ii governs Phase 2.
- **Never quote a local absolute wall time.** Authority is `--runner subprocess`.
  The old "~7×" is local-vs-GRADED and has no Phase-2 counterpart; locally the
  three instruments agree to ~7% run-to-run, so a local margin is not a safety
  factor either way. The shim reads ~18% low for rotation-using estimators.
- **The local meter bills what the grader bills, op for op** (reconciled to zero
  2026-08-24; `preflight` step 12 *refuses* rather than skips if the official
  report is absent). **The 8,192 MB limit is unenforceable here** — Windows has no
  `RLIMIT_AS` — so it is accounting, never a test passed. `docs/GOTCHAS.md`.

## Commands

```powershell
uv sync                                                   # environment
.\scripts\preflight.ps1                                   # THE gate, 12 of 16 steps: official stack + provenance + 30 invariants + contract + ship-file identity + vault seal + FLOP anchor
.\scripts\preflight.ps1 -Full -Log <path>                  # all 16 + a COMPLETE transcript (2>&1 drops the verdicts; see GOTCHAS)
uv run whest validate --estimator estimator.py            # contract only (~46 ms)
uv run python local_sim/official_config.py                # the LIVE round, asserted against the installed package
uv run python local_sim/p2_flop_anchor.py --gate          # does the local meter bill what the grader bills? op by op
uv run whest run --estimator estimator.py --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 --split mini --runner subprocess
uv run python local_sim/phase2_checklist.py               # rounds / dataset tags / boards
uv run python local_sim/seed_blocks.py                    # allocate a block BEFORE any experiment
.venv-gpu\Scripts\python.exe local_sim/vault.py verify    # seal integrity + touch log
uv run python scripts/human_model_state.py                # BLANK/PARTIAL/STALE/FILLED — is the human model current?
uv run python scripts/verify_portability.py               # the harness is still swappable between agents
python scripts/install_git_hooks.py                       # install the agent-independent pre-commit backstop
```

Full reference (GPU venv, probes, dead-lane commands): `docs/SETUP_WINDOWS.md` §7.
**Run `whest` from PowerShell**, or set `PYTHONIOENCODING=utf-8` in a POSIX shell — the
CLI's box-drawing output kills a cp1252 console inside its own error printer.

## How work is done

*The six slash-command workflows (`session-start`, `next-slice`, `review-slice`,
`cold-start`, `routine`, `session-close`) and the `slice-contract` primitive were
removed on 2026-08-21. They were ceremony that described what a careful agent does
anyway, and in practice agents read the files rather than invoking them. **What
they wrapped was never the ceremony — it was two rules, and those are now stated
here directly and are still binding.***

**Start a session by reading `docs/CURRENT_STATE.md`.** It is the snapshot: where
we stand, what is open, which lanes are closed, and what to do next. Then
`experiments/LOG.md` from the bottom for the narrative. That is the whole
orientation procedure; there is no command for it.

### Rule 1 — pre-register before you measure

**Binding for every STRATEGIC slice** (see the classification table below), and it
is the rule this repository has lost the most to when skipped.

Before any data exists, append to `experiments/LOG.md` a dated entry containing:

| field | what |
|---|---|
| **Question** | what we are trying to learn |
| **Evidence** | what is already known, each item marked CONFIRMED / INFERRED / SPECULATIVE |
| **Hypothesis + mechanism** | what we believe and *why it should be true*. No mechanism means it is a search, not an experiment |
| **Intervention** | the smallest change that discriminates; one hypothesis |
| **Predictions** | numbered, with directions and rough magnitudes |
| **Gate** | the threshold, chosen NOW, passed to `competition_score.decide(gate=...)`, which has no default. **Write the gate on the DIFFERENCE the slice isolates, not on a total that contains it** — two gates in one session measured slightly the wrong quantity |
| **Falsifier** | what result rejects the hypothesis |
| **Baseline** | the exact control and its tier |
| **Residual model** | REQUIRED for any cost claim. Report three (`docs/VALIDATION.md` §2b), FREEZE one by name here, and make sure it can see the cost the candidate actually incurs — a counted-FLOPs freeze hides elementwise residual |
| **Statistic** | paired, common random numbers, arithmetic score difference, studentised-t or basic bootstrap. Never percentile or BCa |
| **Seed block** | claimed through `local_sim/seed_ledger.py`, never a literal |
| **Confounders / scope / what must not change** | |

Then **take Samuel's prediction and record it before the result exists.** Three
short questions, two minutes. It is the only record of whether the project's model
of itself is any good, and "I don't know" is a valid answer.

Commit the contract **before** running anything that touches the block, so git
timestamps prove it preceded the data.

### Rule 2 — an independent Skeptic before anything is believed

**Binding after every IMPORTANT or STRATEGIC slice, before any promotion,
submission, or write-up claim.**

Launch the `skeptic` subagent in a context that has **never seen the builder's
reasoning**, with the contract, the diff, the evaluation protocol and the **raw
artifact paths** — never a prose summary. The canonical brief is `docs/SKEPTIC.md`;
`.claude/agents/skeptic.md` and `.codex/agents/skeptic.md` are thin adapters.

It has earned its place: on 2026-08-21 it retracted a lane closure that had been
made on a measurement of a different method, and found an omitted cost column that
changed the next checkpoint's design.

### Everything else is judgement

Classify the work (below), do it, verify it, record it. Escalate the moment a task
turns out to touch `estimator.py`, a seed block, the vault, a gate, a residual
model, a closed lane, or a submission.

When repository evidence is exhausted, write a **Strategist Packet** — a
self-contained brief for a stronger reasoning capability, assuming no repo access
and no shared chat history — and bring the returned plan back as a contract.

**`docs/HUMAN_MODEL.md` is human-owned and must not be edited by agents.** Enforced
mechanically (see § *Enforcement*), not by this sentence.

## Work classification

| class | what | required |
|---|---|---|
| **ROUTINE** | doc edits, figures, board snapshots, report fetching, new tests, lab scripts that touch no seed block and no shipped file | just run it; verification only |
| **IMPORTANT** | anything changing *the instrument* — `harness.py`, `mint_provenance.py`, `edge_cases.py`, `seed_blocks.py`, `seed_ledger.py`, `competition_score.py`, `lifecycle_eval.py`, `vault.py`, a probe's math, grader-report parsing, **or the enforcement layer under `scripts/`** | written prediction first, `scripts/preflight.ps1` green, LOG entry with a knowledge delta |
| **STRATEGIC** | `estimator.py`; spending a decision block; a vault touch; a submission; changing a gate, a residual model, or the three-tier protocol; a write-up claim; reopening a closed lane | pre-registered contract in LOG before data exists (**Rule 1** above) → Samuel's prediction → build → evidence packet → independent Skeptic (**Rule 2**) |

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
| target architecture and sole active execution plan | `docs/GAME_PLAN.md` |
| copy-ready Claude Builder handoff | `docs/CLAUDE_CODE_HANDOFF.md` |
| the human's own beliefs | `docs/HUMAN_MODEL.md` |
| raw evidence | JSON under `experiments/` |
| evaluation state | vault seal, seed-block registry, run reports |

**Switching drivers is exactly:** start the new agent in the repository root and
have it read `docs/CURRENT_STATE.md`, then `experiments/LOG.md` from the bottom. If
that is not sufficient, the previous session failed to record something it should
have — that is a documentation defect, not a reason to consult a transcript.

## Enforcement — mechanical, not prose

Rules this repository considers load-bearing are executable, and the implementation is
shared so both agents enforce the same thing:

| invariant | implementation | wired into |
|---|---|---|
| `docs/HUMAN_MODEL.md` is never written by an agent | `scripts/guard_paths.py` | `.claude/hooks/`, `.codex/hooks/`, `scripts/git-hooks/pre-commit` |
| the vault seal is moved only by `vault.py` | `scripts/guard_paths.py` | same three |
| `whest submit` requires explicit confirmation | `scripts/guard_paths.py` | both agent hooks |
| `vault.py evaluate` requires explicit confirmation | `scripts/guard_paths.py` | both agent hooks |
| vault counter == touch-log line count | `scripts/preflight.ps1` #6, `scripts/git_precommit_guard.py` | preflight, pre-commit |
| the shipped file matches its version archive | `scripts/preflight.ps1` #5 | preflight |
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
| an **independent subagent context** for the Skeptic | `.claude/agents/skeptic.md`, `.codex/agents/skeptic.md` | Rule 2 — after every IMPORTANT/STRATEGIC slice |
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

| agent | instruction discovery | subagents | hooks |
|---|---|---|---|
| any | `AGENTS.md` (this file) | `docs/SKEPTIC.md` | `scripts/guard_paths.py` + `scripts/git-hooks/` |
| Claude Code | `CLAUDE.md` (imports this file) | `.claude/agents/skeptic.md` | `.claude/settings.json` → `.claude/hooks/` |
| Codex | `AGENTS.md` directly | `.codex/agents/skeptic.md` | `.codex/hooks/` (see `.codex/hooks/README.md`) |

*There is no skills layer. The six workflow commands were removed 2026-08-21; what
they enforced is Rules 1 and 2 above.*

**Working-directory assumption: every agent is started from the repository root**
(`c:\Users\samsu\Documents\arc-whest`). All paths in this file and in
every adapter are repo-relative and assume that root. There are no nested
`AGENTS.md` files and no directory-scoped skill overrides; if one is ever added, the
more deeply nested file wins for work inside its directory, and that must be recorded
here. Starting an agent from a subdirectory may prevent skill discovery entirely.

Adapters are thin by rule. If an adapter contains project behaviour that is not in
`docs/`, that is a portability bug — `scripts/verify_portability.py`
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
| `docs/VALIDATION.md` | before measuring anything. The protocol and its proofs |
| `docs/GOTCHAS.md` | before quoting a number, comparing a cost, or trusting an offline path |
| `docs/GAME_PLAN.md` | **sole active Phase-2 strategy and execution plan** |
| `docs/BENCHMARK.md` | our numbers vs the field. **PRIVATE — never the forum** |
| `docs/COMPETITION_BRIEF.md` | plain-English explainer of the task and the math |
| `docs/RESEARCH_NOTES.md` | rules, forum intel, pricing table, links. §2g is **PRIVATE** |
| `docs/CLAUDE_CODE_HANDOFF.md` | copy-ready prompt for Claude to execute the plan |
| `docs/MOONSHOT.md` | archived Phase-1 cumulant-flow plan; historical references only |
| `docs/HUMAN_MODEL.md` | **Samuel's file.** Agents read and critique it; agents never write it |
| `experiments/LOG.md` | the full narrative record, newest at the bottom |

## Re-audit this harness when

The competition phase changes · the evaluator's cost half becomes trustworthy · a
lane reopens · the write-up ships · agent/tool overhead becomes noticeable · a third
agent is added as a driver.
