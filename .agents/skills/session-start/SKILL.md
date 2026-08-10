---
name: session-start
description: Recover project state and orient Samuel before any strategic work in arc-whest. Run this first when entering the repo for a new session, returning after time away or from another project, or when the mental model is weak. Reads durable state only — never implements. Automatically enters FIRST-ORIENTATION MODE when docs/HUMAN_MODEL.md is blank or stale.
---

# session-start — arc-whest (Codex adapter)

**Read `ai/workflows/session-start.md` now and follow it exactly.** That file is the
canonical, agent-independent definition of this workflow, shared with the Claude
adapter at `.claude/skills/session-start/SKILL.md`. This file adds only Codex
bindings. If the two disagree, the canonical workflow wins.

Read `AGENTS.md` first if it is not already in context — it is the repository
operating contract and it governs everything below.

## Codex bindings

- The blank/stale test in Step 2 is **mechanical, not a judgement call**:
  `uv run python scripts/human_model_state.py`. BLANK, PARTIAL, or STALE →
  FIRST-ORIENTATION MODE. Do not decide this by eye; the branch must be identical
  under every agent.
- **This is a handoff entry point.** If the previous session was driven by another
  agent, you have no access to its transcript and must not ask for one. Reconstruct
  everything from `docs/CURRENT_STATE.md`, the tail of `experiments/LOG.md`, and
  `git log`. The last `SESSION CLOSE` block in the LOG names the previous driver and
  the intended next move.
- Run `whest` and `scripts/preflight.ps1` through **PowerShell**
  (`powershell -File .\scripts\preflight.ps1`, or `pwsh`). Under a POSIX shell, set
  `PYTHONIOENCODING=utf-8` first — the CLI's box-drawing output breaks a cp1252
  console inside its own error printer.
- This workflow is read-only. Write nothing, including `docs/HUMAN_MODEL.md` — the
  protected-path guard denies it, and that is correct.
- End by naming the next command (`next-slice`, `cold-start`, or `routine <task>`).
