---
name: routine
description: Execute a clearly scoped, mechanically verifiable task in arc-whest without strategic ceremony — docs, figures, board snapshots, report fetching, tests, refactors, and lab scripts that touch no seed block and no shipped file. Escalates automatically and stops if the task turns out to change strategy, the instrument, evaluation, or a claim.
---

# /routine — arc-whest (Claude adapter)

**Read `ai/workflows/routine.md` now and follow it exactly.** That file is the
canonical, agent-independent definition of this workflow — the scope list, the
five-step loop, the escalation table, and the standing rules — shared with the Codex
adapter at `.agents/skills/routine/SKILL.md`. This file adds only Claude bindings. If
the two disagree, the canonical workflow wins.

## Claude bindings

- Escalation triggers are hard stops: say what you found, stop, hand back to
  `/next-slice`. Do not invoke `slice-contract` from inside this skill.
- `.\scripts\preflight.ps1`, `whest`, and the vault commands run through
  **PowerShell**. Lab scripts run fine under either shell.
- If you touched `ai/workflows/`, `.claude/`, `.agents/`, or `.codex/`, run
  `uv run python scripts/verify_portability.py` — and note that workflow edits are
  IMPORTANT, not routine, so they also need a written prediction and a LOG entry.
- The protected-path guard will deny an edit to `docs/HUMAN_MODEL.md` or the vault
  seal, and will ask before `whest submit` or `vault.py evaluate`. Those prompts are
  the escalation triggers firing mechanically — treat them as such, not as friction.
