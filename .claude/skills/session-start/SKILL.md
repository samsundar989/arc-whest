---
name: session-start
description: Recover project state and orient Samuel before any strategic work in arc-whest. Run this first when entering the repo for a new session, returning after time away or from another project, or when the mental model is weak. Reads durable state only — never implements. Automatically enters FIRST-ORIENTATION MODE when docs/HUMAN_MODEL.md is blank or stale.
---

# /session-start — arc-whest (Claude adapter)

**Read `ai/workflows/session-start.md` now and follow it exactly.** That file is the
canonical, agent-independent definition of this workflow, shared with the Codex
adapter at `.agents/skills/session-start/SKILL.md`. This file adds only Claude
bindings. If the two disagree, the canonical workflow wins.

## Claude bindings

- The blank/stale test in Step 2 is **mechanical, not a judgement call**:
  `uv run python scripts/human_model_state.py`. BLANK, PARTIAL, or STALE →
  FIRST-ORIENTATION MODE.
- Run `whest`-family commands (`whest`, `preflight.ps1`) through **PowerShell**, not
  Bash — the CLI's box-drawing output breaks a cp1252 console inside its own error
  printer.
- Use Read/Glob/Grep for the durable-state sweep rather than shelling out.
- This workflow is read-only. Do not Edit or Write anything, including
  `docs/HUMAN_MODEL.md` — the protected-path guard will deny it, and that is correct.
- End by naming the next command (`/next-slice`, `/cold-start`, or `/routine <task>`).
