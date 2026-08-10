---
name: session-close
description: Close a work session in arc-whest — make sure nothing important exists only in chat, docs/CURRENT_STATE.md is accurate, the working tree and vault state are clean, and the next session has a defined starting point. Different from /review-slice, which closes the epistemic loop on one experiment; this closes the human's session.
---

# /session-close — arc-whest (Claude adapter)

**Read `ai/workflows/session-close.md` now and follow it exactly.** That file is the
canonical, agent-independent definition of this workflow — the six checks and the
hand-off block — shared with the Codex adapter at
`.agents/skills/session-close/SKILL.md`. This file adds only Claude bindings. If the
two disagree, the canonical workflow wins.

## Claude bindings

- The governing test is: **could a different agent, with no access to this
  conversation, resume tomorrow from `session-start` alone?** Run this workflow before
  handing the project to Codex, not only at the end of a day.
- Record `Driver: Claude Code` in the hand-off block.
- `git status --short`, `preflight.ps1`, and `vault.py verify` run through
  **PowerShell**.
- If Samuel filled in `docs/HUMAN_MODEL.md` this session, remind him that committing
  it needs `ARC_WHEST_PROTECTED_OK=1 git commit ...` — the agent-independent
  pre-commit backstop refuses to let a protected path move quietly, and he is the
  legitimate override.
