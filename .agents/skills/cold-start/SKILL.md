---
name: cold-start
description: Detect comprehension debt in arc-whest before it becomes strategically dangerous. Asks Samuel to explain the project from memory FIRST, then compares his answers against repository evidence and teaches only the gaps that affect strategic direction. Run every 3-5 strategic slices, after a major change, after days away, or when he feels like a passenger.
---

# cold-start — arc-whest (Codex adapter)

**Read `ai/workflows/cold-start.md` now and follow it exactly.** That file is the
canonical, agent-independent definition of this workflow — the nine questions, the
five-way classification, the gap-teaching table, and the comprehension-debt signals —
shared with the Claude adapter at `.claude/skills/cold-start/SKILL.md`. This file adds
only Codex bindings. If the two disagree, the canonical workflow wins.

## Codex bindings

- **Step 1 is ask-and-stop.** Present the questions and end your turn. Do not read
  files, do not preview answers, do not run tools first — pre-loading the evidence into
  the transcript is what destroys the measurement.
- Only after Samuel answers: read `docs/CURRENT_STATE.md`, the tail of
  `experiments/LOG.md`, `docs/GOTCHAS.md`, and the code.
- **Never write `docs/HUMAN_MODEL.md`** — propose the slots he should change himself.
  The protected-path guard denies the edit under every agent.
