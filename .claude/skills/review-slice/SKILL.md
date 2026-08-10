---
name: review-slice
description: Close the epistemic loop on a completed IMPORTANT or STRATEGIC slice in arc-whest. Inspects raw evidence rather than the builder's narrative, runs the independent skeptic agent, writes the Knowledge Delta into durable docs, quizzes Samuel on mechanism, and ends in exactly one terminal state. Run after any slice before starting another strategic change.
---

# /review-slice — arc-whest (Claude adapter)

**Read `ai/workflows/review-slice.md` now and follow it exactly.** That file is the
canonical, agent-independent definition of this workflow — the disqualifying
pre-registration check, the ten questions, the Knowledge Delta table, the six terminal
states, and the promotion rule — shared with the Codex adapter at
`.agents/skills/review-slice/SKILL.md`. This file adds only Claude bindings. If the
two disagree, the canonical workflow wins.

## Claude bindings

- **Step 2, independent falsification**: launch the `skeptic` subagent with the Agent
  tool (`subagent_type: "skeptic"`). Pass the Slice Contract, the diff, the evaluation
  protocol, and the **raw artifact paths** — never your summary. The subagent gets a
  fresh context, which is the point; do not brief it with your reasoning.
  Relay its verdict verbatim into `experiments/LOG.md`.
- `.\scripts\preflight.ps1` and `whest` run through **PowerShell**.
- **Never Edit `docs/HUMAN_MODEL.md`** — the protected-path guard denies it under
  every agent, and the git pre-commit backstop denies it again. Propose corrections
  in chat, citing the contradicting evidence.
- Terminal state STRATEGIC_RESEARCH / STRATEGIC_DECISION → emit the Strategist Packet
  from `ai/workflows/next-slice.md` and stop.
