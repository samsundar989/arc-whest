---
name: next-slice
description: Select and execute the single highest-value, epistemically sound next move in arc-whest. Inspects repository evidence first, decides whether the move is execution, diagnostic, evaluation repair, research, or a strategic decision — then classifies it ROUTINE/IMPORTANT/STRATEGIC and routes to the right internal machinery. This is the canonical human entry point for doing work; it wraps the slice-contract skill rather than duplicating it.
---

# next-slice — arc-whest (Codex adapter)

**Read `ai/workflows/next-slice.md` now and follow it exactly.** That file is the
canonical, agent-independent definition of this workflow — the five evidence
questions, the five routes, the classification table, and the Strategist Packet
format — shared with the Claude adapter at `.claude/skills/next-slice/SKILL.md`. This
file adds only Codex bindings. If the two disagree, the canonical workflow wins.

Read `AGENTS.md` first if it is not already in context; § *Work classification* is the
binding definition of ROUTINE / IMPORTANT / STRATEGIC.

## Codex bindings

- **ROUTINE** → invoke the `routine` skill, which loads `ai/workflows/routine.md`.
- **STRATEGIC** → invoke the `slice-contract` skill, which loads
  `ai/workflows/slice-contract.md`. Do not re-derive its protocol inline.
- **IMPORTANT** → stay here; written prediction, then `scripts/preflight.ps1` via
  **PowerShell**, then the LOG entry.
- `uv run python local_sim/seed_blocks.py` allocates a block and must run before any
  measurement that touches one.
- STRATEGIC_RESEARCH / STRATEGIC_DECISION → emit the Strategist Packet and **stop**.
  Do not answer it with web search or a research subagent: escalation is a deliberate
  hand-off to Samuel, who routes it to the strongest reasoning capability available
  (`AGENTS.md` § *Capabilities*). The packet must be self-contained — assume the
  recipient has no repository access and no shared history with you.
