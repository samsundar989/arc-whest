---
name: next-slice
description: Select and execute the single highest-value, epistemically sound next move in arc-whest. Inspects repository evidence first, decides whether the move is execution, diagnostic, evaluation repair, research, or a strategic decision — then classifies it ROUTINE/IMPORTANT/STRATEGIC and routes to the right internal machinery. This is the canonical human entry point for doing work; it wraps the slice-contract skill rather than duplicating it.
---

# /next-slice — arc-whest (Claude adapter)

**Read `ai/workflows/next-slice.md` now and follow it exactly.** That file is the
canonical, agent-independent definition of this workflow — the five evidence
questions, the five routes, the classification table, and the Strategist Packet
format — shared with the Codex adapter at `.agents/skills/next-slice/SKILL.md`. This
file adds only Claude bindings. If the two disagree, the canonical workflow wins.

## Claude bindings

- **ROUTINE** → invoke the `routine` skill (Skill tool), which loads
  `ai/workflows/routine.md`.
- **STRATEGIC** → invoke the `slice-contract` skill (Skill tool), which loads
  `ai/workflows/slice-contract.md`. Do not re-derive its protocol inline.
- **IMPORTANT** → stay here; written prediction, then `.\scripts\preflight.ps1` via
  **PowerShell**, then the LOG entry.
- Seed-block allocation and `preflight.ps1` are PowerShell commands.
- STRATEGIC_RESEARCH / STRATEGIC_DECISION → emit the Strategist Packet and **stop**.
  Do not use web search, research subagents, or workflow orchestration to answer it
  here; escalation is a hand-off to Samuel, by design (`AGENTS.md` § *Capabilities*).
