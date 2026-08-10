---
name: review-slice
description: Close the epistemic loop on a completed IMPORTANT or STRATEGIC slice in arc-whest. Inspects raw evidence rather than the builder's narrative, runs the independent skeptic agent, writes the Knowledge Delta into durable docs, quizzes Samuel on mechanism, and ends in exactly one terminal state. Run after any slice before starting another strategic change.
---

# review-slice — arc-whest (Codex adapter)

**Read `ai/workflows/review-slice.md` now and follow it exactly.** That file is the
canonical, agent-independent definition of this workflow — the disqualifying
pre-registration check, the ten questions, the Knowledge Delta table, the six terminal
states, and the promotion rule — shared with the Claude adapter at
`.claude/skills/review-slice/SKILL.md`. This file adds only Codex bindings. If the two
disagree, the canonical workflow wins.

## Codex bindings

- **Step 2, independent falsification**: spawn the project-scoped `skeptic` agent
  defined at `.codex/agents/skeptic.md`, which loads `ai/workflows/skeptic.md`. Pass
  the Slice Contract, the diff, the evaluation protocol, and the **raw artifact
  paths** — never your summary. It must run in a fresh context that has not seen your
  reasoning; that independence is the entire value.
  If subagent spawning is unavailable in your configuration, run the Skeptic as an
  explicitly separate session and record in the LOG entry that it was run that way.
  Do not skip it and do not self-review: the chain is Builder → objective evaluator →
  independent Skeptic → synthesis.
  Relay the verdict verbatim into `experiments/LOG.md`.
- `scripts/preflight.ps1` and `whest` run through **PowerShell**.
- **Never write `docs/HUMAN_MODEL.md`** — the protected-path guard denies it under
  every agent, and the git pre-commit backstop denies it again. Propose corrections in
  chat, citing the contradicting evidence.
- Terminal state STRATEGIC_RESEARCH / STRATEGIC_DECISION → emit the Strategist Packet
  from `ai/workflows/next-slice.md` and stop.
