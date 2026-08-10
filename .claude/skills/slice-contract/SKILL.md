---
name: slice-contract
description: INTERNAL PRIMITIVE, normally invoked by /next-slice rather than directly. Owns the pre-registration protocol for a STRATEGIC slice in arc-whest — allocate the seed block, write the contract and freeze the residual model into experiments/LOG.md before any data exists, take Samuel's prediction, spend the decision block once, return an Evidence Packet. Invoke directly only when resuming a slice already under contract. For choosing what to work on, use /next-slice; for closing the loop afterwards, use /review-slice.
---

# /slice-contract — arc-whest (Claude adapter)

**Read `ai/workflows/slice-contract.md` now and follow it exactly.** That file is the
canonical, agent-independent definition of this primitive — the strategic test, the
seed-block rules, the contract template, Samuel's prediction checkpoint, the
measure-once rule, the Evidence Packet format, and the standing prohibitions — shared
with the Codex adapter at `.agents/skills/slice-contract/SKILL.md`. This file adds only
Claude bindings. If the two disagree, the canonical workflow wins.

## Claude bindings

- `uv run python local_sim/seed_blocks.py` and `.\scripts\preflight.ps1` run through
  **PowerShell**.
- Write the contract into `experiments/LOG.md` with Edit, and **stop there** until
  Samuel has given his prediction. Do not run anything that touches the block first.
- Record `Driver: Claude Code` in the contract's Driver field.
- The slice ends at the Evidence Packet. **Do not launch the `skeptic` subagent from
  here** — that belongs to `/review-slice`, so that the review is not run by the same
  turn that produced the work.
- `vault.py evaluate` and `whest submit` will trigger a confirmation prompt from the
  shared guard. That prompt is the protocol, not an obstacle.
