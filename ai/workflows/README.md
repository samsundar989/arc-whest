# ai/workflows — the canonical workflow layer

**These files are the source of truth for how work is done in arc-whest.** They are
agent-independent: no Claude-specific, Codex-specific, or model-specific syntax
appears in them, and none may be added.

Every agent adapter — `.claude/skills/*`, `.agents/skills/*` — is a **thin wrapper**
whose only job is to be discoverable by that agent, load the file below, and bind the
capability names (see `AGENTS.md` § *Capabilities*). Adapters must not restate,
paraphrase, or extend workflow content. If an adapter and a workflow disagree, the
workflow wins and the adapter is a bug.

## The six standard workflows — the human control surface

| workflow | when |
|---|---|
| [`session-start.md`](session-start.md) | entering the repo; returning after time away; weak mental model |
| [`next-slice.md`](next-slice.md) | choosing and executing the next highest-value move |
| [`review-slice.md`](review-slice.md) | after any IMPORTANT or STRATEGIC slice |
| [`cold-start.md`](cold-start.md) | every 3–5 strategic slices; comprehension check |
| [`routine.md`](routine.md) | mechanically verifiable work, no ceremony |
| [`session-close.md`](session-close.md) | end of a work session |

Daily loop: `session-start` → `next-slice` → work → `review-slice` → repeat →
`session-close`.

## Internal machinery — invoked *by* the workflows, not by Samuel

| file | role |
|---|---|
| [`slice-contract.md`](slice-contract.md) | pre-registration protocol for a STRATEGIC slice. Normally entered from `next-slice` |
| [`skeptic.md`](skeptic.md) | the independent falsifier's brief. Loaded by whatever agent/context runs the Skeptic role in `review-slice` step 2 |

## Rules for anyone editing this directory

1. **No agent, product, or model names** — not Claude, Codex, GPT, Opus, Fable. Refer
   to roles: *Builder*, *Skeptic*, *strategist*. Refer to capabilities, not tools.
   The one permitted exception is naming a *project* dependency (`uv`, `whest`,
   `flopscope`, PowerShell), because those are genuinely required.
2. **Workflow names have no leading slash here.** `session-start`, not
   `/session-start`. Adapters add whatever invocation sugar their agent uses.
3. **Mechanical invariants belong in `scripts/`, not in prose here.** If a rule must
   hold regardless of which agent is driving, it needs an executable enforcer
   (`scripts/guard_paths.py`, `scripts/git-hooks/pre-commit`,
   `scripts/preflight.ps1`) and the prose here merely explains it.
4. **Editing a workflow is IMPORTANT-class work** (it changes the instrument that
   governs every other change): written prediction, then
   `uv run python scripts/verify_portability.py` green, then a LOG entry.
