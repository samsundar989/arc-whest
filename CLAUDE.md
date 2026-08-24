# CLAUDE.md — arc-whest

**Claude Code adapter. Thin by rule.** The repository operating contract — objective,
layout, hard rules, measurement law, commands, work classification, capabilities,
pointers — is `AGENTS.md`, and it is imported below so it is always in context.
Nothing in this file may contradict it, and no project behaviour may live only here.

@AGENTS.md

---

## Claude-specific bindings

Everything below is *how Claude Code reaches* the shared harness. The semantics are
never here; they are in `docs/`.

### There are no workflow commands

The six slash commands (`/session-start`, `/next-slice`, `/review-slice`,
`/cold-start`, `/routine`, `/session-close`) and the `/slice-contract` primitive
were **removed on 2026-08-21**, along with `ai/workflows/` and both skills
directories. They restated what a careful agent does anyway, and in practice both
agents read the files instead of invoking them.

**What they wrapped is not gone.** Two rules survived because they were the part
that actually caught mistakes, and they are stated directly in `AGENTS.md`
§ *How work is done*:

- **Rule 1 — pre-register before you measure.** Question, evidence, hypothesis and
  mechanism, numbered predictions, the gate, the falsifier, the frozen residual
  model and the seed block, appended to `experiments/LOG.md` **and committed**
  before any data exists. Then Samuel's prediction, also before the result.
- **Rule 2 — an independent Skeptic** after every IMPORTANT or STRATEGIC slice.

To orient at the start of a session: read `docs/CURRENT_STATE.md`, then
`experiments/LOG.md` from the bottom. That is the whole procedure.

### The Skeptic

Rule 2 requires an independent falsifier in a context that never saw the builder's
reasoning. Launch the **`skeptic` subagent** (`.claude/agents/skeptic.md`, which
loads `docs/SKEPTIC.md`) via the Agent tool, with the contract, the diff, the
evaluation protocol, and the **raw artifact paths** — never a prose summary.

It is the one piece of the old workflow layer that demonstrably paid for itself:
on 2026-08-21 it retracted a lane closure made on a measurement of the wrong
method, and found an omitted cost column that changed the next checkpoint's design.

### Enforcement wiring

`.claude/settings.json` registers a `PreToolUse` hook on
`Edit|Write|NotebookEdit|Bash|PowerShell` that runs `.claude/hooks/guard-protected-files.sh`
→ `.claude/hooks/guard_hook.py` → **`scripts/guard_paths.py`**, which is the single
shared policy implementation (Codex and the git pre-commit backstop call the same
file). The `permissions.deny` entries in `settings.json` are belt-and-braces for the
same paths.

Do not add project rules to `settings.json` or to the hook shims. New invariants go
into `scripts/guard_paths.py`, where both agents get them at once.

### Tooling notes for this environment

- Two shells are available and take different syntax: **PowerShell** (primary — use it
  for `whest`, `preflight.ps1`, anything with box-drawing output) and **Bash**
  (POSIX scripts; set `PYTHONIOENCODING=utf-8` before `whest`).
- Prefer the dedicated Read/Glob/Grep tools over shelling out to `cat`/`find`/`rg`.
- `local_sim/` contains numpy **shims** named `flopscope` and `whestbench`. Never
  `sys.path.insert(0, "local_sim")` — that silently swaps the real packages for the
  shims and every "counted FLOP" becomes fiction. Append, never insert.
- After editing anything under `.claude/`, `.codex/`, or `scripts/`, run
  `uv run python scripts/verify_portability.py` — a Claude-only harness change is
  a regression, not a feature.
