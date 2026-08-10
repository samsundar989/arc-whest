# CLAUDE.md — arc-whest

**Claude Code adapter. Thin by rule.** The repository operating contract — objective,
layout, hard rules, measurement law, commands, work classification, capabilities,
pointers — is `AGENTS.md`, and it is imported below so it is always in context.
Nothing in this file may contradict it, and no project behaviour may live only here.

@AGENTS.md

---

## Claude-specific bindings

Everything below is *how Claude Code reaches* the shared harness. The semantics are
never here; they are in `ai/workflows/` and `docs/`.

### The six workflow commands

Invoke with a leading slash. Each `.claude/skills/<name>/SKILL.md` is a thin adapter
that loads the canonical file of the same name in `ai/workflows/`.

| command | canonical semantics |
|---|---|
| `/session-start` | `ai/workflows/session-start.md` |
| `/next-slice` | `ai/workflows/next-slice.md` |
| `/review-slice` | `ai/workflows/review-slice.md` |
| `/cold-start` | `ai/workflows/cold-start.md` |
| `/routine` | `ai/workflows/routine.md` |
| `/session-close` | `ai/workflows/session-close.md` |

`/slice-contract` is the internal primitive beneath `/next-slice`
(`ai/workflows/slice-contract.md`); invoke it directly only to resume a slice already
under contract.

### The Skeptic

`review-slice` step 2 requires an independent falsifier in a context that never saw
the builder's reasoning. Launch the **`skeptic` subagent** (`.claude/agents/skeptic.md`,
which loads `ai/workflows/skeptic.md`) via the Agent tool, with the Slice Contract,
the diff, the evaluation protocol, and the raw artifact paths — never a prose summary.

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
- After editing anything under `ai/workflows/`, `.claude/`, `.agents/`, or `.codex/`,
  run `uv run python scripts/verify_portability.py` — a Claude-only harness change is
  a regression, not a feature.
