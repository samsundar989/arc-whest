# CLAUDE.md — arc-whest

**Claude Code adapter. Thin by rule.** The repository operating contract is `AGENTS.md`,
imported below so it is always in context. Nothing here may contradict it, and no project
behaviour may live only here.

@AGENTS.md

---

## Claude-specific bindings

### Orientation

Read `docs/PLAN.md`, then `RESULTS.md`. That is the whole procedure. There are no
workflow slash commands; `docs/HANDOFF_2026-09-21.md` is the ordered work queue.

### The Skeptic

Kept for the two designated finals and for write-up numbers only, and only when Samuel
asks (`AGENTS.md` rule 6). Launch the **`skeptic` subagent**
(`.claude/agents/skeptic.md`, which loads `docs/SKEPTIC.md`) via the Agent tool, in a
context that never saw the builder's reasoning, with the diff, the evaluation protocol
and the **raw artifact paths** — never a prose summary.

`docs/SKEPTIC.md` predates the Sep 21 reset: where it assumes pre-registered contracts,
seed blocks, decision tiers or a 2× release gate, those are retired (`AGENTS.md` rule 6).
The falsification role is what survives.

### Enforcement wiring

`.claude/settings.json` registers a `PreToolUse` hook on
`Edit|Write|NotebookEdit|Bash|PowerShell` that runs `.claude/hooks/guard-protected-files.sh`
→ `.claude/hooks/guard_hook.py` → **`scripts/guard_paths.py`**, the single shared policy
implementation (Codex and the git pre-commit backstop call the same file). The
`permissions.deny` entries in `settings.json` are belt-and-braces for the same paths.

Do not add project rules to `settings.json` or to the hook shims. New invariants go into
`scripts/guard_paths.py`, where both agents get them at once.

`whest submit` trips an **ask** in the guard. Under `AGENTS.md` rule 3 a non-designated
canary that passes every PLAN §5 gate is pre-approved; confirm the gate table from
`RESULTS.md`, then proceed.

### Tooling notes for this environment

- Two shells, different syntax: **PowerShell** (primary — use it for `whest` and anything
  with box-drawing output) and **Bash** (POSIX scripts; set `PYTHONIOENCODING=utf-8`
  before `whest`).
- Prefer the dedicated Read/Glob/Grep tools over shelling out to `cat`/`find`/`rg`.
- `local_sim/` contains numpy **shims** named `flopscope` and `whestbench`. Never
  `sys.path.insert(0, "local_sim")` — that silently swaps the real packages for the shims
  and every "counted FLOP" becomes fiction. Append, never insert.
- After editing anything under `.claude/`, `.codex/` or `scripts/`, run
  `uv run python scripts/verify_portability.py` — a Claude-only harness change is a
  regression, not a feature.
