# .codex/hooks — Codex enforcement adapter

## What is here

| file | role |
|---|---|
| `guard-protected-files.sh` | interpreter shim — finds a Python and execs the adapter |
| `guard_hook.py` | thin adapter: shared decision → Codex hook contract |

Both are thin. **The policy lives in `scripts/guard_paths.py`**, shared with
`.claude/hooks/` and `scripts/git-hooks/pre-commit`. Add a new invariant there and all
three enforcement points get it at once. Never add a project rule to this directory.

## What it enforces

Identical to the Claude adapter, because it is the same implementation:

| rule | verdict |
|---|---|
| write to `docs/HUMAN_MODEL.md` | **deny** — Samuel-owned |
| write to `experiments/vault/manifest.json` or `touch_log.jsonl` | **deny** — the seal moves only via `vault.py` |
| a shell command that redirects/copies/moves onto any of the above | **deny** |
| `whest submit` | **ask** — irreversible, needs Samuel's explicit approval |
| `vault.py evaluate` | **ask** — spends one of three permitted holdout touches |

## Contract this adapter honours

Because the hook payload and response schema differ across Codex versions, the adapter
satisfies every common contract simultaneously:

- reads a JSON payload on **stdin**, finding tool arguments nested at any depth under
  any of the usual key names (`file_path`, `path`, `command`, …);
- also accepts `--path` / `--command` flags if the runner passes arguments instead;
- writes `{"decision": "...", "rule": "...", "reason": "..."}` to **stdout**;
- writes the human-readable reason to **stderr**;
- exits **0 to allow, 1 to deny, 2 to ask** — so a runner that only checks
  zero/non-zero still blocks, which is the safe direction.

Verify it by hand at any time — this needs no Codex installed:

```bash
python .codex/hooks/guard_hook.py --path docs/HUMAN_MODEL.md          # exit 1, deny
python .codex/hooks/guard_hook.py --command "uv run whest submit"     # exit 2, ask
python .codex/hooks/guard_hook.py --path docs/CURRENT_STATE.md        # exit 0, allow
```

`uv run python scripts/verify_portability.py` runs exactly these and more.

## Wiring — the one step that needs your Codex version

**Status: NOT VERIFIED IN THIS ENVIRONMENT.** Codex is not installed on this machine,
so the registration key names below could not be exercised. Check them against
`codex --help` / your Codex docs before trusting them, and correct this file if they
differ. The adapter itself is verified; only the registration is not.

Deliberately *not* written into `.codex/config.toml`: a config file with a key your
Codex build rejects can prevent it from starting, which is a worse failure than an
unwired hook. Paste this in once you have confirmed the schema:

```toml
# ~/.codex/config.toml, or the repo's .codex/config.toml
[[hooks.pre_tool_use]]
command = ["bash", ".codex/hooks/guard-protected-files.sh"]
timeout_ms = 10000
```

## If your Codex build has no hook system at all

The invariants still hold, because the agent hook is the *fast* path, not the
guarantee. The guarantee is `scripts/git-hooks/pre-commit`, which runs inside git and
therefore holds for any agent, any editor, and any future tool:

```bash
python scripts/install_git_hooks.py
```

It refuses to let a protected path be committed *quietly*.
`ARC_WHEST_PROTECTED_OK=1` is the conscious human override — Samuel needs it when
committing his own `HUMAN_MODEL.md` or a logged vault touch.

Without the pre-tool hook you lose the two **ask** prompts (`whest submit`,
`vault.py evaluate`), which are advisory confirmations rather than integrity
invariants; they remain prose rules in `AGENTS.md` § *Hard rules* and in
`ai/workflows/routine.md`'s escalation table. Record in `experiments/LOG.md` that the
session ran without them.
