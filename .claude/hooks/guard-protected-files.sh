#!/usr/bin/env bash
# Claude Code PreToolUse guard for arc-whest -- interpreter shim only.
#
# The policy lives in scripts/guard_paths.py (shared with .codex/hooks/ and
# scripts/git-hooks/pre-commit). The Claude-shaped JSON lives in
# .claude/hooks/guard_hook.py. This file exists solely because settings.json
# invokes a bash command, and because a hook must not depend on `uv` or on any
# virtualenv being active.
#
# Fails OPEN on a missing interpreter: a broken hook must not brick the session.
# The git pre-commit backstop still blocks a protected-path write in that case.

set -u
repo="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

for candidate in \
    "$repo/.venv/Scripts/python.exe" \
    "$repo/.venv/bin/python" \
    python3 python py
do
    if command -v "$candidate" >/dev/null 2>&1; then
        exec "$candidate" "$repo/.claude/hooks/guard_hook.py"
    fi
done

echo "arc-whest guard: no Python interpreter found; guard skipped." >&2
exit 0
