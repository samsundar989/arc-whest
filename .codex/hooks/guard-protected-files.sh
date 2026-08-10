#!/usr/bin/env bash
# Codex pre-tool guard for arc-whest -- interpreter shim only.
#
# Mirror of .claude/hooks/guard-protected-files.sh. The policy lives in
# scripts/guard_paths.py; the Codex-shaped response lives in
# .codex/hooks/guard_hook.py. Nothing project-specific is duplicated here.
#
# Exit 0 = allow, non-zero = block (reason on stderr).
# Fails OPEN on a missing interpreter; the git pre-commit backstop still holds.

set -u
repo="${CODEX_PROJECT_DIR:-${CODEX_WORKSPACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"

for candidate in \
    "$repo/.venv/Scripts/python.exe" \
    "$repo/.venv/bin/python" \
    python3 python py
do
    if command -v "$candidate" >/dev/null 2>&1; then
        exec "$candidate" "$repo/.codex/hooks/guard_hook.py" "$@"
    fi
done

echo "arc-whest guard: no Python interpreter found; guard skipped." >&2
exit 0
