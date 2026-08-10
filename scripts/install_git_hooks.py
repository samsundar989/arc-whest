#!/usr/bin/env python
"""Install the agent-independent git hooks. Idempotent; safe to re-run.

    python scripts/install_git_hooks.py            install / repair
    python scripts/install_git_hooks.py --check    report only, exit 1 if missing

Copies rather than symlinks: symlink creation on Windows needs Developer Mode or
elevation, and a hook that silently fails to install is worse than a copy that has
to be refreshed.  scripts/verify_portability.py checks the copy is current.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCE_DIR = REPO / "scripts" / "git-hooks"
HOOKS = ["pre-commit"]


def hooks_dir() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--git-path", "hooks"],
        cwd=REPO, capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise SystemExit("not a git repository: %s" % REPO)
    p = Path(out.stdout.strip())
    return p if p.is_absolute() else REPO / p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    target_dir = hooks_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    problems = 0

    for name in HOOKS:
        src, dst = SOURCE_DIR / name, target_dir / name
        if not src.exists():
            print("MISSING SOURCE  %s" % src)
            problems += 1
            continue
        current = dst.read_bytes() if dst.exists() else None
        wanted = src.read_bytes()
        if current == wanted:
            print("ok              %s" % dst)
            continue
        if args.check:
            print("OUT OF DATE     %s  (run: python scripts/install_git_hooks.py)" % dst)
            problems += 1
            continue
        shutil.copyfile(src, dst)
        dst.chmod(0o755)
        print("installed       %s" % dst)

    if problems:
        return 1
    print("\ngit hooks current. The pre-commit backstop holds regardless of which "
          "agent is driving,\nand regardless of whether that agent's own hook "
          "system is wired.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
