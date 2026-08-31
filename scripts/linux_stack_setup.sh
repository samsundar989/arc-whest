#!/usr/bin/env bash
# Build a Linux venv carrying the OFFICIAL Phase-2 stack, for the real 8-GiB
# RLIMIT_AS enforcement Windows cannot perform.
set -euo pipefail
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
VENV="$HOME/.arcwhest-linux"

if [ ! -x "$VENV/bin/python" ]; then
  uv venv "$VENV" --python 3.12 2>&1 | tail -3
fi
uv pip install --quiet --python "$VENV/bin/python" \
    'whestbench==0.16.0' 'flopscope==0.12.0' 2>&1 | tail -5
"$VENV/bin/python" - <<'PY'
import importlib.metadata as md
import resource
import numpy
for p in ("whestbench", "flopscope", "numpy"):
    try:
        print(p, md.version(p))
    except Exception as e:
        print(p, "MISSING", e)
print("RLIMIT_AS", resource.getrlimit(resource.RLIMIT_AS))
import whestbench, flopscope
print("whestbench at", whestbench.__file__)
print("flopscope   at", flopscope.__file__)
PY
