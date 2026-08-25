#!/usr/bin/env bash
set -eu
VENV="$HOME/.venv-whest-p2"
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV" 2>/dev/null || {
    echo "python3-venv missing; trying ensurepip"; python3 -m ensurepip --user || true
    python3 -m venv "$VENV"; }
fi
"$VENV/bin/python" -m pip install --quiet --upgrade pip
"$VENV/bin/python" -m pip install --quiet "whestbench==0.16.0" "flopscope==0.12.0" "numpy==2.2.6"
echo "=== installed ==="
"$VENV/bin/python" - <<'PY'
import importlib.metadata as md, sys
print("python  ", sys.version.split()[0])
for p in ("whestbench", "flopscope", "numpy"):
    try:
        print(f"{p:<11}", md.version(p))
    except Exception as e:
        print(f"{p:<11}", "MISSING", e)
import resource
print("RLIMIT_AS available:", hasattr(resource, "RLIMIT_AS"))
PY
echo "VENV=$VENV"
