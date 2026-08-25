#!/usr/bin/env bash
# The FULL deployment-envelope suite on Linux, where RLIMIT_AS exists, so the
# memory clause of P2-P0 gate G4 is provoked by the SHIPPED code path and not
# only by a standalone script.
set -u
REPO=/mnt/c/Users/samsu/Documents/arc-whest
VENV="$HOME/.venv-whest-p2"
export HF_HOME=/mnt/c/Users/samsu/.cache/huggingface
export PYTHONIOENCODING=utf-8
cd "$REPO"
echo "=============================================================="
echo "DEPLOYMENT ENVELOPE on Linux -- the memory path FIRES here"
echo "=============================================================="
"$VENV/bin/python" - <<'PY'
import importlib.metadata as md, sys, platform, resource
print("platform  ", sys.platform, platform.release())
print("python    ", sys.version.split()[0])
for p in ("whestbench","flopscope","numpy"):
    print(f"{p:<10}", md.version(p))
print("RLIMIT_AS :", hasattr(resource,"RLIMIT_AS"))
PY
echo
"$VENV/bin/python" local_sim/test_deployment_envelope.py --no-write
echo "envelope exit: $?"
