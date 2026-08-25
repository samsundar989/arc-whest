#!/usr/bin/env bash
set -u
echo "=== distro ==="
grep -E '^(NAME|VERSION)=' /etc/os-release
echo "=== kernel ==="
uname -srm
echo "=== python ==="
python3 --version
echo "=== RLIMIT_AS ==="
python3 - <<'PY'
import resource
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
print("RLIMIT_AS constant:", resource.RLIMIT_AS)
print("current soft/hard :", soft, hard)
print("RLIMIT_AS is available on this platform: True")
PY
echo "=== memory visible to the kernel ==="
free -g | head -2
