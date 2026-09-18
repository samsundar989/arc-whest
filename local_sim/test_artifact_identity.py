"""Snapshot path, byte and traversal refusal checks."""

import hashlib
import tempfile
from pathlib import Path

from artifact_identity import resolve


def refuses(fn):
    try:
        fn()
    except (ValueError, FileNotFoundError):
        return True
    return False


with tempfile.TemporaryDirectory(prefix="arc_whest_artifact_") as temp:
    root = Path(temp)
    (root / "data").mkdir()
    file = root / "data" / "row.json"
    file.write_bytes(b"exact evidence\n")
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    assert resolve(root, "data/row.json", sha256=digest) == file
    assert resolve(root, "data\\row.json", sha256=digest) == file
    assert resolve(root, str(file), sha256=digest) == file
    assert refuses(lambda: resolve(root, "../outside.json"))
    assert refuses(lambda: resolve(root, "data/missing.json"))
    assert refuses(lambda: resolve(root, "data/row.json", sha256="0" * 64))

print("artifact identity: relative/Windows/absolute in-root paths agree; traversal, missing, bytes refuse")
