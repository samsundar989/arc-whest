"""Resolve recorded artifact paths against one explicit run root and bytes."""

from __future__ import annotations

import hashlib
from pathlib import Path


def resolve(root, recorded, *, sha256=None, required=True):
    """Return an in-root file; refuse traversal, missing files or changed bytes.

    Backslashes in stored Windows paths and forward slashes in snapshot records
    name the same file. Absolute paths are accepted only within ``root``.
    """
    base = Path(root).resolve(strict=True)
    if not isinstance(recorded, str) or not recorded:
        raise ValueError("recorded artifact path is empty or invalid")
    candidate = Path(recorded.replace("\\", "/"))
    path = (candidate if candidate.is_absolute() else base / candidate).resolve()
    try:
        path.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"artifact escapes run root: {recorded}") from exc
    if required and not path.is_file():
        raise FileNotFoundError(f"artifact missing: {path}")
    if path.is_file() and sha256 is not None:
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != sha256:
            raise ValueError(f"artifact bytes differ: {recorded}")
    return path
