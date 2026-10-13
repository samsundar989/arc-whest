"""Patch atoms for `tools/build_v85.py`. See README.md for the contract."""


def rep(s: str, a: str, b: str) -> str:
    """Replace the single occurrence of anchor `a` by `b`; any other count is an error."""
    c = s.count(a)
    if c != 1:
        raise SystemExit(f"anchor occurs {c}x: {a[:90]!r}")
    return s.replace(a, b, 1)
