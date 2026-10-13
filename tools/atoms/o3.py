"""Lane O, HANDOFF 2026-10-12 Lane O step O3: more exact ages, no old tier.

    o3:<A>      (i) keep the A youngest lineages exact, DROP everything older; no old tier (`_SHARED_ON = False`)

The truncation runs at the end of every layer (the shipped generation-cap site), through a width-aware copy of
`_truncate` (the shipped one assumes every lineage is n wide, which KAS's H keep counts break). This matches the
cost model `x3lib.py` priced as old=False, gens=A (its width-aware truncate, plus `hcols`/`rows` carried, and
P3 asserted off as it is on v84). The per-lineage c4 list already trims itself to the tower (`[-out_tw.k:]`).

Not bit-identical. Anchors are on the v84 base (`fold+age6`) and touch no other O atom's lines. Prices, ops and
the twin arm (the built file, `--raw`): `experiments/2026-10-13_o/O3/RESULTS.md`.
"""
from tools.atoms import rep

_TRUNC = '''

def _truncate_w(tw, gens, n):
    """O3: `_truncate` for towers whose lineages have different widths (KAS H)."""
    if tw is None or gens is None or tw.k <= gens:
        return tw
    assert tw.A_D is None and tw.F is None, "O3 truncate: P2 separable tower, P3 off"
    g = int(gens)
    ws = _lin_widths(tw)
    keep = sum(ws[-g:])
    keepM = sum(ws[-g:-1]) if g > 1 else 0
    M = None if (tw.M is None or g <= 1) else tw.M[:, tw.M.shape[1] - keepM:]
    r = tw.L[0].shape[2]
    Ls = []
    drop = tw.k - g
    for Lg in tw.L:
        c = Lg.shape[0]
        if drop >= c:
            drop -= c
            continue
        Ls.append(Lg[drop:])
        drop = 0
    out = _Tower(None, tw.A_L[:, tw.A_L.shape[1] - keep:], M, tw.w2b[tw.w2b.shape[0] - keep:], g,
                 tw.Z[:, tw.Z.shape[1] - g * r:], Ls, tw.sb[tw.sb.shape[0] - keep:],
                 tw.eb[tw.eb.shape[0] - keep:])
    if tw.Y is not None:
        out.Y = tw.Y[:, tw.Y.shape[1] - 2 * g:]
        out.CA = tw.CA[tw.CA.shape[0] - keep:]
        out.CB = tw.CB[tw.CB.shape[0] - keep:]
    out.rows, out.hcols = tw.rows, tw.hcols
    return out
'''


def make(arg):
    A = int(arg)

    def f(src, z):
        assert "_P3_ON = False\n" in src
        src = rep(src, "_SHARED_ON = True\n",
                  "_SHARED_ON = False\n"
                  f"#: O3 (i): no old tier; the {A} youngest lineages are kept exact, older ones dropped.\n"
                  f"_O3_KEEP = {A}\n")
        src = rep(src, "    out_tw = _truncate(out_tw, gens, n)\n",
                  "    out_tw = _truncate_w(out_tw, _O3_KEEP if gens is None else gens, n)\n")
        return rep(src, "\n\n# ==========================================================================\n"
                        "# the cost model: priced from the shapes",
                   _TRUNC + "\n\n# ==========================================================================\n"
                            "# the cost model: priced from the shapes")
    return f


def apply(src, z):
    raise SystemExit("o3 needs an arm: o3:<A>")
