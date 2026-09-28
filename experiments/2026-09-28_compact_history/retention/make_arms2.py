"""Second retention/elasticity batch (diagnostics, not candidates), from v72 3b6cc397.

  age3_r128 / age3_r256 : the retention curve of the optimistic projection bound.
  age3_r64p3            : rank 64 with 3 range-finder power passes (is basis quality the limit?).
  d21age{m}             : NEW MECHANISM (E5 fallback, cheaper D21 contraction). `_dslice21`
                          contracts only the youngest m lineages (legs, P2 thin leg, hubs);
                          D3 still reads every lineage; the old tier is not read by D21. The
                          birth still subtracts w1^3 * (the D21 actually read), so the unread
                          slice content propagates through the tower's linear Wick scaling and
                          only its nonlinear feedback (covariance / closure / corrector) is lost.
                          The shared join's consumption weights for the joiner are still formed
                          (elementwise) so the old tier's basis rule is unchanged.
"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "estimator_k3_v72_gt.py"
src = SRC.read_text(encoding="utf-8")
assert hashlib.sha256(SRC.read_bytes()).hexdigest().startswith("3b6cc397")


def sub(s, a, b):
    assert s.count(a) == 1, a
    return s.replace(a, b, 1)


D21_ANCHOR = """        MP, PP, PPL = _p2_recon(tw)
        LA = 2.0 * (A_L * M) * tw.w2b + PP * tw.eb
"""
D21_NEW = """        MP, PP, PPL = _p2_recon(tw)
        if _D21_AGE and tw.k > _D21_AGE and tw.F is None:
            return _dslice21_young(tw, off_mask, A_L, M, MP, PP, PPL)
        LA = 2.0 * (A_L * M) * tw.w2b + PP * tw.eb
"""
YOUNG = '''

#: D21 read age: contract only the youngest `_D21_AGE` lineages (0 = all).
_D21_AGE = {m}


def _dslice21_young(tw, off_mask, A_L, M, MP, PP, PPL):
    """`_dslice21` over the youngest `_D21_AGE` lineages only (same term algebra)."""
    n, k, m = M.shape[0], tw.k, int(_D21_AGE)
    r = tw.L.shape[2]
    hub = _hub_parts(tw) if tw.Y is not None else None

    def legs(c0, c1):
        A, Mm, w2, eb, sb = A_L[:, c0:c1], M[:, c0:c1], tw.w2b[c0:c1], tw.eb[c0:c1], tw.sb[c0:c1]
        LA = 2.0 * (A * Mm) * w2 + PP[:, c0:c1] * eb
        LP = (A * A) * w2 + PP[:, c0:c1] * (sb / 3.0) + (2.0 / 3.0) * MP[:, c0:c1]
        if hub is not None:
            G23 = hub[0][:, c0:c1] * (2.0 / 3.0)
            LA = LA + G23 * Mm
            LP = LP + G23 * A
        return A, Mm, LA, LP

    if _SHARED_ON and _shared_rank(n) and (tw.k - tw.ko) >= _SHARED_AGE:
        cj = tw.ko * n
        _A, _M, LAj, LPj = legs(cj, cj + n)
        tw._wts = ((LAj * LAj).sum(axis=0), (LPj * LPj).sum(axis=0))
    c0 = (k - m) * n
    A, Mm, LA, LP = legs(c0, k * n)
    acc = _mm(fnp.concatenate([LA, LP], axis=1), fnp.concatenate([A, Mm], axis=1).T)
    if hub is not None:
        acc = acc + _mm(hub[1][:, 2 * (k - m):] * (2.0 / 3.0), tw.Y[:, 2 * (k - m):].T)
    return (acc + _mm(PPL[:, (k - m) * r:], tw.Z[:, (k - m) * r:].T) * (1.0 / 3.0)) * off_mask

'''
arms = {
    "age3_r128": [("_SHARED_AGE = 5\n", "_SHARED_AGE = 3\n"), ("_SHARED_RANK = 256\n", "_SHARED_RANK = 128\n")],
    "age3_r256": [("_SHARED_AGE = 5\n", "_SHARED_AGE = 3\n")],
    "age3_r64p3": [("_SHARED_AGE = 5\n", "_SHARED_AGE = 3\n"), ("_SHARED_RANK = 256\n", "_SHARED_RANK = 64\n"),
                   ("_SHARED_PASSES = 1\n", "_SHARED_PASSES = 3\n")],
}
for m in (1, 2, 3, 4):
    arms[f"d21age{m}"] = [(D21_ANCHOR, D21_NEW),
                          ("\ndef _dslice21(tw, off_mask):", YOUNG.format(m=m) + "\ndef _dslice21(tw, off_mask):")]
for name, subs in arms.items():
    s = src
    for a, b in subs:
        s = sub(s, a, b)
    p = Path(__file__).parent / f"arm_{name}.py"
    p.write_text(s, encoding="utf-8", newline="\n")
    print(name, hashlib.sha256(p.read_bytes()).hexdigest()[:8])
