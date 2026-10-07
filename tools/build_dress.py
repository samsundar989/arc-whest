"""Build a dressed-leg (P2.1) estimator from a v77-family file, with twin-refit parameters baked in.

    uv run python tools/build_dress.py --base estimator_k3_v77_kash.py \
        --params experiments/2026-10-07_p2/refit_dress_all/params_best.npz \
        --out estimator_k3_v79_dress.py
    uv run python tools/build_dress.py --selftest     # G1bg params, dressing off == v77 bytes

HANDOFF 2026-10-07 Lane P2, P2.1 (R1 §5.1). AUGMENT's heavy-edge kappa3 birth families fold
into the newborn's light leg (no new matmul):

    X~[a, j] = w1e_a C_aj + H[a, j],
    H[a, j]  = 1/2 w2_a D21[a, j]
             + (w3_j / w2_j) (1/2 w1_a D21[j, a] + 1/4 w2_a D22[a, j] + 1/2 w2_a C_aj^2),

w3/w2 = -alpha/sigma, all three blocks off-diagonal. The stored A_L (`fac1`) is the light
leg, so every repeated-slice term that reads it is recomputed from X~: p111_21's two-leg slice
(through fac1), S_sep = (X~ d(e))^T, and the hub compensation d(y) X~ d(c). The newborn's
repeated slices therefore stay at the table's values; only its distinct-index content changes.
This is exactly the twin's `dress` arm (tools/twin/twin.py), at the same source sites.

The refit (twin stage S2t: corrector pages + lambda/theta/feed/tt/t1/cu/cb gains) replaces
the base's baked tables and `_CORR_B64` block. The gate tables and `_Q31_COEF` must be
unchanged (frozen in the refit) and are checked, not rewritten.
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_twin  # noqa: E402

TABLES = (("_LAM_GAIN", "gain__lam"), ("_THETA_GAIN", "gain__theta"),
          ("_FEED_GAIN", "gain__feed"), ("_TT_GAIN", "gain__tt"),
          ("_T1_GAIN", "gain__t1"), ("_CU_GAIN", "gain__cu"),
          ("_CB_GAIN", "gain__cb"))
FROZEN = (("_Q31_COEF", "gain__q31"),)

DRESS_CONST = '''
# ---- P2.1 DRESSED LEGS (tools/build_dress.py; R1 §5.1) ---------------------
# AUGMENT's heavy-edge kappa3 birth families folded into the newborn's light leg.
# Family scales: 1.0 = on, 0.0 = off (an exact identity when all three are 0).
_DRESS_C2 = {c2!r}
_DRESS_D21 = {d21!r}
_DRESS_D22 = {d22!r}


def _dress_H(w, alpha, sigma, C11, D21, D22):
    """H[a, j] of the dressed leg X~ = w1e C + H. Elementwise O(n^2); no matmul.

    r_j = w3_j / w2_j = -alpha_j / sigma_j (finite for decided columns). D21[a, j] is
    kappa3_aaj of the current pre-activation (its dead rows are zero under KAS D), D22 the
    chain's (2,2) slice; all three blocks carry a zero diagonal, so H does too.
    """
    w1, w2 = w[(1, 1)], w[(2, 1)]
    r = -(alpha / sigma)
    H = None
    if _DRESS_D21 != 0.0:
        H = (w2[:, None] * D21 + (w1[:, None] * D21.T) * r[None, :]) * (0.5 * _DRESS_D21)
    if _DRESS_D22 != 0.0:
        t = ((w2[:, None] * D22) * r[None, :]) * (0.25 * _DRESS_D22)
        H = t if H is None else H + t
    if _DRESS_C2 != 0.0:
        t = ((w2[:, None] * (C11 * C11)) * r[None, :]) * (0.5 * _DRESS_C2)
        H = t if H is None else H + t
    return H

'''

PATCHES = [
    ("        fac1 = w1e[:, None] * C11\n",
     "        fac1 = w1e[:, None] * C11\n"
     "        _Hd = _dress_H(w, alpha, sigma, C11, D21, D22)\n"
     "        if _Hd is not None:\n"
     "            # P2.1: the dressed leg X~; every repeated-slice term below reads it\n"
     "            fac1 = fac1 + _Hd\n"
     "        _Xd = fac1\n"),
    ("        S_sep = (e_b[:, None] * C11) * w1e[None, :]\n",
     "        S_sep = e_b[:, None] * _Xd.T\n"),
    ("                resid = (resid - ((yA * w1e)[:, None] * C11) * (cA * _t)[None, :]\n"
     "                         - ((yB * w1e)[:, None] * C11) * (cB * _t)[None, :])\n",
     "                resid = (resid - (yA[:, None] * _Xd) * (cA * _t)[None, :]\n"
     "                         - (yB[:, None] * _Xd) * (cB * _t)[None, :])\n"),
]


def _table(src, name):
    m = re.search("^" + name + r" = (.*)$", src, re.M)
    if not m:
        raise SystemExit(f"no {name} table")
    return m, np.array(ast.literal_eval(m.group(1)), dtype=float)


def build(base: Path, params: Path, dress=(1.0, 1.0, 1.0)) -> str:
    src = build_twin.build(base, _corr_only(params))
    p = dict(np.load(params))
    for name, key in TABLES:
        m, old = _table(src, name)
        new = p[key].astype(float)
        if new.shape != old.shape:
            raise SystemExit(f"{name}: shape {new.shape} != base {old.shape}")
        src = src[:m.start()] + f"{name} = {new.tolist()!r}" + src[m.end():]
    for name, key in FROZEN:
        _, old = _table(src, name)
        if key in p and np.abs(p[key].astype(float) - old).max() != 0.0:
            raise SystemExit(f"{name} differs from the params' {key}: not a frozen refit")
    m = re.search(r"^_GATE_TAB = (.*)$", src, re.M)
    gt = ast.literal_eval(m.group(1))
    for k in ("W1", "b1", "W2", "b2"):
        if f"gate__{k}" in p and np.abs(np.array(gt[k]) - p[f"gate__{k}"]).max() > 0.0:
            raise SystemExit(f"gate {k} differs from the params: not a frozen refit")
    if any(dress):
        c2, d21, d22 = (float(x) for x in dress)
        anchor = "def _layer(W, state, table, dtype, const, init_scale, last_layer, gens,\n"
        if src.count(anchor) != 1:
            raise SystemExit("no _layer anchor")
        src = src.replace(anchor, DRESS_CONST.format(c2=c2, d21=d21, d22=d22) + "\n" + anchor, 1)
        for a, b in PATCHES:
            if src.count(a) != 1:
                raise SystemExit(f"anchor occurs {src.count(a)}x:\n{a}")
            src = src.replace(a, b, 1)
    return src


def _corr_only(params: Path) -> Path:
    """build_twin takes a corr-only npz with bare names."""
    import tempfile
    p = dict(np.load(params))
    td = Path(tempfile.mkdtemp())
    out = td / "corr.npz"
    np.savez(out, **{k.split("__")[1]: v for k, v in p.items() if k.startswith("corr__")})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default=str(ROOT / "estimator_k3_v77_kash.py"))
    ap.add_argument("--params", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--dress", default="1,1,1", help="family scales c2,d21,d22 ('0,0,0' = off)")
    ap.add_argument("--selftest", action="store_true",
                    help="G1bg params with dressing off must rebuild the base byte-exact")
    a = ap.parse_args(argv)
    base = Path(a.base)
    if a.selftest:
        g1 = ROOT / "experiments" / "2026-09-27_gates" / "G1bg" / "params_best.npz"
        same = build(base, g1, (0.0, 0.0, 0.0)) == base.read_text(encoding="utf-8")
        print("selftest: G1bg + no dressing rebuilds the base byte-exact:", same)
        return 0 if same else 1
    dress = tuple(float(x) for x in a.dress.split(","))
    out = build(base, Path(a.params), dress)
    Path(a.out).write_text(out, encoding="utf-8", newline="\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
