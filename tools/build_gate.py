"""Bounded, identity-initialised, STATE-DEPENDENT gates on the third-order factors.

    python tools/build_gate.py --arm birth                     # v68_wino -> v70g_birth (identity)
    python tools/build_gate.py --arm trans                     # v68_wino -> v70g_trans (identity)
    python tools/build_gate.py --arm birth --base estimator_k3_v62_s1.py \
        --out experiments/2026-09-27_gates/estimator_k3_v62_gb.py      # the TWIN candidate
    python tools/build_gate.py --arm birth --base <build_s2 output> --params <twin npz> --out X
    python tools/build_gate.py --selftest                      # build all four, check anchors

Design: `experiments/2026-09-27_gates/DESIGN.md`. Two arms, never together:

  birth  the two-leg all-distinct birth block. Its end vector `w1` (in `fac1 = w1 C11`) is
         replaced by `w1 * g_e` and its middle vector `w2` (`w2b`) by `w2 * g_m`. Because
         `fac1` is also the carrier of the separable leg (`3 A_L d(eb)`) and of the kappa4
         hubs' second leg, the SAME gated `w1e` is used in `S_sep` and in the hub
         compensation -- otherwise the repeated slices would be double counted.
  trans  the inherited Wick transport. `_wick_tower(w1[:,None], adv)` becomes
         `_wick_tower((w1 * g_t)[:,None], adv)`, and the repeated-slice compensation
         `p111_3 = w1^3 D3`, `p111_21 = w1_i^2 w1_j D21` is recomputed with `w1 * g_t`.

Each gate is  g = 1 + S * tanh( h @ W2[l] + b2[l] ),  h = tanh( x @ W1[l] + b1[l] ),
x = clip((alpha, D3/sigma^3, D4/sigma^4) standardised, +-6), per-layer tables. W2 = b2 = 0
makes g == 1.0 EXACTLY, so an identity build reproduces the base bit for bit (the gated
vectors are `w1 * 1.0`). |g - 1| < S always (bounded). Only O(n) vectors are gated; the
factor matrices are the ones the base already builds, so the factor count and the matmul
count are unchanged. Cost: ~20 elementwise/thin ops and ~1e5 FLOPs per non-final layer.

The shipped tables are Python lists in `_GATE_TAB`, turned into arrays once per `predict`
in `_constants` (like `_corr_arrays`), under `const["gate"]`. The twin (`tools/twin/twin.py`
stages G1b / G1t / G1bg / G1tg) trains `gate/W1, b1, W2, b2` by replacing that dict --
no source hook is needed. Imports: stdlib only in the generated code path.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "experiments" / "2026-09-27_gates"

#: global standardisers of (alpha, gamma3, gamma4): the closure-fit statistics over FIT
#: exact labels (experiments/2026-09-29_closure/clos_init.npz ufmu/ufisd, first three).
FMU = (1.5061462e-03, 2.6751632e-05, 3.9658159e-02)
FISD = (0.44968137, 15.288525, 55.01315)

ARMS = {"birth": 2, "trans": 1}              # arm -> number of gate outputs
DEFAULT_S = {"birth": 0.5, "trans": 0.25}

_CODE = '''

# ==========================================================================
# STATE-DEPENDENT FACTOR GATES (tools/build_gate.py; experiments/2026-09-27_gates)
# ==========================================================================
# g = 1 + _GATE_S * tanh(tanh(x W1[l] + b1[l]) W2[l] + b2[l]), x = standardised
# (alpha, D3/sigma^3, D4/sigma^4). W2 = b2 = 0 is EXACTLY the identity (g == 1.0).
#   "birth": fac1 = (w1 g_e) C11 and w2b = w2 g_m  (outputs 0, 1)
#   "trans": the Wick transport of the inherited tower uses w1 g_t  (output 0)
# In both arms the repeated-slice compensation p111_3 / p111_21 (and, for "birth", S_sep and
# the hub compensation, which read fac1's w1) is recomputed from the SAME gated vectors.
_GATE_ON = True
_GATE_ARM = {arm!r}
_GATE_S = {scale!r}
_GATE_XCLIP = 6.0
_GATE_TAB = {tab}


def _gate_arrays(dtype):
    """The gate tables as arrays, rebuilt every `predict` (as `_corr_arrays`)."""
    if not _GATE_ON or not _GATE_TAB:
        return None
    return {{k: fnp.array(v, dtype=dtype) for k, v in _GATE_TAB.items()}}


def _gate_eval(g, li, alpha, sigma, D3, D4, dtype):
    """(n, O) per-neuron gates at layer `li`, or None (identity) past the table."""
    if g is None or li >= g["W1"].shape[0]:
        return None
    inv = fnp.divide(dtype(1.0), sigma)
    inv2 = inv * inv
    X = fnp.stack([alpha, D3 * (inv2 * inv), D4 * (inv2 * inv2)], axis=1)
    X = fnp.clip((X - g["fmu"][li]) * g["fisd"][li], dtype(-_GATE_XCLIP), dtype(_GATE_XCLIP))
    h = fnp.tanh(X @ g["W1"][li] + g["b1"][li])
    return fnp.tanh(h @ g["W2"][li] + g["b2"][li]) * _GATE_S + 1.0

'''

# (anchor, replacement). Each anchor must occur exactly once in the base.
PATCHES = [
    # 1. the birth block: gate evaluation, gated vectors, compensation from the gated vectors
    ("""        w1, w2 = w[(1, 1)], w[(2, 1)]
        fac1 = w1[:, None] * C11
""",
     """        w1, w2 = w[(1, 1)], w[(2, 1)]
        # GATES (tools/build_gate.py). w1e/w2m: the birth block's end/middle vectors;
        # w1t: the inherited tower's Wick scale. Each is exactly w1/w2/w1 at identity.
        w1e = w2m = w1t = None
        _gg = _gate_eval(const.get("gate"), layer_index, alpha, sigma, D3, D4, dtype)
        if _gg is not None and _GATE_ARM == "birth":
            w1e = w1 * _gg[:, 0]
            w2m = w2 * _gg[:, 1]
        elif _gg is not None and _GATE_ARM == "trans":
            w1t = w1 * _gg[:, 0]
        w1e = w1 if w1e is None else w1e
        w2m = w2 if w2m is None else w2m
        w1t = w1 if w1t is None else w1t
        fac1 = w1e[:, None] * C11
"""),
    ("""        w2b = w2[None, :]
        w2b_vec = w2
        w1_sq = w1 * w1
        p111_3 = w1_sq * w1 * D3
        p111_21 = ((w1_sq[:, None] * w1[None, :]) * D21
                   + ((fac1 * fac1) * w2b) * off_mask)
""",
     """        w2b = w2m[None, :]
        w2b_vec = w2m
        w1_sq = w1t * w1t
        p111_3 = w1_sq * w1t * D3
        p111_21 = ((w1_sq[:, None] * w1t[None, :]) * D21
                   + ((fac1 * fac1) * w2b) * off_mask)
"""),
    # 2. the inherited tower's Wick scale
    ("        adv = _wick_tower(w[(1, 1)][:, None], adv)\n",
     "        adv = _wick_tower(w1t[:, None], adv)\n"),
    # 3. the separable leg rides on fac1 = w1e C11, so it must read the gated w1e
    ("        S_sep = (e_b[:, None] * C11) * w1[None, :]\n",
     "        S_sep = (e_b[:, None] * C11) * w1e[None, :]\n"),
    # 4. the hubs' second leg is A_L = fac1: their repeated content reads w1e too
    ("""                resid = (resid - ((yA * w1)[:, None] * C11) * (cA * _t)[None, :]
                         - ((yB * w1)[:, None] * C11) * (cB * _t)[None, :])
""",
     """                resid = (resid - ((yA * w1e)[:, None] * C11) * (cA * _t)[None, :]
                         - ((yB * w1e)[:, None] * C11) * (cB * _t)[None, :])
"""),
    # 5. the tables, once per predict
    ('                "corr": _corr_arrays(dtype)}\n',
     '                "corr": _corr_arrays(dtype),\n'
     '                "gate": _gate_arrays(dtype)}\n'),
]
CODE_ANCHOR = "\ndef _layer(W, state, table, dtype, const, init_scale, last_layer, gens,"
#: the base must not run a branch that reads w1/w1_sq/fac1 in a way this patch does not cover
REQUIRED = ["_P2_SEPARABLE = True\n", "_P3_ON = False\n", "_PROTECT_FEED = False\n",
            "_HUB_COMPENSATE = True\n"]


def identity_tables(arm, hidden=8, seed=20260927, depth=16):
    """W1 ~ N(0, 1/F) and b1 ~ U(-0.5, 0.5) (deterministic); W2 = b2 = 0 (the identity)."""
    F, O = 3, ARMS[arm]
    rng = random.Random(seed)
    W1 = [[[rng.gauss(0.0, 1.0 / math.sqrt(F)) for _ in range(hidden)] for _ in range(F)]
          for _ in range(depth)]
    b1 = [[rng.uniform(-0.5, 0.5) for _ in range(hidden)] for _ in range(depth)]
    W2 = [[[0.0] * O for _ in range(hidden)] for _ in range(depth)]
    b2 = [[0.0] * O for _ in range(depth)]
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2,
            "fmu": [list(FMU) for _ in range(depth)], "fisd": [list(FISD) for _ in range(depth)]}


def tables_from_params(arm, path, hidden=8, seed=20260927, depth=16):
    """Twin npz (gate__W1 ...) over the identity tables: missing keys keep their init."""
    import numpy as np
    tab = identity_tables(arm, hidden, seed, depth)
    p = dict(np.load(path))
    for k in list(tab):
        key = f"gate__{k}"
        if key in p:
            v = p[key].astype(np.float64)
            if list(v.shape) != list(np.asarray(tab[k]).shape):
                raise SystemExit(f"{key}: shape {v.shape} != {np.asarray(tab[k]).shape}")
            tab[k] = v.tolist()
    return tab


def build(base: Path, arm: str, out: Path, scale=None, hidden=8, seed=20260927, params=None):
    src = base.read_text(encoding="utf-8")
    for r in REQUIRED:
        if r not in src:
            raise SystemExit(f"{base.name}: needs `{r.strip()}` (the patch covers only that path)")
    if "_GATE_TAB" in src:
        raise SystemExit(f"{base.name} already carries gates")
    for a, b in PATCHES:
        if src.count(a) != 1:
            raise SystemExit(f"{base.name}: anchor occurs {src.count(a)}x:\n{a}")
        src = src.replace(a, b, 1)
    tab = (tables_from_params(arm, params, hidden, seed) if params
           else identity_tables(arm, hidden, seed))
    code = _CODE.format(arm=arm, scale=float(DEFAULT_S[arm] if scale is None else scale),
                        tab=repr(tab))
    if src.count(CODE_ANCHOR) != 1:
        raise SystemExit("no _layer anchor")
    src = src.replace(CODE_ANCHOR, code + CODE_ANCHOR, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(src, encoding="utf-8", newline="\n")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="estimator_k3_v68_wino.py")
    ap.add_argument("--arm", choices=sorted(ARMS), default="birth")
    ap.add_argument("--out", default="")
    ap.add_argument("--scale", type=float, default=None, help="the bound S (default per arm)")
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--params", default="", help="twin npz: bake trained gate__* tables")
    ap.add_argument("--selftest", action="store_true",
                    help="build the v68 and v62 identity files for both arms")
    a = ap.parse_args(argv)
    if a.selftest:
        for arm, tag in (("birth", "b"), ("trans", "t")):
            for base, out in ((ROOT / "estimator_k3_v68_wino.py",
                               OUTDIR / f"estimator_k3_v70g_{arm}.py"),
                              (ROOT / "estimator_k3_v62_s1.py",
                               OUTDIR / f"estimator_k3_v62_g{tag}.py")):
                build(base, arm, out, hidden=a.hidden, seed=a.seed)
                print(f"wrote {out.relative_to(ROOT)}")
        return 0
    base = Path(a.base)
    if not base.is_absolute():
        base = ROOT / base
    out = Path(a.out) if a.out else OUTDIR / f"estimator_k3_v70g_{a.arm}.py"
    if not out.is_absolute():
        out = ROOT / out
    build(base, a.arm, out, a.scale, a.hidden, a.seed, a.params or None)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
