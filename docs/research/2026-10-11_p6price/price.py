"""Price P6/M6 levers on today's chain with the exact flopscope 0.12.1 shape interpreter.

    python price.py                      # every variant, then a table + prices.json
    python price.py fold+Habs,fold+Hasym2030
    python price.py --write-only fold+Habs   # only write variants/est_<name>.py (for `twin.py eval --raw`)

The source is `estimator_k3_v82c_r1.py` at the repo root (override with PRICE_SRC=<path>).
Each variant is a chain of exact text patches; every anchor must occur once, or the run stops.
Each interpreter run is a fresh process (`run_census.py` installs fake flopscope/whestbench).
F includes the harness's output cast (+32,768 FLOPs, +1 op), so the base reproduces v82c_r1's
real meter: 363,022,283,745 FLOPs, F/B 0.16508, 22,375 ops.

Patches (all on v82c_r1 = v82b bundle + R1 refit):
  fold         M5.1 masked star-Q31 fold, recent lineages (`build_fold.py` logic with v82b's
               one-line `rec = _mm(..., _R.T)` anchor); g* = _STAR_GAIN (unrefit).
  habs(T)      H = kink-active columns by COUNT table T, ranked by alpha^2 (keep the T[l]
               smallest |alpha|). The estimator's `_kh` gather then drops A_L, w2b, hubs,
               e_b, the P2 residual rows and S3 on every dropped column.
  hasym(a,b,T) same, ranked by s = max(-alpha/a, alpha/b): dead side cut at -a, alive side at b.
  J(k)         `_JOIN_SKIP_TAIL = k` (no old-tier join at the last k non-final layers).
  age(k), rank(k), p2(k), passes(k)   `_SHARED_AGE`, `_SHARED_RANK`, `_P2_RANK`, `_SHARED_PASSES`.
  optdepth(L)  per-shape FLOP-optimal Strassen depth (leaf >= L, up to 7 levels).
  join_svd     old-tier join basis = exact top-r eigenspace of G = J D J^T + Q Sg Q^T via
               flopscope's top-k SVD (billed min(4mnk, economy); it computes the exact SVD).
  p2svd        P2 residual by exact top-r SVD instead of the one-power-iteration range finder.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = Path(os.environ.get("PRICE_SRC") or HERE.parents[2] / "estimator_k3_v82c_r1.py")
OUT = HERE / "variants"
B = 2 ** 41

KINK25 = [1024, 1024, 1024, 992, 928, 896, 832, 800, 768, 736, 704, 640, 608, 576, 576, 544]
KINK20 = [1024, 1024, 992, 928, 832, 800, 736, 704, 640, 608, 576, 544, 512, 480, 448, 448]
KINK30 = [1024, 1024, 1024, 1024, 992, 960, 928, 896, 864, 832, 768, 736, 704, 672, 640, 640]
# asymmetric counts from kink_tables.json (FIT q50, nearest 32 -- the convention of every KAS table):
#   keep_raw(a) - (keep_raw(b) - act_raw(b))  =  n - #(alpha < -a) - #(alpha > b)
ASYM_20_25 = [1024, 1024, 992, 960, 896, 832, 800, 736, 704, 672, 640, 608, 576, 544, 512, 480]
ASYM_20_30 = [1024, 1024, 992, 960, 928, 864, 832, 800, 736, 704, 672, 640, 608, 576, 544, 544]

H_OLD = '          "H": [1024, 1024, 1024, 992, 992, 960, 928, 928, 896, 864, 864, 832, 832, 800, 800, 800]}'
H_SEL_OLD = """        if _KH < n:
            _rkh = fnp.argsort(_perm)
            _mh = fnp.where(_rkh >= n - _KH, dtype(1.0), dtype(0.0))
            _kh = _perm[n - _KH:]
"""


def rep(s, a, b):
    c = s.count(a)
    if c != 1:
        raise SystemExit(f"anchor occurs {c}x: {a[:90]!r}")
    return s.replace(a, b, 1)


def J(k):
    return lambda s: rep(s, "_JOIN_SKIP_TAIL = 2\n", f"_JOIN_SKIP_TAIL = {k}\n")


def _h_select(score_expr):
    return ("        if _KH < n:\n"
            f"            _pa = fnp.argsort({score_expr})\n"
            "            _rka = fnp.argsort(_pa)\n"
            "            _mh = fnp.where(_rka < _KH, dtype(1.0), dtype(0.0))\n"
            "            _kh = _pa[:_KH]\n")


def habs(table=KINK25):
    def f(s):
        s = rep(s, H_OLD, f'          "H": {table!r}}}')
        return rep(s, H_SEL_OLD, _h_select("alpha * alpha"))
    return f


def hasym(a, b, table):
    def f(s):
        s = rep(s, H_OLD, f'          "H": {table!r}}}')
        return rep(s, H_SEL_OLD, _h_select(
            f"fnp.maximum(alpha * dtype({-1.0 / a!r}), alpha * dtype({1.0 / b!r}))"))
    return f


def fold(s):
    s = rep(s, "_STAR_GAIN = [",
            "#: M5.1 masked star-Q31 fold (recent lineages). None = use _STAR_GAIN (unrefit).\n"
            "_FOLD_GAIN = None\n_FOLD_TAU = 2.0\n_FOLD_V = None\n_STAR_GAIN = [")
    s = rep(s,
            "        D21 = (_dslice21(adv, off_mask) if adv is not None\n"
            "               else fnp.zeros((n, n), dtype=dtype))\n",
            "        global _FOLD_V\n"
            "        _FOLD_V = None\n"
            "        _FG = _FOLD_GAIN if _FOLD_GAIN is not None else _STAR_GAIN\n"
            "        if (isinstance(c4, list) and adv is not None and adv.k > adv.ko\n"
            "                and layer_index < len(_FG) and _FG[layer_index] != 0.0):\n"
            "            _rv = fnp.where(alpha * alpha <= dtype(_FOLD_TAU * _FOLD_TAU),\n"
            "                            (alpha / sigma) * dtype(-_FG[layer_index]), dtype(0.0))\n"
            "            _FOLD_V = (_rv, c4[adv.ko:])\n"
            "        D21 = (_dslice21(adv, off_mask) if adv is not None\n"
            "               else fnp.zeros((n, n), dtype=dtype))\n"
            "        _FOLD_V = None\n")
    return rep(s,
               "    rec = _mm(fnp.concatenate([LA[:, c:], LP[:, c:]], axis=1), _R.T)\n",
               "    LAr, LPr = LA[:, c:], LP[:, c:]\n"
               "    if _FOLD_V is not None:\n"
               "        _rv, _w3r = _FOLD_V\n"
               "        _w3c = _w3r[0] if len(_w3r) == 1 else fnp.concatenate(_w3r)\n"
               "        _Ar = tw.A_L if tw.ko else A_L\n"
               "        _Mr = tw.M if tw.ko else M\n"
               "        if kr is not None:\n"
               "            _rv = fnp.take(_rv, kr)\n"
               "        _A2 = (_Ar * _Ar) * _w3c\n"
               "        if kr is not None:\n"
               "            _A2k = fnp.take(_A2, kr, axis=0)\n"
               "            _Ak = fnp.take(_Ar, kr, axis=0)\n"
               "            _Mk = fnp.take(_Mr, kr, axis=0)\n"
               "        else:\n"
               "            _A2k, _Ak, _Mk = _A2, _Ar, _Mr\n"
               "        LAr = LAr + (_A2k * _Mk) * _rv[:, None]\n"
               "        LPr = LPr + (_A2k * _Ak) * (_rv * (1.0 / 3.0))[:, None]\n"
               "    rec = _mm(fnp.concatenate([LAr, LPr], axis=1), _R.T)\n")


def age(k):
    return lambda s: rep(s, "_SHARED_AGE = 5\n", f"_SHARED_AGE = {k}\n")


def rank(k):
    return lambda s: rep(s, "_SHARED_RANK = 256\n", f"_SHARED_RANK = {k}\n")


def p2(k):
    return lambda s: rep(s, "_P2_RANK = 4\n", f"_P2_RANK = {k}\n")


def passes(k):
    return lambda s: rep(s, "_SHARED_PASSES = 1\n", f"_SHARED_PASSES = {k}\n")


def optdepth(min_leaf=12, maxd=7):
    def f(s):
        s = rep(s, "def _depth_for(m, k, n):\n", "def _depth_for_old(m, k, n):\n")
        return rep(s, "def _mm(A, B):\n",
                   "def _sbc(b, m, k, n, d):\n"
                   "    if d <= 0 or (m % 2) or (k % 2) or (n % 2):\n"
                   "        return b * m * n * (2 * k - 1)\n"
                   "    mh, kh, nh = m // 2, k // 2, n // 2\n"
                   "    g = 4*b*mh*kh + 4*b*kh*nh + 7*b*mh*kh + 7*b*kh*nh + 7*b*mh*nh + b*m*n\n"
                   "    return g + _sbc(7 * b, mh, kh, nh, d - 1)\n\n\n"
                   "_DF_CACHE = {}\n\n\n"
                   "def _depth_for(m, k, n):\n"
                   "    key = (m, k, n)\n"
                   "    if key in _DF_CACHE:\n"
                   "        return _DF_CACHE[key]\n"
                   "    best = None\n"
                   f"    for d in range(0, {maxd} + 1):\n"
                   f"        if d and (min(m, k, n) >> d) < {min_leaf}:\n"
                   "            break\n"
                   "        c = _sbc(1, m, k, n, d)\n"
                   "        if best is None or c < best[0]:\n"
                   "            best = (c, d)\n"
                   "    _DF_CACHE[key] = best[1]\n"
                   "    return best[1]\n\n\n"
                   "def _mm(A, B):\n")
    return f


def join_svd(s):
    s = rep(s, "_SHARED_FORCE = False\n",
            "_SHARED_FORCE = False\n"
            "#: P6/M6: old-tier join basis = exact top-r eigenspace of J D J^T + Q Sg Q^T (top-k SVD).\n"
            "_JOIN_SVD = True\n")
    s = rep(s,
            "    if tw.ko:\n"
            "        Om = tw.Q\n"
            "    else:\n"
            "        X = J * fnp.sqrt(dJ)\n"
            "        Om = X[:, fnp.argsort((X * X).sum(axis=0))[-r:]]\n",
            "    if _JOIN_SVD:\n"
            "        Om = None\n"
            "    elif tw.ko:\n"
            "        Om = tw.Q\n"
            "    else:\n"
            "        X = J * fnp.sqrt(dJ)\n"
            "        Om = X[:, fnp.argsort((X * X).sum(axis=0))[-r:]]\n")
    return rep(s,
               "    else:\n"
               "        for _p in range(int(_SHARED_PASSES)):\n"
               "            Y = _mm(J, (_mm(J.T, Om)) * dJ[:, None])\n",
               "    elif _JOIN_SVD:\n"
               "        X = J * fnp.sqrt(dJ)[None, :]\n"
               "        if tw.ko:\n"
               "            _ev, _V = fnp.linalg.eigh(tw.Sg)\n"
               "            X = fnp.concatenate(\n"
               "                [X, tw.Q @ (_V * fnp.sqrt(fnp.maximum(_ev, 0.0))[None, :])], axis=1)\n"
               "        Qn, _s, _vt = fnp.linalg.svd(X, full_matrices=False, k=r)\n"
               "        FJ = _mm(Qn.T, J)\n"
               "        Sj = _mm(FJ * dJ, FJ.T)\n"
               "    else:\n"
               "        for _p in range(int(_SHARED_PASSES)):\n"
               "            Y = _mm(J, (_mm(J.T, Om)) * dJ[:, None])\n")


def p2svd(s):
    return rep(s,
               "    Om = R if r >= n else R[:, fnp.argsort((R * R).sum(axis=0))[-r:]]\n"
               "    Q, _ = fnp.linalg.qr(_mm(R, Om))\n",
               "    if r < n:\n"
               "        _U, _s, _Vt = fnp.linalg.svd(R, full_matrices=False, k=r)\n"
               "        return _Vt.T * (_s * 3.0)[None, :], _U\n"
               "    Om = R if r >= n else R[:, fnp.argsort((R * R).sum(axis=0))[-r:]]\n"
               "    Q, _ = fnp.linalg.qr(_mm(R, Om))\n")


FH = [fold, habs()]
VARIANTS = {
    "base": [],
    "fold": [fold],
    "Habs": [habs()],
    "fold+Habs": FH,
    "fold+Hasym2030": [fold, hasym(2.0, 3.0, ASYM_20_30)],
    "fold+Hasym2025": [fold, hasym(2.0, 2.5, ASYM_20_25)],
    "fold+Habs20": [fold, habs(KINK20)],
    "fold+Habs30": [fold, habs(KINK30)],
    "J3": [J(3)],
    "J4": [J(4)],
    "age6": [age(6)],
    "rank288": [rank(288)],
    "rank320": [rank(320)],
    "p2r8": [p2(8)],
    "fold+Habs+J3": FH + [J(3)],
    "fold+Habs+J4": FH + [J(4)],
    "fold+Habs+age6": FH + [age(6)],
    "fold+Habs+rank288": FH + [rank(288)],
    "fold+Habs+od12": FH + [optdepth(12)],
    "fold+Habs+od16": FH + [optdepth(16)],
    "fold+Habs+jsvd": FH + [join_svd],
    "fold+Habs+pass2": FH + [passes(2)],
    "fold+Habs+p2svd": FH + [p2svd],
    "fold+Habs+jsvd+rank224": FH + [join_svd, rank(224)],
    "fold+Habs+jsvd+rank192": FH + [join_svd, rank(192)],
    "fold+Habs+jsvd+age4": FH + [join_svd, age(4)],
}


def build(name):
    s = SRC.read_text(encoding="utf-8")
    for p in VARIANTS[name]:
        s = p(s)
    OUT.mkdir(exist_ok=True)
    fn = OUT / f"est_{name.replace('+', '_')}.py"
    fn.write_text(s, encoding="utf-8")
    return fn


def price(fn):
    out = subprocess.run([sys.executable, str(HERE / "run_census.py"), str(fn)],
                         capture_output=True, text=True, cwd=str(HERE))
    m = re.search(r"TOTAL (\d+) ops (\d+)", out.stdout)
    if not m:
        raise RuntimeError(out.stderr[-2000:])
    return int(m.group(1)) + 32768, int(m.group(2)) + 1


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    write_only = "--write-only" in sys.argv
    names = args[0].split(",") if args else list(VARIANTS)
    res = {}
    for nm in names:
        fn = build(nm)
        if write_only:
            print("wrote", fn)
            continue
        F, ops = price(fn)
        res[nm] = {"flops": F, "F/B": F / B, "ops": ops}
        print(f"{nm:20s} F {F:16,d}  F/B {F / B:.5f}  ops {ops:6,d}", flush=True)
    if res:
        path = HERE / "prices.json"
        old = json.loads(path.read_text()) if path.exists() else {}
        old.update(res)
        path.write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    main()
