"""Bake twin S2 parameters (corrector pages + per-layer lambda/theta/feed gains) into a file.

    python tools/build_s2.py --base estimator_k3_v62_s1.py \
        --params experiments/2026-09-25_twin/S2/params_best.npz --out estimator_k3_v64_s2.py

HANDOFF 2026-09-25 Step 2 stage S2. The twin trains three per-layer multiplicative gains
through the chain (`tools/twin/twin.py` TWIN_PATCHES):

    lambda  _own_lambda(i, depth)   x _LAM_GAIN[i]      (indexed by the SOURCE layer i)
    theta   _own_extra's th         x _THETA_GAIN[l]    (the current layer)
    feed    the hub feed scale _sf  x _FEED_GAIN[l]

This writes them as Python float tables with the same three multiplications, all on
Python scalars -- no flopscope operation, so FLOPs are unchanged and op calls are
unchanged wherever the product was already != 1.0 (verified by the cost run). Past the
table's length each gain is 1.0 (shape-generic, hard rule 7). The corrector pages go
through `tools/build_twin.py` unchanged.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

PATCHES = [
    # lambda: the multiplier is applied to the function's return value
    ("def _own_lambda(i, depth):\n",
     "def _own_lambda(i, depth):\n"
     "    return _own_lambda_base(i, depth) * (_LAM_GAIN[i] if 0 <= i < len(_LAM_GAIN) else 1.0)\n"
     "\n\n"
     "def _own_lambda_base(i, depth):\n"),
    ('def _own_extra(x):\n    th = _OWN_THETA_FAM * _prof("theta")\n',
     'def _own_extra(x):\n    th = _OWN_THETA_FAM * _prof("theta")\n'
     '    if 0 <= _CUR_LAYER < len(_THETA_GAIN):\n'
     '        th = th * _THETA_GAIN[_CUR_LAYER]\n'),
    ('            _sf = _OWN_S_FEED * _prof_at("feed", layer_index, depth)\n',
     '            _sf = _OWN_S_FEED * _prof_at("feed", layer_index, depth)\n'
     '            if layer_index < len(_FEED_GAIN):\n'
     '                _sf = _sf * _FEED_GAIN[layer_index]\n'),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="estimator_k3_v62_s1.py")
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    p = dict(np.load(a.params))
    corr = {k.split("__")[1]: v for k, v in p.items() if k.startswith("corr__")}
    with tempfile.TemporaryDirectory() as td:
        cp = Path(td) / "corr.npz"
        np.savez(cp, **corr)
        tmp = Path(td) / "with_corr.py"
        subprocess.run([sys.executable, str(ROOT / "tools" / "build_twin.py"), "--base", a.base,
                        "--params", str(cp), "--out", str(tmp)], check=True)
        src = tmp.read_text(encoding="utf-8")
    for anchor, repl in PATCHES:
        if src.count(anchor) != 1:
            raise SystemExit(f"anchor occurs {src.count(anchor)}x:\n{anchor}")
        src = src.replace(anchor, repl, 1)
    tables = "\n# Twin S2 per-layer gains (tools/build_s2.py). 1.0 is exactly the identity.\n"
    for name, key in (("_LAM_GAIN", "gain__lam"), ("_THETA_GAIN", "gain__theta"),
                      ("_FEED_GAIN", "gain__feed")):
        vals = [float(x) for x in p[key]] if key in p else []
        tables += f"{name} = {vals!r}\n"
    anchor = "\n_CUR_LAYER = 0\n"
    if src.count(anchor) != 1:
        raise SystemExit("no _CUR_LAYER anchor")
    src = src.replace(anchor, tables + anchor, 1)
    Path(a.out).write_text(src, encoding="utf-8", newline="\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
