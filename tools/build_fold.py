"""Step 4(b) arms: birth-layer-aware fold age for the shared old-tier basis.

    python tools/build_fold.py --base estimator_k3_v65_s2t.py --out-dir experiments/<dir>/arms \
        --arms "A3U3,A4U3,A3U5"

PLAN rev 13 §3 Step 4(b): "lineages born at layers <= 3 folded at age 3 (they are
contracted x0.6-0.88 per layer)". Today every lineage stays exact until it is
`_SHARED_AGE` = 5 layers old, then folds into the rank-256 shared basis. Lineage j is
born at layer j (one birth per layer from layer 0), and the fold always takes the
OLDEST recent lineage, whose birth layer is therefore `tw.ko`. So the arm replaces
the four `(tw.k - tw.ko) >=/< _SHARED_AGE` tests with `_fold_age(tw)`:

    age = _FOLD_EARLY_AGE if (_FOLD_EARLY_AGE and tw.ko <= _FOLD_EARLY_UPTO) else _SHARED_AGE

A deterministic function of the layer index only -- NOT the per-lineage adaptive age
PLAN §5 closed (that read lineage energies). With `_FOLD_EARLY_AGE = 0` the predicate
is `_SHARED_AGE` exactly and the file is the base. Arm names: A<age>U<upto>.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_PRED = '''

#: Step 4(b) (tools/build_fold.py): lineages born at layer <= _FOLD_EARLY_UPTO fold
#: into the shared basis at age _FOLD_EARLY_AGE instead of _SHARED_AGE. 0 = off.
_FOLD_EARLY_AGE = {age}
_FOLD_EARLY_UPTO = {upto}


def _fold_age(tw):
    """The fold age that applies to the OLDEST recent lineage (born at layer tw.ko)."""
    if _FOLD_EARLY_AGE and tw.ko <= _FOLD_EARLY_UPTO:
        return _FOLD_EARLY_AGE
    return _SHARED_AGE
'''


def build(src: str, age: int, upto: int) -> str:
    n_ge = src.count("(tw.k - tw.ko) >= _SHARED_AGE")
    n_lt = src.count("(tw.k - tw.ko) < _SHARED_AGE")
    if (n_ge, n_lt) != (2, 2):
        raise SystemExit(f"expected 2+2 fold tests, found {n_ge}+{n_lt}")
    src = src.replace("(tw.k - tw.ko) >= _SHARED_AGE", "(tw.k - tw.ko) >= _fold_age(tw)")
    src = src.replace("(tw.k - tw.ko) < _SHARED_AGE", "(tw.k - tw.ko) < _fold_age(tw)")
    anchor = "\ndef _shared_rank(n):\n"
    if src.count(anchor) != 1:
        raise SystemExit("no _shared_rank anchor")
    return src.replace(anchor, _PRED.format(age=age, upto=upto) + anchor, 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="estimator_k3_v65_s2t.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arms", default="A3U3,A4U3,A3U5")
    a = ap.parse_args(argv)
    src = (ROOT / a.base).read_text(encoding="utf-8")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.base).stem
    for arm in ["off"] + [x for x in a.arms.split(",") if x]:
        if arm == "off":
            age, upto = 0, 0
        else:
            m = re.fullmatch(r"A(\d+)U(\d+)", arm)
            if not m:
                raise SystemExit(f"bad arm {arm!r}")
            age, upto = int(m.group(1)), int(m.group(2))
        p = out / f"{stem}__fold_{arm}.py"
        p.write_text(build(src, age, upto), encoding="utf-8", newline="\n")
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
