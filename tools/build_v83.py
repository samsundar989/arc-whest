"""Build v83 (HANDOFF 2026-10-11 Lane P6, P6.1 step 4) = v82c + fold + kink-active H by count.

    uv run python tools/build_v83.py --params <refit params_best.npz> --variant fold+Habs --out estimator_k3_v83.py

Pipeline (each stage is an existing, reproducible builder; the first two rebuild v82c_r1 byte for byte
from R1's params, sha 66f3ee4a):
  1. `build_star.py --all` on v80_hyg with the refit's leaves (corrector, gains, S2t tables, sg);
  2. `build_v82.py --items 0,1,4,5,7`;
  3. the strategist's exact text patches from `docs/research/2026-10-11_p6price/price.py` for
     `--variant` (fold, habs(T), hasym(a,b,T), J(k), ...), so the code is exactly what was metered;
  4. bake `_FOLD_GAIN` from the refit's `gain__fg` (absent -> stays None = use _STAR_GAIN, unrefit).
`--ws` sets `_STRASSEN_MIN_BLOCK = 40` (the B twin).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs/research/2026-10-11_p6price"))
import price  # noqa: E402


def patches(variant: str):
    """A price.py VARIANTS key, or '+'-joined atoms: fold, od<L>, J<k>, age<k>, jsvd."""
    if variant in price.VARIANTS:
        return price.VARIANTS[variant]
    out = []
    for at in variant.split("+"):
        if at == "fold":
            out.append(price.fold)
        elif at == "jsvd":
            out.append(price.join_svd)
        elif at.startswith("od"):
            out.append(price.optdepth(int(at[2:])))
        elif at.startswith("age"):
            out.append(price.age(int(at[3:])))
        elif at.startswith("J"):
            out.append(price.J(int(at[1:])))
        else:
            raise SystemExit(f"unknown atom {at!r} in {variant!r}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--variant", required=True, help="a price.py VARIANTS key, e.g. fold+Habs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ws", action="store_true")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as td:
        star, v82 = Path(td) / "star.py", Path(td) / "v82.py"
        subprocess.run([sys.executable, str(ROOT / "tools/build_star.py"), "--all",
                        "--base", str(ROOT / "estimator_k3_v80_hyg.py"), "--params", a.params,
                        "--out", str(star)], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
        subprocess.run([sys.executable, str(ROOT / "tools/build_v82.py"), "--items", "0,1,4,5,7",
                        "--base", str(star), "--out", str(v82)], check=True, cwd=ROOT,
                       stdout=subprocess.DEVNULL)
        src = v82.read_text(encoding="utf-8")
    for p in patches(a.variant):
        src = p(src)
    z = np.load(a.params)
    if "gain__fg" in z.files:
        src = price.rep(src, "_FOLD_GAIN = None\n", f"_FOLD_GAIN = {[float(x) for x in z['gain__fg']]!r}\n")
    if a.ws:
        src = price.rep(src, "_STRASSEN_MIN_BLOCK = 20\n", "_STRASSEN_MIN_BLOCK = 40\n")
    Path(a.out).write_text(src, encoding="utf-8", newline="\n")
    print(f"wrote {a.out}: {a.variant}, fg {'baked' if 'gain__fg' in z.files else 'unrefit (_STAR_GAIN)'}"
          f"{', ws' if a.ws else ''}")


if __name__ == "__main__":
    main()
