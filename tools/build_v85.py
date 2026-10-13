"""Build v85+ (HANDOFF 2026-10-12 Lane P7, P7.0c) = v84's builder + lane S / K / O atoms.

    uv run python tools/build_v85.py --params <params_best.npz> --atoms sb2 --out estimator_k3_v85.py
    uv run python tools/build_v85.py --params experiments/2026-10-11_p6/r8/age6/params_best.npz --atoms "" --check-v84

Pipeline:
  1. `build_v83.py --variant <--base>` (default `fold+age6` = v84_age6; R8 params rebuild it byte for byte,
     sha c1b098c0 -- `--check-v84` asserts that);
  2. the '+'-joined `--atoms`, applied in the order given, each from `tools/atoms/<name>.py`.

Atom slots (owner lane, handoff step):
  S: sb2 (S1 variant B, bit-identical) · sbtrim (op trim, bit-identical; after sb2, before depth) · depth16 / depth8 (S2)
     · sbks (S3 variant C). Order: sb2+sbtrim+depthX; k atoms anywhere after sb2.
  K: k0 (gain-fill dropped D21 rows) · k1 (kink-only D21 rows) · k2 (leg output rows) · k3 (two-column H) · k4 (last-layer
     transport) · k5 (adaptive counts, optional)
  O: o1 (SVD join / smaller rank) · o2 (split leg ranks) · o3 (ages / no old tier) · o4 (spend items)
An arm of an atom is `name:arg` (e.g. `k1:3.0`, `o1:224`).

Atom contract (`tools/atoms/README.md`): a module `tools/atoms/<name>.py` defining
  apply(src: str, z) -> str          for `name`
  make(arg: str) -> (src, z) -> str  for `name:arg`
where `z` is the loaded params npz (atoms that bake learned constants read their keys from it). Patches are exact
text replacements with every anchor occurring once (`tools.atoms.rep`).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

V84_SHA = "c1b098c0"


def atom(spec: str):
    name, _, arg = spec.partition(":")
    try:
        mod = importlib.import_module(f"tools.atoms.{name}")
    except ModuleNotFoundError as e:
        if e.name == f"tools.atoms.{name}":
            raise SystemExit(f"atom {name!r} not delivered yet (expected tools/atoms/{name}.py)")
        raise
    if arg:
        return mod.make(arg)
    return mod.apply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--base", default="fold+age6", help="build_v83 --variant for the starting point (v84)")
    ap.add_argument("--atoms", default="", help="'+'-joined lane atoms, e.g. sb2+k0+k1:3.0")
    ap.add_argument("--out", default="")
    ap.add_argument("--ws", action="store_true", help="B twin (_STRASSEN_MIN_BLOCK = 40)")
    ap.add_argument("--check-v84", action="store_true", help="assert the no-atom build is v84_age6 byte for byte")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "base.py"
        cmd = [sys.executable, str(ROOT / "tools/build_v83.py"), "--params", a.params, "--variant", a.base,
               "--out", str(base)] + (["--ws"] if a.ws else [])
        subprocess.run(cmd, check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
        src = base.read_text(encoding="utf-8")
    if a.check_v84:
        sha = hashlib.sha256(src.encode("utf-8")).hexdigest()[:8]
        if sha != V84_SHA:
            raise SystemExit(f"base build sha {sha} != v84_age6 {V84_SHA}")
        print(f"base == v84_age6 ({V84_SHA})")
    z = np.load(a.params)
    specs = [s for s in a.atoms.split("+") if s]
    for s in specs:
        src = atom(s)(src, z)
    if a.out:
        Path(a.out).write_text(src, encoding="utf-8", newline="\n")
        sha = hashlib.sha256(src.encode("utf-8")).hexdigest()[:8]
        print(f"wrote {a.out} ({sha}): {a.base} + [{' + '.join(specs) or 'no atoms'}]{', ws' if a.ws else ''}")


if __name__ == "__main__":
    main()
