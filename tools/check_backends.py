"""Do the numpy and cupy accuracy backends agree, and how closely?

    python tools/check_backends.py <candidate.py> [--n 2] [--panel dev]

PLAN §4 asks for "predictions match the numpy path to 1e-6 relative on 2 nets".
**They do not, and they cannot.** Both backends run the chain in float32; the
GPU reduces in a different order, and sixteen layers of a cumulant chain amplify
that into ~1e-3 relative on a per-net final-layer MSE. It is not TF32 — measured
on a 1024^3 f32 product against a float64 reference, cupy is *more* accurate
than the CPU BLAS (3.5e-7 against 7.2e-7) — it is float32 associativity, and no
setting turns it off without changing the candidate.

So the backends are not interchangeable, and this script exists to say by how
much rather than to pass or fail:

* **cupy is a SCREENING instrument.** Use it where only the ranking matters and
  the signal is far above 1e-3 — the D2 fit's 400 evaluations, sweeps, ablations.
* **numpy is the instrument of record.** Every number that enters `RESULTS.md`,
  every PLAN §5 gate and every submission decision comes from it.

The per-net error is signed and roughly independent across nets, so the 100-net
mean is far tighter than any single net; the summary prints both, because the
mean is what a DEV score actually is.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"


def run(candidate: Path, panel: str, n: int, backend: str, out: Path) -> dict:
    env = dict(os.environ)
    env["WHEST_BACKEND"] = backend
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    if backend == "numpy":
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env[v] = "8"
    r = subprocess.run(
        [sys.executable, str(TOOLS / "_predict_worker.py"), str(candidate),
         panel, "0", str(n), str(out)],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise SystemExit(f"{backend} worker failed\n{r.stderr[-3000:]}")
    d = np.load(out)
    return {"preds": d["preds"], "truth": d["truth"],
            "stdout": r.stdout.strip().splitlines()[-1]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("candidate")
    ap.add_argument("--panel", default="dev")
    ap.add_argument("--n", type=int, default=2)
    a = ap.parse_args(argv)

    cand = Path(a.candidate)
    if not cand.is_absolute():
        cand = ROOT / cand

    with tempfile.TemporaryDirectory() as td:
        cpu = run(cand, a.panel, a.n, "numpy", Path(td) / "cpu.npz")
        gpu = run(cand, a.panel, a.n, "cupy", Path(td) / "gpu.npz")

    c, g, t = cpu["preds"], gpu["preds"], cpu["truth"].astype(np.float32)
    rel = np.abs(g - c) / np.maximum(np.abs(c), 1e-30)
    cm = ((c - t) ** 2).mean(axis=1)
    gm = ((g - t) ** 2).mean(axis=1)
    dmse = (gm - cm) / cm

    print(f"candidate  {cand.name}")
    print(f"panel      {a.panel.upper()}  n={a.n}")
    print(f"  numpy    {cpu['stdout']}")
    print(f"  cupy     {gpu['stdout']}")
    print()
    print("  per-element activation difference")
    print(f"    max absolute            {np.abs(g - c).max():.3e}")
    print(f"    max relative            {rel.max():.3e}")
    print(f"    mean relative           {rel.mean():.3e}")
    print("  per-net final-layer MSE")
    print(f"    max |relative| diff     {np.abs(dmse).max():.3e}")
    print(f"    mean signed diff        {dmse.mean():+.3e}")
    print("  panel mean MSE (what a DEV score is made of)")
    print(f"    numpy                   {cm.mean():.9e}")
    print(f"    cupy                    {gm.mean():.9e}")
    print(f"    relative                {(gm.mean() - cm.mean()) / cm.mean():+.3e}")
    print()
    print("  cupy is a SCREENING instrument. numpy is the instrument of record:")
    print("  every RESULTS.md row, PLAN §5 gate and submission decision uses it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
