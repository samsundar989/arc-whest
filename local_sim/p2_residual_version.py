"""flopscope 0.10 vs 0.12: is the residual gap the STACK, or the SESSION?

P2-P0 review, second Skeptic pass. `experiments/LOG.md`, 2026-08-24.

WHAT IS LEFT TO EXPLAIN
-----------------------
The P2-CP1 ladder (2026-08-21, whestbench 0.14 / flopscope 0.10, raw-NumPy weights)
recorded far larger residuals than today:

    arm     2026-08-21    today, SAME raw-NumPy weights    today, fnp weights
    k2p1     140.9 ms              70.1 ms                      50.4 ms
    k2p2     260.7 ms             110.9 ms                      92.0 ms

`local_sim/p2_weight_repr_residual.py` settled the last column: the weight
representation is worth a fixed **~19 ms** slab -- the sixteen 1024x1024
float32->float64 conversions that the meter could not see before P2-P0 and that
therefore landed in residual.

It does NOT explain the middle column. Holding the representation fixed, today is
still **2.01x** (k2p1) and **2.35x** (k2p2) faster than 2026-08-21. Two candidates
remain and they imply opposite protocols:

  * **the stack.** flopscope 0.12 may simply cost less wall time per traced op than
    0.10, or split `wall - backend - overhead` differently. Then the older numbers
    are the wrong instrument and today's are the design figures.
  * **the session.** Machine state -- thermal, cache, background load, whatever --
    differing by 2x between days. Then a **between-session term is real and
    mandatory**, and every headroom claim in this repository has to carry it.

This file discriminates by running the identical candidate under both flopscope
releases **in the same session, on the same host, minutes apart**. Session state is
held fixed by construction, so whatever difference survives is the stack.

WHY IT CAN BE DONE AT ALL
-------------------------
`.venv-fs010` pins flopscope 0.10 and has no whestbench, so the candidate is loaded
against the same minimal stub `p2_flop_anchor.py` uses. Nothing here needs the
whestbench package: it opens a `BudgetContext`, calls `predict`, and reads the
context's own residual.

WHAT IT DOES NOT ESTABLISH
--------------------------
Nothing about the grader's seconds. Local wall-time absolutes do not transfer; the
RATIO between two interpreters in one session is the quantity, and it is measured
within one session by construction. Draws no seed block.

Run on a QUIET machine, from the repo root:
    uv run python local_sim/p2_residual_version.py [--n 8] [--repeats 3]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

OUT = os.path.join(_ROOT, "experiments", "2026-08-24_p2_p0",
                   "p2_residual_version.json")

VENVS = {
    "0.10": os.path.join(_ROOT, ".venv-fs010", "Scripts", "python.exe"),
    "0.12": os.path.join(_ROOT, ".venv", "Scripts", "python.exe"),
}

WIDTH, DEPTH, BUDGET = 1024, 16, 2 ** 41
#: seed-ledger: no block named
SEED_BASE = 5_151_500


def _emit(candidate: str, n: int, repr_mode: str) -> dict:
    """Run `predict` n times under a real BudgetContext; report residual seconds."""
    import numpy as np

    import flopscope as flops
    import flopscope.numpy as fnp

    from p2_flop_anchor import _require_real_flopscope, _stub_whestbench

    _require_real_flopscope(flops)
    try:
        flops.configure(symmetry_warnings=False)
    except Exception:
        pass
    _stub_whestbench()

    from mint_provenance import canonical_mint

    import importlib.util as u
    spec = u.spec_from_file_location("cand_v", os.path.join(_ROOT, candidate))
    mod = u.module_from_spec(spec)
    spec.loader.exec_module(mod)
    est = mod.Estimator()

    class _Ctx:
        width, depth, flop_budget, seed = WIDTH, DEPTH, BUDGET, 0

    try:
        est.setup(_Ctx())
    except Exception:
        pass

    ctor = np.asarray if repr_mode == "np" else fnp.array
    resid, wall, fl = [], [], []
    for i in range(n):
        w_np, est_seed, _ = canonical_mint(WIDTH, DEPTH, SEED_BASE + i)
        mlp = sys.modules["whestbench"].MLP(
            weights=[ctor(w) for w in w_np], width=WIDTH, depth=DEPTH, seed=est_seed)
        with flops.BudgetContext(flop_budget=BUDGET, quiet=True) as ctx:
            est.predict(mlp, BUDGET)
            fl.append(float(ctx.flops_used))
        resid.append(float(ctx.residual_wall_time_s or 0.0))
        wall.append(float(ctx.wall_time_s or 0.0))

    return {"flopscope": getattr(flops, "__version__", "?"),
            "repr": repr_mode, "candidate": candidate,
            "residual_s": resid, "wall_s": wall, "flops": fl}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3,
                    help="interleaved A/B rounds; interleaving is what keeps a "
                         "slow drift from being read as a version effect")
    ap.add_argument("--candidate", default="local_sim/p2_arms/arm_k2p2.py")
    ap.add_argument("--repr", default="np", choices=("np", "fnp"),
                    help="hold the weight representation FIXED; 'np' matches the "
                         "2026-08-21 ladder")
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    if args.emit:
        print(json.dumps(_emit(args.candidate, args.n, args.repr)))
        return 0

    import numpy as np

    print("=" * 92)
    print("RESIDUAL: flopscope 0.10 vs 0.12, same host, same session, interleaved")
    print("=" * 92)
    print(f"  candidate {args.candidate}   weights={args.repr}   n={args.n} "
          f"x {args.repeats} interleaved rounds")
    print("  RUN THIS ON A QUIET MACHINE.")
    print()

    rounds = []
    for k in range(args.repeats):
        for tag in ("0.10", "0.12"):
            py = VENVS[tag]
            if not os.path.exists(py):
                print(f"  no interpreter for flopscope {tag} at {py}")
                return 1
            p = subprocess.run(
                [py, os.path.abspath(__file__), "--emit", "--n", str(args.n),
                 "--candidate", args.candidate, "--repr", args.repr],
                capture_output=True, text=True, cwd=_ROOT)
            if p.returncode != 0:
                print(f"  round {k} flopscope {tag} FAILED:\n{p.stderr[-2500:]}")
                return 1
            d = json.loads(p.stdout.strip().splitlines()[-1])
            got = str(d["flopscope"]).split("+")[0]
            if not got.startswith(tag):
                print(f"  round {k}: interpreter reports flopscope {d['flopscope']}, "
                      f"expected {tag}.x")
                return 1
            d["round"] = k
            rounds.append(d)
            r = np.asarray(d["residual_s"], float)
            print(f"  round {k}  flopscope {tag}  residual median "
                  f"{np.median(r) * 1e3:7.1f} ms   counted {d['flops'][0]:,.0f}")

    print()
    med = {}
    for tag in ("0.10", "0.12"):
        r = np.concatenate([np.asarray(x["residual_s"], float)
                            for x in rounds if x["flopscope"].startswith(tag)])
        med[tag] = float(np.median(r))
        print(f"  flopscope {tag}: residual median {med[tag] * 1e3:7.1f} ms "
              f"over {r.size} readings")
    ratio = med["0.10"] / med["0.12"] if med["0.12"] else float("nan")
    print(f"\n  0.10 / 0.12 = {ratio:.2f}x")

    print()
    print("  HOW TO READ IT")
    print("  - ratio near 2x: the STACK explains the gap against 2026-08-21, the")
    print("    older residuals are the wrong instrument, and today's are the")
    print("    design figures.")
    print("  - ratio near 1x: the stack is NOT the cause, so the residue is")
    print("    genuine BETWEEN-SESSION drift and every headroom claim in this")
    print("    repository must carry a term for it.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({
            "artifact": "p2_residual_version",
            "artifact_version": 1,
            "candidate": args.candidate,
            "repr": args.repr,
            "n_per_round": args.n,
            "repeats": args.repeats,
            "rounds": rounds,
            "median_residual_s": med,
            "ratio_010_over_012": ratio,
            "note": ("Interleaved A/B in one session on one host, so session state "
                     "is held fixed by construction and the surviving difference "
                     "is the flopscope release. Local absolutes do not transfer; "
                     "the ratio is the quantity. Draws no seed block."),
        }, fh, indent=2)
    print(f"\n  wrote {os.path.relpath(args.out, _ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
