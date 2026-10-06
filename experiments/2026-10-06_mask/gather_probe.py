"""P2 prerequisite: device-side index selection and gathers on the REAL flopscope meter.

    uv run python experiments/2026-10-06_mask/gather_probe.py

Questions (HANDOFF 2026-10-06 Lane P, P2 variant A): does an integer index array from
`fnp.argsort` / `fnp.argpartition` feed `fnp.take` / fancy indexing with ZERO scalar
materializations, and what does each step bill? Also the scatter-as-gather spelling:
pad the compact block with a zero row/column and `take` with an inverse index, so no
array is ever assigned into (grader arrays are immutable).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "local_sim"))           # append, never insert (CLAUDE.md)

import flopscope as flops  # noqa: E402
import flopscope.numpy as fnp  # noqa: E402
from p2_weff import MaterializationCounter  # noqa: E402

n, K = 1024, 896
rng = fnp.random.default_rng(0)
score = rng.standard_normal(n)
A = rng.standard_normal((n, 4 * n))
B = rng.standard_normal((n, n))
rows = []


def step(name, fn):
    with MaterializationCounter() as mc:
        with flops.BudgetContext(flop_budget=2 ** 41, quiet=True) as b:
            out = fn()
    try:
        ns = b.summary_dict(by_namespace=True).get("by_namespace") or {}
        ops = sum(float(v.get("calls", 0.0)) for v in ns.values())
    except Exception:
        ops = float("nan")
    rows.append({"step": name, "flops": int(b.flops_used), "ops": ops,
                 "materializations": sum(mc.counts.values()), "detail": mc.counts,
                 "shape": list(getattr(out, "shape", ()))})
    print(json.dumps(rows[-1]), flush=True)
    return out


perm = step("argsort(n)", lambda: fnp.argsort(score))
keep = step("perm[-K:] (view)", lambda: perm[-K:])
part = step("argpartition(n, n-K)", lambda: fnp.argpartition(score, n - K)[n - K:])
step("take rows A[keep] (K x 4n)", lambda: fnp.take(A, keep, axis=0))
step("fancy A[keep] (K x 4n)", lambda: A[keep])
step("take cols A[:, keep]", lambda: fnp.take(A, keep, axis=1))
step("take 2x B[keep][:, keep] (K x K)",
     lambda: fnp.take(fnp.take(B, keep, axis=0), keep, axis=1))
rank = step("argsort(perm) = rank", lambda: fnp.argsort(perm))
inv = step("inverse index where(rank>=n-K, rank-(n-K), K)",
           lambda: fnp.where(rank >= n - K, rank - (n - K), K))
Bc = fnp.take(fnp.take(B, keep, axis=0), keep, axis=1)
step("scatter-as-gather: pad K+1 then take x2 (n x n)",
     lambda: fnp.take(fnp.take(fnp.pad(Bc, ((0, 1), (0, 1))), inv, axis=0), inv, axis=1))
step("matmul W[:, keep]^T-style (n x K)@(K x 4n)",
     lambda: fnp.take(B, keep, axis=1) @ fnp.take(A, keep, axis=0))

(Path(__file__).parent / "gather_probe.json").write_text(json.dumps(rows, indent=1))
