"""Slice 2 lever 2a: what is batched Strassen-Winograd ACTUALLY worth on this chain?

The ranked lever list prices batched Strassen at `(7/8)^L` counted FLOPs and calls
L=4 worth `0.586x`. **That figure is the textbook one and it is wrong here**, for a
reason flopscope makes unavoidable: the 15 Winograd additions are BILLED, the
stacks and concatenates that batch them are BILLED at 1 FLOP per element, and the
batch axis multiplies by 7 at every level while the operands only halve. So the
glue grows as `(7/4)^L` while the saving shrinks. The textbook ratio is what you
get when additions are free. They are not.

WHAT IS MEASURED, AND WHY IT IS MEASURED RATHER THAN DERIVED
------------------------------------------------------------
Three facts were established by `local_sim/p2_batched_matmul_probe.py` and they
are what make the lever possible at all:

  1. `fnp.matmul` on 3-D operands is **ONE** flopscope operation, billed at
     exactly `b * m * n * (2k - 1)` -- no batching surcharge, verified to the FLOP
     at `b = 128`. So the seven sub-products of a level cost ONE operation.
  2. Slicing is **free** (0 FLOPs, 0 operations): quadrant extraction is a view.
  3. A whole batched Strassen level costs **23 operations**, flat, independent of
     how wide the batch already is. So operations grow LINEARLY in `L` -- which is
     the entire reason this is viable, since the repo's naive recursion issues
     ~1.1M operations per MLP and the gate is 24,000.

The gate: `PHASE2_OP_COUNT_GATE = 24,000` operations per MLP, which is `0.290 s`
at the graded `12.1 us/op` against a `0.4 s` cliff. Measured on two graded
submissions -- see `docs/VALIDATION.md` section 2b-iii.

A TRAP THIS MODULE EXISTS TO AVOID
----------------------------------
Do NOT build probe operands as `fnp.zeros(shape) + 1.0`. flopscope tags a square
all-equal array as SYMMETRIC and bills operations on it at the unique-entry rate,
so every FLOP figure taken that way is ~2x too low. Measured: a `(7,128,128)` add
billed 57,792 = `7 * 128 * 129 / 2` instead of 114,688. All operands here are
random.

ROUTINE: a lab script. It spends no seed block (`# seed-ledger: no block named` --
`canonical_mint` runs on a fixed seed outside every registered block, purely to
give `predict` something to run on), moves no gate, and changes no shipped file.

Run:  uv run python local_sim/p2_strassen_price.py --candidate estimator_k3_full.py
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# APPEND, never insert: `local_sim/` holds numpy shims named `flopscope` and
# `whestbench`, and prepending this directory would meter the shim.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import numpy as np  # noqa: E402

WIDTH, DEPTH = 1024, 16
BUDGET = 2 ** 41
B_FLOPS = float(2 ** 41)


# --------------------------------------------------------------------------
# harness -- the same minimal one `p2_op_census.py` uses
# --------------------------------------------------------------------------

def _require_real_flopscope(flops) -> None:
    path = os.path.abspath(getattr(flops, "__file__", "") or "")
    if path.startswith(os.path.abspath(_HERE) + os.sep):
        raise RuntimeError(
            f"REFUSED: `flopscope` resolved to the local shim at {path}.")


def _load(path):
    name = "cand_" + os.path.basename(path).replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class _Ctx:
    def __init__(self, seed, tmp):
        self.seed = seed
        self.scratch_dir = tmp
        self.submission_dir = _ROOT


def _mint(seed, width, depth):
    import flopscope.numpy as fnp

    from mint_provenance import canonical_mint

    w, _e, _s = canonical_mint(width, depth, seed)
    return [fnp.array(np.ascontiguousarray(np.asarray(x, dtype=np.float32)))
            for x in w]


def meter(path: str, *, width: int, depth: int, seed: int, mlp_index: int = 1):
    """Meter ONE steady-state `predict` and hand back its raw op log."""
    import flopscope
    import whestbench

    _require_real_flopscope(flopscope)
    mod = _load(path)
    est = mod.Estimator()
    est.setup(_Ctx(seed, _ROOT))          # OUTSIDE the meter: its own 5 s window

    records, used = None, 0
    for i in range(mlp_index + 1):
        weights = _mint(seed + i, width, depth)
        mlp = whestbench.MLP(weights=weights, width=width, depth=depth,
                             seed=seed + i)
        with flopscope.BudgetContext(BUDGET) as ctx:
            est.predict(mlp, BUDGET)
            used = int(ctx.flops_used)
            records = list(ctx.op_log)
    return records, used


# --------------------------------------------------------------------------
# the Strassen cost model -- MEASURED per shape class, not derived
# --------------------------------------------------------------------------

_RNG = np.random.default_rng(20260907)


def _rand(*shape):
    """A NON-SYMMETRIC operand. See the trap in the module docstring."""
    import flopscope.numpy as fnp
    return fnp.asarray(_RNG.standard_normal(shape).astype(np.float32))


def strassen_batched(A, B, depth):
    """`A` is `(b,m,k)`, `B` is `(b,k,n)`. One batched matmul per level.

    Every level multiplies the batch axis by 7 and halves m, k and n, so the
    whole level is issued as ONE batched call however wide the batch has grown.
    That is what keeps operations linear in `depth` instead of `7^depth`.
    """
    import flopscope.numpy as fnp

    b, m, k = A.shape
    n = B.shape[2]
    if depth == 0 or m % 2 or k % 2 or n % 2:
        return A @ B
    mh, kh, nh = m // 2, k // 2, n // 2

    a11, a12 = A[:, :mh, :kh], A[:, :mh, kh:]
    a21, a22 = A[:, mh:, :kh], A[:, mh:, kh:]
    b11, b12 = B[:, :kh, :nh], B[:, :kh, nh:]
    b21, b22 = B[:, kh:, :nh], B[:, kh:, nh:]

    SA = fnp.concatenate([a11 + a22, a21 + a22, a11, a22,
                          a11 + a12, a21 - a11, a12 - a22], axis=0)
    SB = fnp.concatenate([b11 + b22, b11, b12 - b22, b21 - b11,
                          b22, b11 + b12, b21 + b22], axis=0)

    M = strassen_batched(SA, SB, depth - 1)              # (7b, mh, nh)
    m1, m2, m3, m4, m5, m6, m7 = (M[i * b:(i + 1) * b] for i in range(7))

    c11 = m1 + m4 - m5 + m7
    c12 = m3 + m5
    c21 = m2 + m4
    c22 = m1 - m2 + m3 + m6
    return fnp.concatenate([fnp.concatenate([c11, c12], axis=2),
                            fnp.concatenate([c21, c22], axis=2)], axis=1)


def price_shape(m, k, n, batch, depth):
    """MEASURE (FLOPs, ops) for `batch` matmuls of `(m,k)@(k,n)` at Strassen `depth`.

    Measured, not modelled: the glue's billing rules are flopscope's, not ours.
    """
    import flopscope as flops

    A, B = _rand(batch, m, k), _rand(batch, k, n)
    with flops.BudgetContext(flop_budget=int(1e16)) as ctx:
        strassen_batched(A, B, depth)
        return int(ctx.flops_used), len(ctx.op_log)


# --------------------------------------------------------------------------

def matmul_census(records):
    """Group the matmul operations by operand shape."""
    g = collections.defaultdict(lambda: {"calls": 0, "flops": 0})
    for rec in records:
        if rec.op_name != "matmul":
            continue
        key = tuple(tuple(s) for s in rec.shapes)
        g[key]["calls"] += 1
        g[key]["flops"] += int(rec.flop_cost)
    rows = [{"shapes": k, **v} for k, v in g.items()]
    rows.sort(key=lambda r: -r["flops"])
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidate", default="estimator_k3_full.py")
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--seed", type=int, default=1_000_003)
    ap.add_argument("--max-level", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    import official_config as oc

    gate = oc.PHASE2_OP_COUNT_GATE
    per_op = oc.GRADED_RESIDUAL_S_PER_OP

    records, used = meter(a.candidate, width=a.width, depth=a.depth,
                          seed=a.seed)
    n_ops = len(records)
    rows = matmul_census(records)
    mm_flops = sum(r["flops"] for r in rows)
    mm_calls = sum(r["calls"] for r in rows)

    print("=" * 92)
    print(f"STRASSEN PRICE  --  {a.candidate}   {a.width} x {a.depth}")
    print("=" * 92)
    print(f"  metered            {used:>20,} FLOPs   {n_ops:,} operations")
    print(f"  of which matmul    {mm_flops:>20,} FLOPs   {mm_calls:,} calls"
          f"   ({100.0 * mm_flops / used:.2f}% of FLOPs,"
          f" {100.0 * mm_calls / n_ops:.2f}% of operations)")
    print(f"  B-fraction         {used / B_FLOPS:>20.5f}")
    print(f"  op gate            {gate:,} ops   at {per_op * 1e6:.1f} us/op"
          f" -> {gate * per_op:.3f} s of a 0.400 s cliff")
    print(f"  op headroom        {gate - n_ops:,} operations to spend")

    print("\n  MATMUL SHAPE CENSUS (the 93% of the bill Slice 2 is aimed at)")
    print(f"  {'m':>6}{'k':>7}{'n':>6}{'calls':>7}{'FLOPs':>20}{'% of total':>12}"
          f"{'even?':>7}")
    classes = []
    for r in rows:
        (sa, sb) = r["shapes"]
        if len(sa) != 2 or len(sb) != 2:
            print(f"  (non-2D matmul operand {r['shapes']}, {r['calls']} calls)")
            continue
        m, k = sa
        k2, n = sb
        even = "yes" if (m % 2 == 0 and k % 2 == 0 and n % 2 == 0) else "NO"
        print(f"  {m:>6}{k:>7}{n:>6}{r['calls']:>7}{r['flops']:>20,}"
              f"{100.0 * r['flops'] / used:>11.2f}%{even:>7}")
        classes.append((m, k, n, r["calls"], r["flops"]))

    # ---- price each class at each level, MEASURED -------------------------
    print("\n  MEASURED PRICE PER SHAPE CLASS  (batched, glue included)")
    print(f"  {'m':>6}{'k':>7}{'n':>6}{'calls':>7}{'L':>3}"
          f"{'FLOPs':>20}{'x base':>9}{'(7/8)^L':>9}{'ops':>7}{'saved/op':>14}")
    best = {}
    for (m, k, n, calls, base_flops) in classes:
        f0, o0 = price_shape(m, k, n, calls, 0)
        for L in range(0, a.max_level + 1):
            if L == 0:
                f, o = f0, o0
            else:
                try:
                    f, o = price_shape(m, k, n, calls, L)
                except Exception as exc:            # noqa: BLE001
                    print(f"  {m:>6}{k:>7}{n:>6}{calls:>7}{L:>3}"
                          f"   REFUSED: {type(exc).__name__}: {exc}")
                    break
            saved = f0 - f
            dops = o - o0
            print(f"  {m:>6}{k:>7}{n:>6}{calls:>7}{L:>3}{f:>20,}"
                  f"{f / f0:>9.4f}{(7 / 8) ** L:>9.4f}{o:>7}"
                  f"{(saved / dops if dops else 0):>14,.0f}")
            best.setdefault((m, k, n), []).append((L, f, o, f0, o0))
        print()

    # ---- the whole-chain roll-up ------------------------------------------
    print("  WHOLE-CHAIN ROLL-UP: apply the SAME level to every class")
    print(f"  {'L':>3}{'chain FLOPs':>20}{'B':>10}{'x now':>8}"
          f"{'ops':>9}{'gate':>8}{'residual s':>12}{'adj score':>12}")
    RAW = 3.7793e-08          # graded raw final-layer MSE of 330212
    for L in range(0, a.max_level + 1):
        tot_f, tot_o = used, n_ops
        ok = True
        for (m, k, n), entries in best.items():
            e = [x for x in entries if x[0] == L]
            if not e:
                ok = False
                break
            _L, f, o, f0, o0 = e[0]
            tot_f += (f - f0)
            tot_o += (o - o0)
        if not ok:
            print(f"  {L:>3}   (a class refused at this level)")
            continue
        mult = max(0.1, tot_f / B_FLOPS)
        print(f"  {L:>3}{tot_f:>20,}{tot_f / B_FLOPS:>10.5f}"
              f"{tot_f / used:>8.4f}{tot_o:>9,}"
              f"{'ok' if tot_o <= gate else 'OVER':>8}"
              f"{tot_o * per_op:>12.3f}{mult * RAW:>12.4e}")
    print("\n  'adj score' holds the raw MSE at the GRADED 3.7793e-08 of 330212 and")
    print("  moves only the multiplier. It is what Strassen is worth IF it is")
    print("  exact -- parity has to be measured separately, and float32")
    print("  reassociation at L=3 was 1.3e-06 relative in the probe.")

    if a.out:
        payload = {"candidate": a.candidate, "width": a.width, "depth": a.depth,
                   "metered_flops": used, "n_ops": n_ops,
                   "matmul_flops": mm_flops, "matmul_calls": mm_calls,
                   "op_gate": gate, "graded_s_per_op": per_op,
                   "classes": [{"m": m, "k": k, "n": n, "calls": c, "flops": fl}
                               for (m, k, n, c, fl) in classes],
                   "priced": {f"{m}x{k}x{n}": [
                       {"L": L, "flops": f, "ops": o} for (L, f, o, _f0, _o0)
                       in entries] for (m, k, n), entries in best.items()}}
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
