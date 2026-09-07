"""Lever 2a, corrected: the binding constraint on Strassen depth is MEMORY, not operations.

WHAT THE HANDOFF ASSUMED, AND WHAT THE MEASUREMENT SAYS
-------------------------------------------------------
The Slice 2 handoff says to "ship the deepest L whose ops <= 24k", predicting L=3
"at ~15-20k ops" and calling L=4 "feasible only if the per-layer batching gets the
base op count under ~4k". Both halves of that are wrong at this chain's shapes,
and in opposite directions:

  * The base op count is already 3,232, and a FULLY BATCHED Strassen level costs a
    FLAT 23 operations per shape class regardless of how wide the batch has grown
    (`local_sim/p2_batched_matmul_probe.py`). With 28 shape classes that is 644
    operations per level, so even L=5 lands near 6,500 -- comfortably inside the
    24,000 gate. **Operations are not the binding constraint.**

  * What binds instead is memory, which the handoff does not mention. Every level
    multiplies the batch axis by 7 while halving each dimension, so the operand
    stacks grow as `(7/4)^L = 1.75, 3.06, 5.36, 9.38` for L = 1..4. The widest
    shape class here is `(1024, 14336) @ (14336, 1024)` with 4 calls, whose two
    operands are 235 MB each; at L=4 a fully batched stack is `2.2 GB` per
    operand. The graded memory limit is **8,192 MB** and this candidate already
    peaks at `2,827.5 MB`. A fully batched L=4 does not fit.

THE SPELLING THAT FIXES IT
--------------------------
Expand the TOP `s` recursion levels as `7^s` sequential sub-problems and batch
everything below them. Sequential branches never coexist, so peak memory is set by
one branch's subtree rather than by the whole stack, while the batched tail keeps
the operation count linear. `s = 1` divides peak memory by about 7 and costs only
a few hundred extra operations per class -- which this chain has in abundance.

This module measures FLOPs, operations AND peak RSS for `(L, s)` pairs at the
REAL shapes, so the trade is priced rather than argued.

THE TRAP, AGAIN
---------------
Operands here are random. `fnp.zeros(shape) + 1.0` is tagged SYMMETRIC by
flopscope and billed at the unique-entry rate -- a `(7,128,128)` add came to
57,792 FLOPs instead of 114,688 -- so every FLOP figure taken that way is ~2x low.

ROUTINE: a lab script. Spends no seed block (`# seed-ledger: no block named`),
moves no gate, changes no shipped file.

Run:  uv run python local_sim/p2_strassen_memory.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import numpy as np  # noqa: E402

B_FLOPS = float(2 ** 41)

#: The chain's matmul shape classes, from `p2_strassen_price.py --out`.
#: (m, k, n, calls). 148 calls carrying 99.80% of 1,714,554,351,206 FLOPs.
CLASSES = [
    (1024, 14336, 1024, 4), (1024, 13312, 1024, 4), (1024, 12288, 1024, 4),
    (1024, 11264, 1024, 4), (1024, 1024, 14336, 3), (1024, 10240, 1024, 4),
    (1024, 1024, 1024, 39), (1024, 1024, 13312, 3), (1024, 9216, 1024, 4),
    (1024, 1024, 12288, 3), (1024, 1024, 11264, 3), (1024, 8192, 1024, 4),
    (1024, 1024, 10240, 3), (1024, 1024, 15360, 2), (1024, 7168, 1024, 4),
    (1024, 1024, 9216, 3), (1024, 6144, 1024, 4), (1024, 1024, 8192, 3),
    (1024, 1024, 7168, 3), (1024, 5120, 1024, 4), (1024, 1024, 6144, 3),
    (1024, 4096, 1024, 4), (1024, 1024, 5120, 3), (1024, 3072, 1024, 4),
    (1024, 1024, 4096, 3), (1024, 1024, 3072, 3), (1024, 2048, 1024, 4),
    (1024, 1024, 2048, 3),
]
BASE_FLOPS = 1_714_554_351_206
BASE_OPS = 3_232
#: The graded raw final-layer MSE of submission 330212. Held fixed: Strassen is
#: an EXACT transformation, so only the multiplier is allowed to move.
GRADED_RAW = 3.7793e-08

_RNG = np.random.default_rng(20260907)


def _rand(*shape):
    """A NON-SYMMETRIC operand. See the trap in the module docstring."""
    import flopscope.numpy as fnp
    return fnp.asarray(_RNG.standard_normal(shape).astype(np.float32))


_PEAK = None


def _peak_rss_mb() -> float:
    """PEAK working set of this process, in MB. Windows first, then POSIX.

    `psutil` is not installed in this environment, so Windows goes through
    `K32GetProcessMemoryInfo` directly. The value is a process-lifetime PEAK and
    never decreases, so a per-class DELTA is meaningless once anything large has
    been allocated -- price one class per process (see `--only-worst`).
    """
    global _PEAK
    if _PEAK is None:
        _PEAK = _make_peak()
    return _PEAK()


def _make_peak():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]

        fn = None
        for lib, name in ((ctypes.windll.kernel32, "K32GetProcessMemoryInfo"),
                          (ctypes.windll.psapi, "GetProcessMemoryInfo")):
            try:
                fn = getattr(lib, name)
                break
            except AttributeError:
                continue
        fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        fn.restype = wintypes.BOOL
        h = ctypes.windll.kernel32.GetCurrentProcess()

        def peak():
            c = PMC()
            c.cb = ctypes.sizeof(PMC)
            if not fn(h, ctypes.byref(c), c.cb):
                raise ctypes.WinError()
            return c.PeakWorkingSetSize / (1024.0 * 1024.0)
        return peak

    def peak_posix():
        import resource
        return float(resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    return peak_posix


# --------------------------------------------------------------------------

def _combos(A, B):
    """The seven Winograd operand pairs, as a list of `(SA_i, SB_i)`.

    Slicing is free in flopscope (0 FLOPs, 0 operations -- it is a view), so the
    only billed work here is the ten additions.
    """
    b, m, k = A.shape
    n = B.shape[2]
    mh, kh, nh = m // 2, k // 2, n // 2
    a11, a12 = A[:, :mh, :kh], A[:, :mh, kh:]
    a21, a22 = A[:, mh:, :kh], A[:, mh:, kh:]
    b11, b12 = B[:, :kh, :nh], B[:, :kh, nh:]
    b21, b22 = B[:, kh:, :nh], B[:, kh:, nh:]
    return [
        (a11 + a22, b11 + b22),      # M1
        (a21 + a22, b11),            # M2
        (a11, b12 - b22),            # M3
        (a22, b21 - b11),            # M4
        (a11 + a12, b22),            # M5
        (a21 - a11, b11 + b12),      # M6
        (a12 - a22, b21 + b22),      # M7
    ]


def _reassemble(m1, m2, m3, m4, m5, m6, m7):
    import flopscope.numpy as fnp
    c11 = m1 + m4 - m5 + m7
    c12 = m3 + m5
    c21 = m2 + m4
    c22 = m1 - m2 + m3 + m6
    return fnp.concatenate([fnp.concatenate([c11, c12], axis=2),
                            fnp.concatenate([c21, c22], axis=2)], axis=1)


def strassen(A, B, depth, split_top=0):
    """Strassen-Winograd with the top `split_top` levels expanded SEQUENTIALLY.

    `A` is `(b,m,k)`, `B` is `(b,k,n)`.

    A batched level issues ONE matmul for all `7^L` sub-products, which is what
    keeps operations linear -- but it must hold the whole `(7/4)^L` stack at once.
    An expanded level issues its seven sub-problems one after another, so only one
    subtree's memory is live. `split_top` chooses how many levels pay operations
    to save memory.
    """
    import flopscope.numpy as fnp

    b, m, k = A.shape
    n = B.shape[2]
    if depth == 0 or m % 2 or k % 2 or n % 2:
        return A @ B

    if split_top > 0:
        parts = []
        for (sa, sb) in _combos(A, B):
            parts.append(strassen(sa, sb, depth - 1, split_top - 1))
        return _reassemble(*parts)

    pairs = _combos(A, B)
    SA = fnp.concatenate([p[0] for p in pairs], axis=0)
    SB = fnp.concatenate([p[1] for p in pairs], axis=0)
    del pairs
    M = strassen(SA, SB, depth - 1, 0)
    del SA, SB
    return _reassemble(*(M[i * b:(i + 1) * b] for i in range(7)))


def price(m, k, n, batch, depth, split_top):
    """(FLOPs, operations, peak RSS delta in MB) for one shape class."""
    import flopscope as flops

    A, B = _rand(batch, m, k), _rand(batch, k, n)
    before = _peak_rss_mb()
    with flops.BudgetContext(flop_budget=int(1e16)) as ctx:
        strassen(A, B, depth, split_top)
        f, o = int(ctx.flops_used), len(ctx.op_log)
    return f, o, _peak_rss_mb() - before


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--levels", default="0,1,2,3,4",
                    help="comma-separated Strassen depths to price")
    ap.add_argument("--split-top", type=int, default=1,
                    help="levels expanded sequentially (0 = fully batched)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    levels = [int(x) for x in a.levels.split(",")]

    import official_config as oc
    gate, per_op = oc.PHASE2_OP_COUNT_GATE, oc.GRADED_RESIDUAL_S_PER_OP

    print("=" * 96)
    print(f"STRASSEN DEPTH vs MEMORY   split_top={a.split_top}"
          f"   ({'fully batched' if a.split_top == 0 else f'top {a.split_top} level(s) sequential'})")
    print("=" * 96)
    print(f"  base chain     {BASE_FLOPS:,} FLOPs   {BASE_OPS:,} operations"
          f"   {BASE_FLOPS / B_FLOPS:.5f} B")
    print(f"  op gate        {gate:,}   at {per_op * 1e6:.1f} us/op"
          f" -> {gate * per_op:.3f} s of a 0.400 s cliff")
    print(f"  memory limit   8,192 MB graded; this candidate peaks at 2,827.5 MB")
    print(f"  NOTE: the 8,192 MB limit is UNENFORCEABLE on this host (no"
          f" RLIMIT_AS on Windows).\n         The RSS column below is"
          f" ACCOUNTING, not a test passed -- confirm on"
          f"\n         `local_sim/linux_memory_gate.py`. docs/GOTCHAS.md.")

    rows = []
    print(f"\n  {'L':>3}{'chain FLOPs':>20}{'B':>9}{'x now':>8}{'ops':>8}"
          f"{'gate':>7}{'worst class RSS MB':>21}{'adj score':>12}")
    for L in levels:
        tot_f, tot_o, worst_rss, worst_cls = BASE_FLOPS, BASE_OPS, 0.0, None
        failed = None
        for (m, k, n, b) in CLASSES:
            try:
                f0, o0, _ = price(m, k, n, b, 0, 0)
                f, o, rss = price(m, k, n, b, L, a.split_top)
            except MemoryError as exc:                          # noqa: BLE001
                failed = f"MemoryError at ({m},{k},{n})x{b}: {exc}"
                break
            tot_f += (f - f0)
            tot_o += (o - o0)
            if rss > worst_rss:
                worst_rss, worst_cls = rss, (m, k, n, b)
        if failed:
            print(f"  {L:>3}   {failed}")
            continue
        mult = max(0.1, tot_f / B_FLOPS)
        print(f"  {L:>3}{tot_f:>20,}{tot_f / B_FLOPS:>9.5f}{tot_f / BASE_FLOPS:>8.4f}"
              f"{tot_o:>8,}{'ok' if tot_o <= gate else 'OVER':>7}"
              f"{worst_rss:>21,.0f}{mult * GRADED_RAW:>12.4e}")
        rows.append({"L": L, "split_top": a.split_top, "flops": tot_f,
                     "B": tot_f / B_FLOPS, "ops": tot_o,
                     "worst_class_rss_mb": worst_rss,
                     "worst_class": worst_cls,
                     "adj_score_at_graded_raw": mult * GRADED_RAW})

    print("\n  'adj score' holds the raw MSE at the GRADED 3.7793e-08 of submission")
    print("  330212 and moves ONLY the multiplier, which is the whole claim:")
    print("  Strassen is exact in real arithmetic. Parity in float32 is a separate")
    print("  measurement and is NOT established by this table.")

    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"split_top": a.split_top, "base_flops": BASE_FLOPS,
                       "base_ops": BASE_OPS, "op_gate": gate,
                       "graded_s_per_op": per_op, "graded_raw": GRADED_RAW,
                       "rows": rows}, fh, indent=2)
        print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
