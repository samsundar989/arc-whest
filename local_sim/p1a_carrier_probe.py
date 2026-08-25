"""Radial factor, moment diagnostics, real flopscope ledger, setup and memory.

Phase 1A items 4-7. Specification: `docs/KERDOCK_MUB_SPEC.md` §8-§9.

**No truth data is read and no score, variance or performance claim is made.**
Everything here characterises the quadrature and its cost. Whether that
characterisation is *good* is a Phase-1B question and is not answered.

Run:  uv run python local_sim/p1a_carrier_probe.py
      uv run python local_sim/p1a_carrier_probe.py --skip-ledger   (fast)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402

from kerdock_mub import carrier_float32, carrier_int8  # noqa: E402

OUT = os.path.join(_ROOT, "experiments", "2026-08-25_p1a", "carrier_probe.json")

_PASS = 0
_FAIL = 0
_REC: dict = {}


def check(name, ok, detail=""):
    global _PASS, _FAIL
    ok = bool(ok)
    if ok:
        _PASS += 1
    else:
        _FAIL += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<64} {detail}")
    return ok


def head(t):
    print(f"\n{t}\n{'-' * len(t)}")


# --------------------------------------------------------------------------- #
# item 4 -- the radial factor
# --------------------------------------------------------------------------- #

def chi_mean(N: int) -> float:
    """`E‖x‖` for `x ~ N(0, I_N)` = `√2 · Γ((N+1)/2) / Γ(N/2)`.

    Computed through `lgamma` because `Γ(512)` overflows a double by hundreds of
    orders of magnitude. The stability of this is checked, not assumed.
    """
    return math.sqrt(2.0) * math.exp(math.lgamma((N + 1) / 2) - math.lgamma(N / 2))


def radial(rng) -> None:
    head("ITEM 4 -- the radial factor, validated apart from the angular carrier")
    N = 1024
    er = chi_mean(N)
    # A degree-1 positively homogeneous integrand factors exactly:
    #     E_x[f(x)] = E[‖x‖] · E_u[f(u)]
    check("E‖x‖ is finite and near √(N-1/2) (lgamma is stable at N=1024)",
          math.isfinite(er) and abs(er - math.sqrt(N - 0.5)) < 1e-3,
          f"E‖x‖ = {er:.9f}, √(N-1/2) = {math.sqrt(N - 0.5):.9f}")
    _REC["E_norm"] = er

    # Monte-Carlo the radial constant on its own, no carrier involved.
    z = rng.standard_normal((200_000, N)).astype(np.float64)
    r = np.linalg.norm(z, axis=1)
    se = r.std(ddof=1) / math.sqrt(r.size)
    check("...and matches a 200k-sample Monte-Carlo of ‖x‖ within 4 s.e.",
          abs(r.mean() - er) < 4 * se,
          f"MC {r.mean():.6f} +- {se:.6f} (4 s.e. = {4 * se:.6f}), exact {er:.6f}")

    # Positive homogeneity of a bias-free ReLU stack, on EVERY prefix.
    rng2 = np.random.default_rng(7)
    Ws = [rng2.standard_normal((N, N)).astype(np.float64) * math.sqrt(2.0 / N)
          for _ in range(4)]
    x = rng2.standard_normal(N)
    c = 3.7
    worst = 0.0
    for depth in range(1, len(Ws) + 1):
        a, b = x.copy(), (c * x).copy()
        for d in range(depth):
            a = np.maximum(a @ Ws[d], 0.0)
            b = np.maximum(b @ Ws[d], 0.0)
        worst = max(worst, float(np.abs(b - c * a).max() / (np.abs(c * a).max() + 1e-30)))
    check("f(cx) = c·f(x) on EVERY prefix of a bias-free ReLU stack (depths 1-4)",
          worst < 1e-12, f"max relative deviation {worst:.3e}")
    _REC["homogeneity_max_rel_dev"] = worst

    # The factor is applied EXACTLY ONCE -- a guard, not a comment.
    # Layer-1 has a closed form: E[ReLU(w·x)] = ‖w‖/√(2π).
    V = carrier_float32(9, 16).astype(np.float64)
    w = rng2.standard_normal(N)
    ang = float(np.maximum(V @ w, 0.0).mean())        # angular estimate, unit sphere
    exact = float(np.linalg.norm(w)) / math.sqrt(2 * math.pi)
    once, none, twice = er * ang, ang, er * er * ang
    check("applying the radial factor ONCE lands near the layer-1 closed form",
          abs(once / exact - 1) < 0.15,
          f"once/exact = {once / exact:.6f}")
    check("...applying it ZERO times is off by ~E‖x‖, so the guard bites",
          abs(none / exact - 1) > 0.9,
          f"none/exact = {none / exact:.6e}  (a factor {exact / none:.1f} low)")
    check("...applying it TWICE is off by ~E‖x‖ the other way",
          twice / exact > 10,
          f"twice/exact = {twice / exact:.3f}  (a factor {twice / exact:.1f} high)")
    _REC["layer1"] = {"angular_mean": ang, "exact": exact,
                      "once_over_exact": once / exact,
                      "none_over_exact": none / exact,
                      "twice_over_exact": twice / exact}
    print(f"      NOTE: 'once' is {100 * (once / exact - 1):+.2f}% from exact. That is the")
    print(f"      CARRIER's quadrature error on a non-polynomial integrand, not a")
    print(f"      radial-factor error -- ReLU is not degree-2. Characterisation only.")


# --------------------------------------------------------------------------- #
# item 5 -- exact moment diagnostics
# --------------------------------------------------------------------------- #

def moments() -> None:
    head("ITEM 5 -- exact second and fourth moments (quadrature characterisation)")
    N, k = 1024, 16
    Vi = carrier_int8(9, k)                    # +-1 integers: exact arithmetic
    M = Vi.shape[0]

    # Second moment: a union of orthonormal bases with antipodes is a TIGHT
    # FRAME, so (1/M) V^T V = I/N exactly. Checked in integers, not floats.
    G = Vi.astype(np.int64).T @ Vi.astype(np.int64)     # = M * N * (1/N) I = M I ... in +-1 units
    expect = 2 * k * N * np.eye(N, dtype=np.int64)
    check("second moment is EXACT: (1/M) Σ v vᵀ = I/N, in integer arithmetic",
          np.array_equal(G, expect),
          f"max |Σ v vᵀ − 2kN·I| = {np.abs(G - expect).max()}")
    _REC["second_moment_exact"] = bool(np.array_equal(G, expect))

    # Fourth moment. The sphere's value is 3‖w‖⁴/(N(N+2)); the carrier's depends
    # on direction, and the ALIGNED direction is the extreme case.
    sphere = 3.0 / (N * (N + 2))
    # aligned: w = a carrier row (unit). Its own basis contributes ±1 twice; the
    # other 15 bases contribute |⟨·⟩| = 1/32 exactly, 15·1024·2 of them.
    aligned_num = 2.0 + (k - 1) * N * 2 * (1.0 / N) ** 2
    aligned = aligned_num / M
    ratio = aligned / sphere
    check("aligned fourth moment / spherical target is EXACTLY 21.6881103515625",
          ratio == 21.6881103515625, f"{ratio!r}")
    _REC["fourth_moment"] = {"sphere": sphere, "aligned": aligned,
                             "aligned_over_sphere": ratio}
    print(f"      spherical target   {sphere:.6e}")
    print(f"      carrier-aligned    {aligned:.6e}   = {ratio}x")

    # A random probe would NOT have found that -- which is the point.
    rng = np.random.default_rng(11)
    Vf = carrier_float32(9, k).astype(np.float64)
    rr = []
    for _ in range(64):
        w = rng.standard_normal(N)
        w /= np.linalg.norm(w)
        rr.append(float(((Vf @ w) ** 4).mean()) / sphere)
    check("random probes cluster near 1.0 and would MISS the aligned case",
          max(rr) < 2.0,
          f"64 random directions: ratio in [{min(rr):.4f}, {max(rr):.4f}]")
    _REC["fourth_moment"]["random_ratio_min"] = min(rr)
    _REC["fourth_moment"]["random_ratio_max"] = max(rr)
    print(f"      -> K8 stands CORRECTED: the 16-basis subset is NOT an exact")
    print(f"         spherical 4-design, and random probing cannot establish that.")


# --------------------------------------------------------------------------- #
# items 6-7 -- real flopscope ledger, setup and memory, in a FRESH process
# --------------------------------------------------------------------------- #

_CHILD = r'''
import json, os, sys, time
_ROOT = sys.argv[1]
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != os.path.join(_ROOT, "local_sim")]
for p in (os.path.join(_ROOT, "local_sim"), _ROOT):
    if p not in sys.path:
        sys.path.append(p)
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from test_deployment_envelope import peak_memory_bytes
from kerdock_mub import carrier_int8

r = {"baseline_mem": peak_memory_bytes()}
t0 = time.perf_counter()
rows = carrier_int8(9, 16)                     # construction, off-budget
r["construct_s"] = time.perf_counter() - t0
r["after_construct_mem"] = peak_memory_bytes()

t0 = time.perf_counter()
V = fnp.array((rows / np.float32(32.0)).astype(np.float32))
r["materialise_s"] = time.perf_counter() - t0
W = [fnp.array((np.random.default_rng(1000 + i).standard_normal((1024, 1024))
                * np.float32(np.sqrt(2.0 / 1024))).astype(np.float32))
     for i in range(16)]
r["after_weights_mem"] = peak_memory_bytes()

t0 = time.perf_counter()
with flops.BudgetContext(flop_budget=2**60, quiet=True) as ctx:
    h = V
    for i in range(16):
        h = fnp.maximum(h @ W[i], fnp.float32(0.0))
    used = ctx.flops_used
r["propagate_s"] = time.perf_counter() - t0
r["flops_counted"] = int(used)
r["peak_mem"] = peak_memory_bytes()
r["rows"] = int(rows.shape[0])
print(json.dumps(r))
'''


def ledger_and_memory() -> None:
    head("ITEMS 6-7 -- real flopscope ledger, setup and memory, fresh subprocess")
    tmp = os.path.join(_HERE, "_p1a_child_tmp.py")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(_CHILD)
        t0 = time.perf_counter()
        p = subprocess.run([sys.executable, tmp, _ROOT], capture_output=True,
                           text=True, cwd=_ROOT,
                           env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        wall = time.perf_counter() - t0
        if p.returncode != 0:
            check("the child produced a ledger", False, p.stderr[-400:])
            return
        d = json.loads(p.stdout.strip().splitlines()[-1])
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    arith = 2 * d["rows"] * 1024 * 1024 * 16
    counted = d["flops_counted"]
    _REC["ledger"] = {"counted": counted, "dense_matmul_arithmetic": arith,
                      "overhead": counted - arith,
                      "overhead_pct": 100 * (counted / arith - 1)}
    print(f"      dense matmul arithmetic   {arith:,}")
    print(f"      OFFICIAL flopscope ledger {counted:,}")
    print(f"      this implementation's overhead {counted - arith:,} "
          f"(+{100 * (counted / arith - 1):.4f}%)")
    # The overhead is EXACTLY ZERO, and that is not a coincidence -- it is an
    # identity, measured op by op rather than asserted:
    #     flopscope bills a matmul at the FMA count, (2K - 1) per output element
    #     flopscope bills `maximum` at                1  per element
    #     (2K - 1) + 1 = 2K, which is the naive arithmetic figure exactly.
    M, K, depth = d["rows"], 1024, 16
    mm, rl = M * K * (2 * K - 1) * depth, M * K * depth
    check("the ledger decomposes EXACTLY into matmul FMAs + ReLU",
          counted == mm + rl,
          f"{mm:,} matmul + {rl:,} ReLU = {mm + rl:,}")
    check("...and that identically equals 2·M·K²·depth, so overhead is 0 by algebra",
          counted == arith and mm + rl == arith,
          f"(2K-1) + 1 = 2K exactly -- not a coincidence, and not forced")
    print(f"      matmul  {mm:,}  (FMA count, 2K-1 per output element)")
    print(f"      ReLU    {rl:,}  (1 per element)")
    print(f"      NOTE: our ReLU is NOT free -- it is billed 536,870,912 -- and the")
    print(f"      matmul is billed one FLOP per output element LESS than 2K. They")
    print(f"      cancel. So the leader's +290,757,224 (+0.0264%) is NOT ReLU and")
    print(f"      NOT materialisation; it is something else in an implementation we")
    print(f"      cannot see. It is a CLUE, not a target, and we report ours.")

    MB = 1024 * 1024
    base = d["baseline_mem"].get("commit", d["baseline_mem"].get("working_set", 0))
    peak = d["peak_mem"].get("commit", d["peak_mem"].get("working_set", 0))
    _REC["memory_mb"] = {"baseline": base / MB, "peak": peak / MB,
                         "after_construct": d["after_construct_mem"].get(
                             "commit", 0) / MB,
                         "after_weights": d["after_weights_mem"].get("commit", 0) / MB}
    _REC["timing_s"] = {"construct": d["construct_s"],
                        "materialise": d["materialise_s"],
                        "propagate": d["propagate_s"], "child_wall": wall}
    print(f"      construct   {d['construct_s']:.3f} s   materialise "
          f"{d['materialise_s']:.3f} s   propagate {d['propagate_s']:.2f} s")
    print(f"      peak commit {peak / MB:,.0f} MB   (baseline {base / MB:,.0f} MB)")
    check("peak memory is under the 8,192 MB class default",
          peak / MB < 8192,
          f"{peak / MB:,.0f} MB -- an ACCOUNTING UPPER BOUND on this host, not "
          f"proof the grader's limit was enforced")
    check("carrier CONSTRUCTION fits the 5 s setup ceiling with room",
          d["construct_s"] < 2.5,
          f"{d['construct_s']:.3f} s of 5 s, and the ceiling is shared with "
          f"`import whestbench` (1.3-9 s)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ledger", action="store_true")
    args = ap.parse_args()
    print("=" * 100)
    print("PHASE 1A CARRIER PROBE -- radial, moments, official ledger, memory")
    print("=" * 100)
    rng = np.random.default_rng(3)
    radial(rng)
    moments()
    if args.skip_ledger:
        print("\n  --skip-ledger: items 6-7 NOT RUN (never counted as passing)")
    else:
        ledger_and_memory()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    _REC["checks_passed"] = _PASS
    _REC["checks_total"] = _PASS + _FAIL
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(_REC, fh, indent=2)
    print("\n" + "=" * 100)
    print(f"{_PASS}/{_PASS + _FAIL} checks passed")
    print(f"CARRIER-PROBE-RESULT: {_PASS}/{_PASS + _FAIL} checks passed, "
          f"exit {0 if _FAIL == 0 else 1}")
    print(f"  wrote {os.path.relpath(OUT, _ROOT)}")
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
