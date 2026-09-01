"""WP3 — the unfactored AUGMENT target, its inventory, and its price.

Contract: `experiments/LOG.md`, *CONTRACT WP3* (2026-08-31), committed at
`35b8477` before `p2_augment.py` existed. Predictions A1-A6 appear below as
named checks.

TORCH, SO NOT IN PREFLIGHT
--------------------------
Upstream needs torch and the grader sandbox has none, so this suite runs under
the GPU venv and is deliberately NOT wired into `scripts/preflight.ps1` — the
same treatment every other upstream-driven probe in this repository gets. It
runs on CPU; nothing here needs a GPU, so nothing goes through `gpuq`.

Run: .venv-gpu\\Scripts\\python.exe local_sim/test_p2_augment.py
     .venv-gpu\\Scripts\\python.exe local_sim/test_p2_augment.py --live
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import numpy as np  # noqa: E402

import p2_augment as A  # noqa: E402

_FAILED = []
_TOTAL = [0]

#: Outside every registered block. Nothing here reads truth: every comparison
#: is between two PREDICTIONS.
SEEDS = (90000000701, 90000000702)


def check(name, ok, detail=""):
    _TOTAL[0] += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<64}{detail}")
    if not ok:
        _FAILED.append(name)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--live", action="store_true",
                    help="also price 1024x16 (~7 s on CPU)")
    a = ap.parse_args()
    import torch
    from mint_provenance import canonical_mint

    print("WP3 - the unfactored AUGMENT target\n")

    # ------------------------------------------------------------------ U --
    print("U  the reference is the pinned object, or nothing below means anything")
    v = A._ensure_upstream()
    check("U: upstream is rev 93d091a4 plus exactly the recorded patch",
          v["pass"] and v["rev"] == v["expected_rev"],
          "rev %s, diff md5 %s == %s"
          % (v["rev"][:12], v["changed_md5"][:8], v["recorded_md5"][:8]))
    check("U: the recorded patch is the ONLY difference",
          v["changed_lines"] == v["recorded_lines"],
          "%d changed lines, %d recorded" % (v["changed_lines"],
                                             v["recorded_lines"]))

    # ----------------------------------------------------------------- A1 --
    print("\nA1  the state inventory is {1:(n,), 2:(n,n), 3:(n,) r=1, 4:() r=2}")
    for n, L in ((16, 3), (32, 4), (64, 4)):
        w, _, _ = canonical_mint(n, L, SEEDS[0])
        out = A.run_augment(w, k_max=2, kind="AUGMENT", factor=False)
        for layer in range(L):
            inv = A.state_inventory(out, "act%d" % layer)
            want = {1: {"r": 0, "shape": (n,), "numel": n},
                    2: {"r": 0, "shape": (n, n), "numel": n * n},
                    3: {"r": 1, "shape": (n,), "numel": n},
                    4: {"r": 2, "shape": (), "numel": 1}}
            if layer == L - 1:
                check(f"A1: n={n} L={L} inventory at the LAST layer", inv == want,
                      "deg3 is a VECTOR and deg4 a SCALAR")
            elif inv != want:
                check(f"A1: n={n} L={L} inventory at layer {layer}", False,
                      str(inv))
        extra = n + 1
        base = n + n * n
        check(f"A1: n={n} AUGMENT adds {extra} numbers to K2's {base}",
              extra / base < 0.07,
              "%.3f%% state increase -- memory cannot kill this lane"
              % (100.0 * extra / base))

    # ----------------------------------------------------------------- A3 --
    # Run A3 before A2: if the augmentation changes nothing, the lane is dead
    # and the factor question is moot.
    print("\nA3  AUGMENT actually CHANGES the prediction (else the lane is dead)")
    for n, L in ((32, 4), (64, 4)):
        w, _, _ = canonical_mint(n, L, SEEDS[0])
        aug = A.degree1_means(A.run_augment(w, kind="AUGMENT", factor=False), L)
        base = A.degree1_means(A.run_augment(w, kind="BASE", factor=False), L)
        rel = float(np.abs(aug - base).max() / np.abs(base).max())
        relf = float(np.abs(aug[-1] - base[-1]).max() / np.abs(base[-1]).max())
        check(f"A3: n={n} AUGMENT differs from BASE (plain K2) by > 1e-6",
              rel > 1e-6,
              "max rel %.4e over all rows, %.4e on the SCORED row" % (rel, relf))

    # ----------------------------------------------------------------- A2 --
    # A2 AS CONTRACTED IS REFUTED, in the direction the contract anticipated:
    # "If they agree, the warning does not bite at this k_max and the cheaper
    # form is admissible." They agree EXACTLY, and upstream says why --
    # `kprop_harmonic.py:386`: "nonlin_kprop with factor=True called with
    # k_max<=2. Identical to unfactored." Factoring is only implemented for
    # k_max 3 or 4, so at k_max=2 the flag is inert.
    #
    # Checked as a MECHANISM, not a numerical coincidence: the numbers must
    # agree AND upstream's source must say they will. `factor=False` remains
    # what this slice freezes, because it is the form that stays correct if the
    # lane ever moves to k_max=3, where the warning DOES bite.
    print("\nA2  factor=True at k_max=2 -- REFUTED as contracted, and here is why")
    up_src = os.path.join(A.BRIDGE.DEFAULT_SRC, "mlp_kprop", "kprop_harmonic.py")
    with open(up_src, encoding="utf-8") as fh:
        up_text = fh.read()
    check("A2: upstream SOURCE says the flag is inert at k_max<=2",
          "factor=True called with k_max<=2. Identical to unfactored" in up_text,
          "kprop_harmonic.py:386 -- the mechanism, not a coincidence")
    check("A2: ...and factoring is only implemented for k_max 3 or 4",
          "Only supported for k_max=3 or 4" in up_text,
          "so k_max=2 cannot be affected by it")
    for n, L in ((32, 4), (64, 4)):
        w, _, _ = canonical_mint(n, L, SEEDS[0])
        unf = A.degree1_means(A.run_augment(w, kind="AUGMENT", factor=False), L)
        fac = A.degree1_means(A.run_augment(w, kind="AUGMENT", factor=True), L)
        rel = float(np.abs(unf - fac).max() / np.abs(unf).max())
        check(f"A2: n={n} the two agree EXACTLY at k_max=2", rel == 0.0,
              "max rel %.4e -- the warning does not bite here" % rel)
    check("A2: the frozen target is factor=False regardless",
          A.freeze_target(canonical_mint(32, 3, SEEDS[0])[0],
                          k_max=2)["factor"] is False,
          "the form that stays correct if the lane moves to k_max=3")

    # ----------------------------------------------------------------- A5 --
    print("\nA5  float32 and float64 agree on the scored quantity")
    for n, L in ((32, 4), (64, 4)):
        w, _, _ = canonical_mint(n, L, SEEDS[0])
        f64 = A.degree1_means(A.run_augment(w, kind="AUGMENT", factor=False,
                                            dtype=torch.float64), L)
        f32 = A.degree1_means(A.run_augment(w, kind="AUGMENT", factor=False,
                                            dtype=torch.float32), L)
        rel = float(np.abs(f64 - f32).max() / np.abs(f64).max())
        check(f"A5: n={n} float32 vs float64 agree to < 1e-5", rel < 1e-5,
              "max rel %.3e" % rel)

    # ----------------------------------------------------------------- A6 --
    print("\nA6  the frozen target is reproducible in a FRESH process")
    n, L = 32, 4
    w, _, _ = canonical_mint(n, L, SEEDS[0])
    d1 = A.target_digest(A.freeze_target(w, k_max=2))
    child = (
        "import sys,os;sys.path[:]=[p for p in sys.path if os.path.abspath(p or '.')"
        # The child must put UPSTREAM on its path before touching the flow.
        # `_ensure_upstream()` is what does that, and omitting it produced
        # "No module named 'mlp_kprop'" rather than a digest mismatch -- a
        # harness failure wearing a reproducibility failure's clothes.
        "!=%r]+[%r,%r];import p2_augment as A;A._ensure_upstream();"
        "from mint_provenance import canonical_mint;"
        "w,_,_=canonical_mint(%d,%d,%d);"
        "print(A.target_digest(A.freeze_target(w,k_max=2)))"
        % (_HERE, _ROOT, _HERE, n, L, SEEDS[0]))
    r = subprocess.run([sys.executable, "-c", child], capture_output=True,
                       text=True, cwd=_ROOT, errors="replace")
    d2 = (r.stdout or "").strip().splitlines()[-1] if r.stdout.strip() else ""
    check("A6: a fresh process reproduces the target digest", d1 == d2,
          "%s vs %s" % (d1[:16], d2[:16] or (r.stderr or "")[-40:]))
    d3 = A.target_digest(A.freeze_target(w, k_max=2))
    check("A6: ...and a second in-process build agrees", d1 == d3, d1[:16])

    # ----------------------------------------------------------------- A4 --
    print("\nA4  the price, as a RATIO inside one meter")
    widths = [32, 64, 128, 256] + ([1024] if a.live else [])
    depth = 16 if a.live else 4
    ratios = []
    for n in widths:
        L = 16 if (a.live and n == 1024) else 4
        w, _, _ = canonical_mint(n, L, SEEDS[0])
        pr = A.price_ratio(w, k_max=2)
        sc = A.screen(pr["ratio"])
        ratios.append((n, pr["ratio"]))
        check(f"A4: n={n:<5} ratio {pr['ratio']:.5f} <= 3x", pr["ratio"] <= 3.0,
              "excess %.2f%%  -> %s" % (100 * (pr["ratio"] - 1), sc["verdict"]))
    # A4 predicted "roughly width-independent". It is NOT, and it improves.
    excesses = [(n, r - 1.0) for n, r in ratios]
    halving = all(excesses[i + 1][1] < 0.62 * excesses[i][1]
                  for i in range(len(excesses) - 1))
    check("A4: the EXCESS falls ~1/n -- REFUTES 'width-independent', favourably",
          halving,
          "  ".join("n=%d:%.2f%%" % (n, 100 * e) for n, e in excesses))
    check("A4: ...so the augmented terms are Theta(n^2) on a Theta(n^3) base",
          excesses[-1][1] < excesses[0][1] / 4,
          "an n-vector and a scalar cannot add an n^3 term")

    # the screen must refuse a broken measurement rather than score it
    for bad in (float("nan"), 0.0, -1.0, float("inf")):
        try:
            A.screen(bad)
            check(f"A4: screen({bad}) REFUSES", False, "it returned a verdict")
        except ValueError:
            check(f"A4: screen({bad!r}) REFUSES rather than scoring KILL", True,
                  "a broken measurement is not an expensive arm")

    if a.live:
        live = [r for n, r in ratios if n == 1024][0]
        sc = A.screen(live)
        check("A4: at 1024 the projected multiplier stays PINNED at the floor",
              abs(sc["projected_multiplier"] - 0.1) < 1e-12,
              "projected %s FLOPs of a %s free zone"
              % (format(int(sc["projected_flopscope_flops"]), ","),
                 format(A.FLOOR_FLOPS, ",")))
        check("A4: ...and the verdict is PORT", sc["verdict"] == "PORT",
              "ratio %.5f vs a %.4f bar" % (live, A.AFFORDABLE_RATIO))
        check("A4: the projection is LABELLED a projection",
              "PROJECTION" in sc["PROJECTION_CAVEAT"]
              and "flopscope" in sc["PROJECTION_CAVEAT"],
              "upstream's meter is not flopscope")

    print()
    if _FAILED:
        print(f"FAILED {len(_FAILED)}: " + "; ".join(_FAILED[:6]))
        raise SystemExit(1)
    print(f"{_TOTAL[0]}/{_TOTAL[0]} checks passed")


if __name__ == "__main__":
    main()
