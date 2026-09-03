"""Does `estimator_wp15_k3_canary.py` compute the same thing as the lab flow?

The grader sandbox has no `local_sim/` and no `experiments/`, so the canary is a
STANDALONE copy of `p2_k3_fnp.k3_flow` with the frozen tables embedded as a
zlib/base64 blob. **A copy nobody checks drifts**, and this repository has a
worked example of exactly that: `p2_slice_cost.py::_third_order_layer` is a
second spelling of the same layer that is cost-correct and numerically wrong,
and its own docstring says so.

This asserts three things, and it is cheap enough to run on every touch of
either file:

1. **The embedded tables are the frozen tables.** Decompressed and compared
   field by field against `experiments/2026-09-02_wp14_fnp/k3_coefs_simple.json`,
   whose sha256 `p2_k3_fnp.COEFS_SHA256` pins.
2. **The two programs agree numerically**, on real canonically-minted weights,
   to float32 rounding, on every layer's mean -- not just the last, because the
   grader scores the final layer but a drift in an early layer would be masked.
3. **They agree on PRICE too**, to the FLOP. A copy that computes the same
   numbers by a different op sequence would invalidate the measurement in
   `layer_price_1024.json`, which is the only reason the canary exists.

WHAT THIS DOES *NOT* ESTABLISH, and it is the same gap the lab flow has:
neither program's values are checked against upstream at non-zero `kappa_3`
(`p2_k3_fnp.NOT_DONE` step 6, CLOSED 2026-09-03 by `p2_k3_parity.py`). Two
spellings agreeing with each other still says only the
transcription did not drift; it says nothing about whether the transcription was
right in the first place.

`# seed-ledger: no block named` -- `canonical_mint` is called directly on a fixed
seed for a determinism check; no block is drawn and nothing is selected.

Run:
    uv run python local_sim/p2_k3_candidate_parity.py
    uv run python local_sim/p2_k3_candidate_parity.py --width 256 --depth 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# APPEND, never insert: `local_sim/` holds numpy shims named `flopscope` and
# `whestbench`, and prepending it turns every counted FLOP below into fiction.
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

CANDIDATE = os.path.join(_ROOT, "estimator_wp15_k3_canary.py")
BUDGET = 2 ** 41


def _load_candidate():
    import importlib.util

    spec = importlib.util.spec_from_file_location("k3_canary", CANDIDATE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    import numpy as np

    import flopscope as flops
    import flopscope.numpy as fnp

    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1_000_921)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import p2_k3_fnp as lab
    from mint_provenance import canonical_mint

    lab._require_real_flopscope(flops)
    cand = _load_candidate()

    print("=" * 78)
    print("candidate parity -- estimator_wp15_k3_canary.py vs p2_k3_fnp.k3_flow")
    print("=" * 78)

    # ---- 1. the embedded tables ARE the frozen tables ----------------------
    frozen = lab.load_coefs()                     # refuses on sha256 drift
    embedded = cand._tables()
    same_tables = all(frozen[k] == embedded[k]
                      for k in ("terms", "wick", "pK_to_K", "lap2_d4"))
    print(f"  embedded tables identical to the frozen artifact: {same_tables}")
    print(f"    (frozen artifact pinned at sha256 {lab.COEFS_SHA256[:16]}...)")

    # ---- 2 & 3. same numbers, same price -----------------------------------
    w_np, _e, _s = canonical_mint(a.width, a.depth, a.seed)
    dt = fnp.float32

    audit = lab.DtypeAudit("float32")
    w = [fnp.array(np.asarray(x, dtype=np.float32)) for x in w_np]
    with flops.BudgetContext(flop_budget=10 * BUDGET, quiet=True) as c1:
        lab_out, _st = lab.k3_flow(fnp, flops, w, frozen, dt, audit)
    lab_flops, lab_ops = float(c1.flops_used), len(getattr(c1, "op_log", []) or [])

    class _MLP:
        pass

    m = _MLP()
    m.width = a.width
    m.weights = [fnp.array(np.asarray(x, dtype=np.float32)) for x in w_np]
    m.seed = a.seed
    est = cand.Estimator()
    est.setup(type("Ctx", (), {"width": a.width, "depth": a.depth})())
    with flops.BudgetContext(flop_budget=10 * BUDGET, quiet=True) as c2:
        can_out = est.predict(m, 10 * BUDGET)
    can_flops, can_ops = float(c2.flops_used), len(getattr(c2, "op_log", []) or [])

    L = np.asarray(lab_out, dtype=np.float64)
    C = np.asarray(can_out, dtype=np.float64)
    shape_ok = L.shape == C.shape == (a.depth, a.width)
    per_layer = [float(np.sqrt(((L[i] - C[i]) ** 2).mean()
                               / max((L[i] ** 2).mean(), 1e-300)))
                 for i in range(min(L.shape[0], C.shape[0]))]
    worst = max(per_layer) if per_layer else float("nan")

    print(f"  shape                       lab {L.shape}  candidate {C.shape}"
          f"  ok={shape_ok}")
    print(f"  worst per-layer rel-rms     {worst:.3e}   (over ALL {len(per_layer)}"
          " layers, not just the scored one)")
    print(f"  counted FLOPs               lab {lab_flops:,.0f}   candidate"
          f" {can_flops:,.0f}   delta {can_flops - lab_flops:+,.0f}")
    print(f"  flopscope operations        lab {lab_ops:,}   candidate {can_ops:,}")
    print(f"  finite output               {bool(np.isfinite(C).all())}")

    tol = 1e-6
    ok = (same_tables and shape_ok and worst <= tol
          and can_flops == lab_flops and bool(np.isfinite(C).all()))
    print(f"\n  tolerance {tol:.0e} relative RMS, and FLOPs must match EXACTLY")
    print(f"  VERDICT  {'PASS' if ok else 'FAIL'}")
    print()
    print("  This says the transcription did not drift. It does NOT say the")
    print("  transcription was right: neither program's values are checked")
    print("  against upstream at non-zero kappa_3 -- that is p2_k3_parity.py,")
    print("  which measured 1.692e-11 end to end on 8/8 P01 nets on 2026-09-03.")
    print("=" * 78)

    if a.out:
        doc = {"check": "estimator_wp15_k3_canary.py vs p2_k3_fnp.k3_flow",
               "width": a.width, "depth": a.depth, "seed": a.seed,
               "embedded_tables_identical": same_tables,
               "coefs_sha256": lab.COEFS_SHA256,
               "shape_ok": shape_ok, "per_layer_rel_rms": per_layer,
               "worst_rel_rms": worst, "tolerance": tol,
               "lab_flops": lab_flops, "candidate_flops": can_flops,
               "lab_ops": lab_ops, "candidate_ops": can_ops,
               "finite": bool(np.isfinite(C).all()), "pass": ok,
               "caveat": "agreement between two spellings; NOT parity against "
                         "upstream at non-zero kappa_3"}
        out = os.path.abspath(os.path.join(_ROOT, a.out))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1)
        print(f"  wrote {os.path.relpath(out, _ROOT)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
