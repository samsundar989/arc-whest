"""Skeptic follow-up (a), LABEL-FREE: live float32 closure layer vs float64 reference at 1024x16.

Captures v32 boundaries on the canonical network (seed 90_000_016_001, no block) at layers
3 and 8 with `same_state.capture_v32`, then runs the LIVE `_layer` of a closure arm on each
boundary (tower slices injected, W = I, hubs off) in float32 and in float64, and compares
mu_out / cov_out / g_out / S3 / K21 with `ref_step` on the arm's declared table in float64.
Also reports non-finite entries in the reference families (the live path zeroes them).

    uv run python local_sim/closure_audit/live_check.py --arm estimator_k3_v35_Q.py \
        --table local_sim/closure/closure_budget4.json --out <json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_LS = os.path.dirname(_HERE)
ROOT = os.path.dirname(_LS)
sys.path[:] = [q for q in sys.path if os.path.abspath(q or ".") not in (_HERE, _LS)] + [ROOT, _LS, _HERE]
os.environ.setdefault("SS_LAYERS", "3,8")

import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--table", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import flopscope.numpy as fnp
    import k4feed_oracle as K
    import same_state as SS
    import test_ref_step as T
    import ref_step
    import s6_build_closure as B
    from mint_provenance import canonical_mint

    w, es, _ = canonical_mint(SS.WIDTH, SS.DEPTH, SS.SEED)
    caps, _ = SS.capture_v32(w)
    out = {"artifact": "closure_audit.live_check", "label_free": True, "rows": []}
    n = SS.WIDTH
    off = 1.0 - np.eye(n)
    for arm, table in zip(a.arm, a.table):
        tb = [dict(output=list(k[0]), factors=list(k[1]), derivs=list(k[2]), coef=str(c))
              for k, c in B.load_table(os.path.join(ROOT, table)).items() if k[0] != (1, 2)]
        for l, bd in sorted(caps.items()):
            bd = dict(bd)
            ref = ref_step.step(bd, tb, want_families=True)
            nonfinite = int(sum((~np.isfinite(v)).sum() for v in ref["families"].values()))
            mu, sigma = bd["mu_pre"], np.sqrt(np.maximum(np.diag(bd["cov_pre"]), 1e-12))
            w1, w2 = ref_step.wick(1, 1, mu, sigma), ref_step.wick(2, 1, mu, sigma)
            C11 = bd["cov_pre"] * off
            fac1 = w1[:, None] * C11
            p111 = (w1[:, None] ** 2 * w1[None, :]) * bd["D21"] + ((fac1 * fac1) * w2[None, :]) * off
            for dname, dt in (("f64", fnp.float64), ("f32", fnp.float32)):
                live = T.run_live(K._load(os.path.join(ROOT, arm)), bd, dtype=dt, layer_index=l)
                K21 = live["R"] + (live["eb"][:, None] * C11) * w1[None, :] + p111
                row = {"arm": arm, "table": table, "layer": l, "dtype": dname,
                       "ref_nonfinite_family_entries": nonfinite,
                       "live_nonfinite": int(sum((~np.isfinite(live[k])).sum()
                                                 for k in ("mu_out", "cov_out", "g_out"))),
                       "rel": {"mu_out": T.rel(live["mu_out"], ref["mu_out"]),
                               "cov_out": T.rel(live["cov_out"], ref["cov_out"]),
                               "g_out": T.rel(live["g_out"], ref["g_out"]),
                               "S3": T.rel(live["sb"], ref["K"][(3,)] - w1 ** 3 * bd["D3"]),
                               "K21": T.rel(K21, ref["K"][(2, 1)])}}
                out["rows"].append(row)
                print(arm, "L%d" % l, dname, {k: "%.1e" % v for k, v in row["rel"].items()},
                      "nonfinite ref/live", nonfinite, row["live_nonfinite"], flush=True)
            with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
