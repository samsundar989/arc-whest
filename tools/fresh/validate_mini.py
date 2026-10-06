"""V2 acceptance test for the labeller: our N = 1e8 labels vs the organizers' N = 1e9.

For each organizer `mini` net labelled by `mc.py --mini`:

  * final layer: E[MSE(ours, organizer)] = floor_ours + floor_org, where
    floor_ours = mean_j var_j / 1e8 (our own variances) and
    floor_org  = avg_variance / 1e9 (the published column, final layer).
    A biased labeller shows up as a ratio > 1.
  * layer 1: the exact mean is ||W[:, j]|| / sqrt(2 pi) (closed form), so
    E[MSE(ours, exact)] = our layer-1 floor alone -- a test with NO organizer noise.
  * all layers: MSE(ours, organizer) per layer vs our floor + theirs (theirs per layer
    is unpublished; ours dominates 10:1, so the ratio is reported against ours x 1.1).

    python tools/fresh/validate_mini.py --n 10          (real .venv)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
import gen  # noqa: E402

sys.path.append(str(gen.ROOT / "local_sim"))  # APPEND: real packages first
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--N", type=float, default=1e8)
    a = ap.parse_args(argv)
    import real_data as rd  # noqa: PLC0415
    N = int(a.N)
    rows, r_fin, r_l1, r_all = [], [], [], []
    for i in range(a.n):
        w, truth, meta = rd.get(i, "mini", dtype=np.float32)
        p = gen.label_path("mini", int(meta["mlp_seed"]), N)
        with np.load(p) as d:
            assert str(d["weight_sha256"]) == gen.sha256_weights(w), "label/weights mismatch"
            means, var = np.array(d["means"]), np.array(d["var"])
        fl = var.mean(axis=1) / N
        fo = float(meta["avg_variance"]) / 1e9
        mse_fin = float(((means[-1] - truth[-1]) ** 2).mean())
        exact1 = np.linalg.norm(w[0].astype(np.float64), axis=0) / math.sqrt(2 * math.pi)
        mse_l1 = float(((means[0] - exact1) ** 2).mean())
        mse_layers = ((means - truth) ** 2).mean(axis=1)
        r_fin.append(mse_fin / (fl[-1] + fo))
        r_l1.append(mse_l1 / fl[0])
        r_all.append(float((mse_layers / (fl * 1.1)).mean()))
        rows.append(dict(i=i, name=meta["mlp_name"], mse_final=mse_fin,
                         floor_ours_final=float(fl[-1]), floor_org_final=fo,
                         ratio_final=r_fin[-1], mse_l1_vs_exact=mse_l1,
                         floor_ours_l1=float(fl[0]), ratio_l1=r_l1[-1],
                         ratio_all_layers=r_all[-1]))
        print(f"  mini[{i}] final MSE {mse_fin:.3e} vs floors {fl[-1]:.3e}+{fo:.3e} "
              f"ratio {r_fin[-1]:.3f} | L1 vs exact ratio {r_l1[-1]:.3f} | "
              f"all-layer ratio {r_all[-1]:.3f}", flush=True)

    def summ(x):
        x = np.asarray(x)
        se = x.std(ddof=1) / math.sqrt(len(x))
        return {"mean": float(x.mean()), "se": float(se)}

    out = {"N": N, "n": a.n, "rows": rows, "ratio_final": summ(r_fin),
           "ratio_l1_exact": summ(r_l1), "ratio_all_layers": summ(r_all),
           "pooled_final_mse": float(np.mean([r["mse_final"] for r in rows])),
           "pooled_floor_sum": float(np.mean([r["floor_ours_final"] + r["floor_org_final"]
                                              for r in rows]))}
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    op = gen.ROOT / "experiments" / "2026-10-06_laneV" / "mc_validation_mini.json"
    op.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {op}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
